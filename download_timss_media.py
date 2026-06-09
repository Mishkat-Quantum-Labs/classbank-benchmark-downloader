"""
TalkBank ClassBank - TIMSS Media Downloader
=============================================

Downloads video recordings from TalkBank's TIMSS-Math and TIMSS-Science collections.

TIMSS data is organized by country:
    https://media.talkbank.org/class/TIMSS-Math/{country}/
    https://media.talkbank.org/class/TIMSS-Science/{country}/

Requires a free TalkBank account (register at https://class.talkbank.org).
Credentials are loaded from a .env file (see .env.example).

Usage:
    python download_timss_media.py [--output-dir ./dataset] [--subjects Math,Science] [--countries USA]
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEDIA_EXTENSIONS = (".mp4", ".mp3", ".wav", ".m4v", ".m4a", ".avi", ".mov", ".m4b")

TIMSS_SUBJECTS = ["Math", "Science"]

TIMSS_COUNTRIES = {
    "Math": ["USA"],
    "Science": ["USA"],
}

MEDIA_BASE_URL = "https://media.talkbank.org:443/class/TIMSS-{subject}/{country}/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_timss_media.log")],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def create_session():
    """Authenticate with TalkBank and return session. Reads credentials from .env."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=20, pool_maxsize=20, max_retries=3
    )
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json",
    })

    email = os.environ.get("TALKBANK_EMAIL")
    password = os.environ.get("TALKBANK_PASSWORD")

    if not email or not password:
        logger.error("Missing credentials. Set TALKBANK_EMAIL and TALKBANK_PASSWORD in .env file.")
        logger.error("See .env.example for the required format.")
        sys.exit(1)

    resp = session.post(
        "https://sla2.talkbank.org:443/logInUser",
        data=json.dumps({"email": email, "pswd": password}),
        timeout=30,
    )

    if resp.status_code == 200:
        data = resp.json()
        if data.get("success"):
            session.headers.pop("Content-Type", None)
            logger.info("Authenticated with TalkBank.")
            return session

    logger.error("Authentication failed. Register at https://class.talkbank.org")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Media File Discovery
# ---------------------------------------------------------------------------


def get_media_file_list(subject, country, session):
    """Scrape media file listing for a TIMSS subject/country."""
    media_url = MEDIA_BASE_URL.format(subject=subject, country=country)
    logger.info(f"Fetching media listing: {media_url}")

    def is_media_file(href):
        return any(href.lower().endswith(ext) for ext in MEDIA_EXTENSIONS)

    def normalize_url(url):
        parts = url.split("://", 1)
        if len(parts) == 2:
            return parts[0] + "://" + parts[1].replace("//", "/")
        return url

    def scrape_directory(dir_url, depth=0):
        if depth > 3:
            return []

        files = []
        try:
            resp = session.get(dir_url, timeout=30)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            subdirs = []

            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "?f=save" in href:
                    continue
                if "Parent" in link.get_text() or href == dir_url:
                    continue

                if is_media_file(href):
                    url = normalize_url(href) if href.startswith("http") else normalize_url(dir_url.rstrip("/") + "/" + href.lstrip("/"))
                    files.append(url)
                elif href.startswith("http") and not is_media_file(href):
                    base = media_url.rstrip("/")
                    if href.startswith(base) and href != dir_url.rstrip("/"):
                        subdirs.append(normalize_url(href.rstrip("/") + "/"))

            # Recursively check subdirectories
            for subdir_url in subdirs:
                time.sleep(0.2)
                files.extend(scrape_directory(subdir_url, depth + 1))

        except Exception as e:
            if depth == 0:
                logger.error(f"  Error fetching {dir_url}: {e}")

        return files

    files = scrape_directory(media_url)
    logger.info(f"  Found {len(files)} media files for TIMSS-{subject}/{country}")
    return files


# ---------------------------------------------------------------------------
# Download Logic
# ---------------------------------------------------------------------------


def download_file(session, url, output_path):
    """Download a single media file with resume support."""
    try:
        if output_path.exists() and output_path.stat().st_size > 100:
            return True, output_path.name, "skipped"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Probe to get total file size
        resp = session.get(url, timeout=30)
        if resp.status_code not in (200, 206):
            return False, output_path.name, f"HTTP {resp.status_code}"

        content_range = resp.headers.get("content-range", "")
        total_size = int(content_range.split("/")[-1]) if "/" in content_range else 0
        if total_size < 100:
            return False, output_path.name, "empty file on server"

        # Resume support
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        start_byte = 0
        if tmp_path.exists():
            start_byte = tmp_path.stat().st_size
            if start_byte >= total_size:
                if output_path.exists():
                    output_path.unlink()
                tmp_path.rename(output_path)
                size_mb = start_byte / (1024 * 1024)
                return True, output_path.name, f"{size_mb:.1f} MB (resumed)"

        # Download with range header
        headers = {"Range": f"bytes={start_byte}-{total_size - 1}"}
        resp = session.get(url, headers=headers, timeout=600, stream=True)
        if resp.status_code not in (200, 206):
            return False, output_path.name, f"HTTP {resp.status_code}"

        mode = "ab" if start_byte > 0 else "wb"
        written = start_byte
        with open(tmp_path, mode) as f:
            for chunk in resp.iter_content(1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    written += len(chunk)

        if written > 100:
            if output_path.exists():
                output_path.unlink()
            tmp_path.rename(output_path)
            size_mb = written / (1024 * 1024)
            resumed = " (resumed)" if start_byte > 0 else ""
            return True, output_path.name, f"{size_mb:.1f} MB{resumed}"
        else:
            tmp_path.unlink(missing_ok=True)
            return False, output_path.name, "empty download"

    except Exception as e:
        return False, output_path.name, str(e)[:80]


def download_country_media(subject, country, output_dir, session):
    """Download all media files for a TIMSS subject/country."""
    media_dir = output_dir / "media" / f"TIMSS-{subject}"
    media_dir.mkdir(parents=True, exist_ok=True)

    files = get_media_file_list(subject, country, session)
    if not files:
        return 0

    # Clean up 0-byte files from previous failed attempts
    for f in media_dir.iterdir():
        if f.is_file() and f.stat().st_size == 0:
            f.unlink()

    downloaded = 0
    skipped = 0
    failed = 0

    for url in tqdm(files, desc=f"  TIMSS-{subject}/{country}", leave=False):
        filename = url.split("/")[-1].split("?")[0]
        output_path = media_dir / filename
        success, name, info = download_file(session, url, output_path)
        if success:
            if info == "skipped":
                skipped += 1
            else:
                downloaded += 1
                logger.info(f"    {name} ({info})")
        else:
            failed += 1
            logger.warning(f"    FAILED: {name} - {info}")

    logger.info(f"  [TIMSS-{subject}/{country}] {downloaded} new, {skipped} skipped, {failed} failed")
    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download TIMSS media files from TalkBank ClassBank."
    )
    parser.add_argument("--output-dir", type=str, default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated: Math,Science (default: both)")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated countries (default: all per subject)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else TIMSS_SUBJECTS

    session = create_session()

    total_downloaded = 0
    for subject in subjects:
        if subject not in TIMSS_COUNTRIES:
            logger.warning(f"Unknown TIMSS subject: {subject}. Use Math or Science.")
            continue

        available_countries = TIMSS_COUNTRIES[subject]
        if args.countries:
            countries = [c.strip() for c in args.countries.split(",")]
            countries = [c for c in countries if c in available_countries]
        else:
            countries = available_countries

        logger.info(f"=== TIMSS-{subject} ===")
        logger.info(f"Countries: {', '.join(countries)}")

        for country in countries:
            logger.info(f"--- TIMSS-{subject}/{country} ---")
            total_downloaded += download_country_media(
                subject, country, output_dir, session
            )
            time.sleep(1)

    # Summary
    timss_media_dir = output_dir / "media"
    total_files = 0
    for subject in TIMSS_SUBJECTS:
        subject_dir = timss_media_dir / f"TIMSS-{subject}"
        if subject_dir.exists():
            total_files += sum(
                1 for _ in subject_dir.rglob("*")
                if _.is_file() and _.stat().st_size > 100
            )
    logger.info(f"Done: {total_files} total TIMSS media files on disk")


if __name__ == "__main__":
    main()
