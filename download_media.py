"""
TalkBank ClassBank - Media Downloader
======================================

Downloads audio/video recordings from TalkBank's ClassBank collection.

Media files are hosted at:
    https://media.talkbank.org/class/{corpus}/

Requires a free TalkBank account (register at https://class.talkbank.org).

Usage:
    python download_media.py [--output-dir ./dataset] [--corpora APT,Bradford] [--parallel 5]
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEDIA_EXTENSIONS = (".mp4", ".mp3", ".wav", ".m4v", ".m4a", ".avi", ".mov", ".m4b")

# Corpora ordered by expected size (smallest first)
ENGLISH_CORPORA = [
    "Bradford", "Horowitz", "Crowley", "Roth", "Warren", "Rahm",
    "Stevens", "Frederiksen", "Moschkovich", "JLS", "Looney", "Person",
    "CogInst", "Graesser", "MacWhinney", "DISPEL", "CarlaJim", "Curtis",
    "APT", "TIMSS-Math", "TIMSS-Science",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_media.log")],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def create_session(email=None, password=None):
    """Authenticate with TalkBank and return session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json",
    })

    if email is None:
        email = os.environ.get("TALKBANK_EMAIL") or input("TalkBank email: ").strip()
    if password is None:
        password = os.environ.get("TALKBANK_PASSWORD")
        if not password:
            import getpass
            password = getpass.getpass("TalkBank password: ")

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


def get_media_file_list(corpus, session):
    """Scrape media file listing for a corpus from TalkBank media server."""
    media_url = f"https://media.talkbank.org:443/class/{corpus}/"
    logger.info(f"Fetching media listing: {media_url}")

    try:
        resp = session.get(media_url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"  Cannot access media for {corpus}: HTTP {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        files = []
        subdirs = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "?f=save" in href:
                continue

            if any(href.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
                if href.startswith("http"):
                    files.append(href)
                else:
                    files.append(f"https://media.talkbank.org:443/class/{corpus}/{href}")
            elif (href.endswith("/") and "class" in href
                  and "Parent" not in link.get_text() and href != media_url):
                if href.startswith("http"):
                    subdirs.append(href)

        # Recursively check subdirectories
        for subdir_url in subdirs:
            try:
                time.sleep(0.3)
                sub_resp = session.get(subdir_url, timeout=30)
                if sub_resp.status_code == 200:
                    sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
                    for sub_link in sub_soup.find_all("a", href=True):
                        sub_href = sub_link["href"]
                        if "?f=save" in sub_href:
                            continue
                        if any(sub_href.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
                            if sub_href.startswith("http"):
                                files.append(sub_href)
                            else:
                                files.append(f"{subdir_url}{sub_href}")
            except Exception:
                pass

        logger.info(f"  Found {len(files)} media files for {corpus}")
        return files

    except Exception as e:
        logger.error(f"  Error fetching media listing for {corpus}: {e}")
        return []


# ---------------------------------------------------------------------------
# Download Logic
# ---------------------------------------------------------------------------


def download_file(session, url, output_path):
    """Download a single media file using range requests."""
    try:
        if output_path.exists() and output_path.stat().st_size > 100:
            return True, output_path.name, "skipped"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file size
        resp = session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        if resp.status_code not in (200, 206):
            return False, output_path.name, f"HTTP {resp.status_code}"

        content_range = resp.headers.get("content-range", "")
        total_size = int(content_range.split("/")[-1]) if "/" in content_range else 0
        if total_size < 100:
            return False, output_path.name, "empty file on server"

        # Download to tmp then rename
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        resp = session.get(
            url,
            headers={"Range": f"bytes=0-{total_size - 1}"},
            timeout=600,
            stream=True,
        )
        if resp.status_code not in (200, 206):
            return False, output_path.name, f"HTTP {resp.status_code}"

        written = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)

        if written > 100:
            if output_path.exists():
                output_path.unlink()
            tmp_path.rename(output_path)
            size_mb = written / (1024 * 1024)
            return True, output_path.name, f"{size_mb:.1f} MB"
        else:
            tmp_path.unlink(missing_ok=True)
            return False, output_path.name, "empty download"

    except Exception as e:
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False, output_path.name, str(e)[:60]


def download_corpus_media(corpus, output_dir, session, max_workers=5):
    """Download all media files for a corpus with parallel workers."""
    media_dir = output_dir / "media" / corpus
    media_dir.mkdir(parents=True, exist_ok=True)

    files = get_media_file_list(corpus, session)
    if not files:
        return 0

    # Clean up 0-byte files from previous failed attempts
    for f in media_dir.iterdir():
        if f.is_file() and f.stat().st_size == 0:
            f.unlink()

    downloaded = 0
    skipped = 0
    failed = 0

    tasks = []
    for url in files:
        filename = url.split("/")[-1].split("?")[0]
        output_path = media_dir / filename
        tasks.append((url, output_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, session, url, path): (url, path)
            for url, path in tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc=f"  {corpus}", leave=False
        ):
            success, name, info = future.result()
            if success:
                if info == "skipped":
                    skipped += 1
                else:
                    downloaded += 1
                    logger.info(f"    {name} ({info})")
            else:
                failed += 1
                logger.warning(f"    FAILED: {name} - {info}")

    logger.info(f"  [{corpus}] {downloaded} new, {skipped} skipped, {failed} failed")
    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download media files from TalkBank ClassBank."
    )
    parser.add_argument("--output-dir", type=str, default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--corpora", type=str, default=None,
                        help="Comma-separated corpora to download (default: all)")
    parser.add_argument("--email", type=str, default=None,
                        help="TalkBank email (or set TALKBANK_EMAIL env var)")
    parser.add_argument("--password", type=str, default=None,
                        help="TalkBank password (or set TALKBANK_PASSWORD env var)")
    parser.add_argument("--parallel", type=int, default=5,
                        help="Number of parallel downloads (default: 5)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    corpora = [c.strip() for c in args.corpora.split(",")] if args.corpora else ENGLISH_CORPORA

    logger.info(f"Output: {output_dir}")
    logger.info(f"Corpora: {len(corpora)}")
    logger.info(f"Parallel downloads: {args.parallel}")

    session = create_session(args.email, args.password)

    total_downloaded = 0
    for corpus in corpora:
        logger.info(f"--- {corpus} ---")
        total_downloaded += download_corpus_media(corpus, output_dir, session, args.parallel)
        time.sleep(1)

    # Summary
    total_files = sum(
        1 for _ in (output_dir / "media").rglob("*")
        if _.is_file() and _.stat().st_size > 100
    ) if (output_dir / "media").exists() else 0
    logger.info(f"Done: {total_files} total media files on disk")


if __name__ == "__main__":
    main()
