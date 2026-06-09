"""
TalkBank ClassBank - Media Downloader
======================================

Downloads audio/video recordings from TalkBank's ClassBank collection.

Media files are hosted at:
    https://media.talkbank.org/class/{corpus}/

Requires a free TalkBank account (register at https://class.talkbank.org).
Credentials are loaded from a .env file (see .env.example).

Usage:
    python download_media.py [--output-dir ./dataset] [--corpora APT,Bradford]
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
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEDIA_EXTENSIONS = (".mp4", ".mp3", ".wav", ".m4v", ".m4a", ".avi", ".mov", ".m4b")

# Corpora ordered by expected size (smallest first)
# TIMSS-Math and TIMSS-Science handled by download_timss_media.py
ENGLISH_CORPORA = [
    "Bradford", "Horowitz", "Crowley", "Roth", "Warren", "Rahm",
    "Stevens", "Frederiksen", "Moschkovich", "JLS", "Looney", "Person",
    "CogInst", "Graesser", "MacWhinney", "DISPEL", "CarlaJim", "Curtis",
    "APT",
]

# Mapping from access page names to actual media folder names
CORPUS_NAME_MAP = {
    "Cognition&Instruction": "CogInst",
    "Greeno/VanDeSande": "CarlaJim",
}

# TIMSS handled separately by download_timss_media.py
SKIP_CORPORA = ["TIMSS-Math", "TIMSS-Science"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_media.log")],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus Discovery from Access Page
# ---------------------------------------------------------------------------


def get_corpora_from_access_page():
    """Scrape the ClassBank access page to get the list of available corpora."""
    url = "https://talkbank.org/class/access/index.html"
    logger.info(f"Fetching corpus list from: {url}")

    try:
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if resp.status_code != 200:
            logger.warning(f"Could not fetch access page (HTTP {resp.status_code}), using fallback list.")
            return ENGLISH_CORPORA

        soup = BeautifulSoup(resp.text, "html.parser")
        corpora = []

        # Each corpus row has a link like <a href="APT.html">APT</a>
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if href.endswith(".html") and "/" not in href and href != "index.html":
                corpus_name = link.get_text(strip=True)
                if not corpus_name:
                    continue

                # Map display names to folder names
                folder_name = CORPUS_NAME_MAP.get(corpus_name, corpus_name)

                # Skip TIMSS (handled by separate script)
                if folder_name in SKIP_CORPORA:
                    continue

                corpora.append(folder_name)

        if corpora:
            logger.info(f"  Found {len(corpora)} corpora on access page")
            return corpora
        else:
            logger.warning("  No corpora found on access page, using fallback list.")
            return ENGLISH_CORPORA

    except Exception as e:
        logger.warning(f"  Error fetching access page: {e}. Using fallback list.")
        return ENGLISH_CORPORA


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


def get_media_file_list(corpus, session):
    """Scrape media file listing for a corpus from TalkBank media server."""
    media_url = f"https://media.talkbank.org:443/class/{corpus}/"
    logger.info(f"Fetching media listing: {media_url}")

    def is_media_file(href):
        return any(href.lower().endswith(ext) for ext in MEDIA_EXTENSIONS)

    def normalize_url(url):
        parts = url.split("://", 1)
        if len(parts) == 2:
            return parts[0] + "://" + parts[1].replace("//", "/")
        return url

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
            if "Parent" in link.get_text() or href == media_url:
                continue

            if is_media_file(href):
                url = normalize_url(href) if href.startswith("http") else normalize_url(media_url.rstrip("/") + "/" + href.lstrip("/"))
                files.append(url)
            elif href.startswith("http") and not is_media_file(href):
                base = f"https://media.talkbank.org:443/class/{corpus}"
                if href.startswith(base) and href != media_url.rstrip("/"):
                    subdirs.append(normalize_url(href.rstrip("/") + "/"))

        # Recursively check subdirectories
        for subdir_url in subdirs:
            try:
                time.sleep(0.2)
                sub_resp = session.get(subdir_url, timeout=30)
                if sub_resp.status_code == 200:
                    sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
                    for sub_link in sub_soup.find_all("a", href=True):
                        sub_href = sub_link["href"]
                        if "?f=save" in sub_href:
                            continue
                        if is_media_file(sub_href):
                            url = normalize_url(sub_href) if sub_href.startswith("http") else normalize_url(subdir_url.rstrip("/") + "/" + sub_href.lstrip("/"))
                            files.append(url)
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


def download_file(session, url, output_path, skip_files=None):
    """Download a single media file with resume support.
    
    Args:
        skip_files: Optional set of filenames to skip (e.g. files moved to data-limitation-set)
    """
    try:
        if output_path.exists() and output_path.stat().st_size > 100:
            return True, output_path.name, "skipped"

        # Skip files that exist in data-limitation-set
        if skip_files and output_path.name in skip_files:
            return True, output_path.name, "skipped"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use HEAD request to probe file size (avoids downloading the whole file)
        resp = session.head(url, timeout=30, allow_redirects=True)
        if resp.status_code not in (200, 206):
            # Some servers don't support HEAD, fall back to Range: bytes=0-0
            resp = session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
            if resp.status_code not in (200, 206):
                return False, output_path.name, f"HTTP {resp.status_code}"

        # Determine total file size from headers
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range:
            total_size = int(content_range.split("/")[-1])
        elif resp.headers.get("content-length"):
            total_size = int(resp.headers["content-length"])
        else:
            total_size = 0

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


def download_corpus_media(corpus, output_dir, session, workers=1):
    """Download all media files for a corpus with configurable parallelism."""
    media_dir = output_dir / "media" / corpus
    media_dir.mkdir(parents=True, exist_ok=True)

    files = get_media_file_list(corpus, session)
    if not files:
        return 0

    # Build set of files already in data-limitation-set (moved out due to quality issues)
    # These should not be re-downloaded
    limitation_dir = output_dir.parent / "data-limitation-set" / "dataset" / "media" / corpus
    skip_files = set()
    if limitation_dir.exists():
        skip_files = {f.name for f in limitation_dir.iterdir() if f.is_file() and f.stat().st_size > 100}
        if skip_files:
            logger.info(f"  Skipping {len(skip_files)} files already in data-limitation-set")

    # Clean up 0-byte files from previous failed attempts
    for f in media_dir.iterdir():
        if f.is_file() and f.stat().st_size == 0:
            f.unlink()

    # Clean up .tmp files for files that are in data-limitation-set
    if skip_files:
        for f in media_dir.iterdir():
            if f.is_file() and f.suffix == ".tmp":
                # e.g. "01-Character.mp4.tmp" -> check if "01-Character.mp4" is in skip_files
                original_name = f.stem  # removes .tmp, leaves "01-Character.mp4"
                if original_name in skip_files:
                    f.unlink()
                    logger.info(f"  Cleaned up orphaned tmp: {f.name}")

    downloaded = 0
    skipped = 0
    failed = 0

    if workers <= 1:
        for url in tqdm(files, desc=f"  {corpus}", leave=False):
            filename = url.split("/")[-1].split("?")[0]
            output_path = media_dir / filename
            success, name, info = download_file(session, url, output_path, skip_files)
            if success:
                if info == "skipped":
                    skipped += 1
                else:
                    downloaded += 1
                    logger.info(f"    {name} ({info})")
            else:
                failed += 1
                logger.warning(f"    FAILED: {name} - {info}")
    else:
        tasks = []
        for url in files:
            filename = url.split("/")[-1].split("?")[0]
            output_path = media_dir / filename
            tasks.append((url, output_path))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_file, session, url, path, skip_files): (url, path)
                for url, path in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {corpus}", leave=False):
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel download threads (default: 1)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    corpora = [c.strip() for c in args.corpora.split(",")] if args.corpora else get_corpora_from_access_page()

    logger.info(f"Output: {output_dir}")
    logger.info(f"Corpora: {len(corpora)}")
    logger.info(f"Workers: {args.workers}")

    session = create_session()

    total_downloaded = 0
    for corpus in corpora:
        logger.info(f"--- {corpus} ---")
        total_downloaded += download_corpus_media(corpus, output_dir, session, workers=args.workers)
        time.sleep(1)

    # Summary
    total_files = sum(
        1 for _ in (output_dir / "media").rglob("*")
        if _.is_file() and _.stat().st_size > 100
    ) if (output_dir / "media").exists() else 0
    logger.info(f"Done: {total_files} total media files on disk")


if __name__ == "__main__":
    main()
