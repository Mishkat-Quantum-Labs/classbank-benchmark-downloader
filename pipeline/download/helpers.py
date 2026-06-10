"""Shared download helpers — file download with resume, URL normalization."""

import logging
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from pipeline.config import (
    MEDIA_EXTENSIONS,
    TALKBANK_ACCESS_PAGE,
    MEDIA_CORPORA,
    CORPUS_NAME_MAP,
    SKIP_CORPORA,
)

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """Remove duplicate slashes in the path portion of a URL."""
    parts = url.split("://", 1)
    if len(parts) == 2:
        return parts[0] + "://" + parts[1].replace("//", "/")
    return url


def is_media_file(href: str) -> bool:
    """Check if a URL/href ends with a known media extension."""
    return any(href.lower().endswith(ext) for ext in MEDIA_EXTENSIONS)


def get_corpora_from_access_page() -> list[str]:
    """Scrape the ClassBank access page for available corpora.

    Falls back to the MEDIA_CORPORA constant if scraping fails.
    """
    logger.info(f"Fetching corpus list from: {TALKBANK_ACCESS_PAGE}")

    try:
        resp = requests.get(
            TALKBANK_ACCESS_PAGE,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        if resp.status_code != 200:
            logger.warning(
                f"Could not fetch access page (HTTP {resp.status_code}), using fallback list."
            )
            return MEDIA_CORPORA

        soup = BeautifulSoup(resp.text, "html.parser")
        corpora = []

        for link in soup.find_all("a"):
            href = link.get("href", "")
            if href.endswith(".html") and "/" not in href and href != "index.html":
                corpus_name = link.get_text(strip=True)
                if not corpus_name:
                    continue

                folder_name = CORPUS_NAME_MAP.get(corpus_name, corpus_name)
                if folder_name in SKIP_CORPORA:
                    continue
                corpora.append(folder_name)

        if corpora:
            logger.info(f"  Found {len(corpora)} corpora on access page")
            return corpora
        else:
            logger.warning("  No corpora found on access page, using fallback list.")
            return MEDIA_CORPORA

    except Exception as e:
        logger.warning(f"  Error fetching access page: {e}. Using fallback list.")
        return MEDIA_CORPORA


def download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    skip_files: Optional[set[str]] = None,
) -> tuple[bool, str, str]:
    """Download a single file with resume support.

    Args:
        session: Authenticated requests session.
        url: URL to download from.
        output_path: Local path to save the file.
        skip_files: Optional set of filenames to skip.

    Returns:
        Tuple of (success, filename, info_message).
    """
    try:
        if output_path.exists() and output_path.stat().st_size > 100:
            return True, output_path.name, "skipped"

        if skip_files and output_path.name in skip_files:
            return True, output_path.name, "skipped"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Probe file size via HEAD
        resp = session.head(url, timeout=30, allow_redirects=True)
        if resp.status_code not in (200, 206):
            resp = session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
            if resp.status_code not in (200, 206):
                return False, output_path.name, f"HTTP {resp.status_code}"

        # Determine total file size
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
            for chunk in resp.iter_content(1024 * 1024):
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


def scrape_media_listing(
    listing_url: str,
    session: requests.Session,
    max_depth: int = 1,
) -> list[str]:
    """Scrape a TalkBank media directory listing for downloadable file URLs.

    Args:
        listing_url: The directory URL to scrape.
        session: Authenticated session.
        max_depth: Maximum recursion depth for subdirectories.

    Returns:
        List of direct download URLs.
    """

    def _scrape(dir_url: str, depth: int) -> list[str]:
        if depth > max_depth:
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
                    url = (
                        normalize_url(href)
                        if href.startswith("http")
                        else normalize_url(dir_url.rstrip("/") + "/" + href.lstrip("/"))
                    )
                    files.append(url)
                elif href.startswith("http") and not is_media_file(href):
                    base = listing_url.rstrip("/")
                    if href.startswith(base) and href != dir_url.rstrip("/"):
                        subdirs.append(normalize_url(href.rstrip("/") + "/"))

            for subdir_url in subdirs:
                time.sleep(0.2)
                files.extend(_scrape(subdir_url, depth + 1))

        except Exception as e:
            if depth == 0:
                logger.error(f"  Error fetching {dir_url}: {e}")

        return files

    return _scrape(listing_url, 0)
