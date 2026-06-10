"""Download media files from TalkBank ClassBank (non-TIMSS corpora)."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from pipeline.auth import create_session
from pipeline.config import DATASET_DIR, LIMITATION_SET_DIR, MEDIA_BASE_URL
from pipeline.download.helpers import (
    download_file,
    get_corpora_from_access_page,
    scrape_media_listing,
)

logger = logging.getLogger(__name__)


def get_media_file_list(corpus: str, session) -> list[str]:
    """Scrape media file listing for a corpus from TalkBank media server."""
    media_url = MEDIA_BASE_URL.format(corpus=corpus)
    logger.info(f"Fetching media listing: {media_url}")
    files = scrape_media_listing(media_url, session, max_depth=1)
    logger.info(f"  Found {len(files)} media files for {corpus}")
    return files


def download_corpus_media(
    corpus: str,
    output_dir: Path,
    session,
    workers: int = 1,
) -> int:
    """Download all media files for a corpus with configurable parallelism."""
    media_dir = output_dir / "media" / corpus
    media_dir.mkdir(parents=True, exist_ok=True)

    files = get_media_file_list(corpus, session)
    if not files:
        return 0

    # Build skip set from data-limitation-set
    limitation_dir = LIMITATION_SET_DIR / "media" / corpus
    skip_files: set[str] = set()
    if limitation_dir.exists():
        skip_files = {
            f.name for f in limitation_dir.iterdir()
            if f.is_file() and f.stat().st_size > 100
        }
        if skip_files:
            logger.info(f"  Skipping {len(skip_files)} files already in data-limitation-set")

    # Clean up 0-byte files from previous failed attempts
    for f in media_dir.iterdir():
        if f.is_file() and f.stat().st_size == 0:
            f.unlink()

    # Clean up orphaned .tmp files
    if skip_files:
        for f in media_dir.iterdir():
            if f.is_file() and f.suffix == ".tmp":
                original_name = f.stem
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
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"  {corpus}", leave=False
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


def download_all_media(
    output_dir: Path | None = None,
    corpora: list[str] | None = None,
    workers: int = 1,
) -> int:
    """Download media for all (or specified) corpora.

    Args:
        output_dir: Base dataset directory. Defaults to DATASET_DIR.
        corpora: List of corpus names. Defaults to auto-discovered from access page.
        workers: Number of parallel download threads.

    Returns:
        Total number of newly downloaded files.
    """
    if output_dir is None:
        output_dir = DATASET_DIR

    if corpora is None:
        corpora = get_corpora_from_access_page()

    logger.info(f"Output: {output_dir}")
    logger.info(f"Corpora: {len(corpora)}")
    logger.info(f"Workers: {workers}")

    session = create_session()

    total_downloaded = 0
    for corpus in corpora:
        logger.info(f"--- {corpus} ---")
        total_downloaded += download_corpus_media(corpus, output_dir, session, workers=workers)
        time.sleep(1)

    # Summary
    total_files = sum(
        1 for _ in (output_dir / "media").rglob("*")
        if _.is_file() and _.stat().st_size > 100
    ) if (output_dir / "media").exists() else 0
    logger.info(f"Done: {total_files} total media files on disk")

    return total_downloaded
