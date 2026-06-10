"""Download TIMSS media files from TalkBank ClassBank."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from pipeline.auth import create_session
from pipeline.config import (
    DATASET_DIR,
    LIMITATION_SET_DIR,
    TIMSS_COUNTRIES,
    TIMSS_MEDIA_URL,
    TIMSS_SUBJECTS,
)
from pipeline.download.helpers import download_file, scrape_media_listing

logger = logging.getLogger(__name__)


def get_timss_media_file_list(subject: str, country: str, session) -> list[str]:
    """Scrape media file listing for a TIMSS subject/country."""
    media_url = TIMSS_MEDIA_URL.format(subject=subject, country=country)
    logger.info(f"Fetching media listing: {media_url}")
    files = scrape_media_listing(media_url, session, max_depth=3)
    logger.info(f"  Found {len(files)} media files for TIMSS-{subject}/{country}")
    return files


def download_country_media(
    subject: str,
    country: str,
    output_dir: Path,
    session,
    workers: int = 1,
) -> int:
    """Download all media files for a TIMSS subject/country."""
    media_dir = output_dir / "media" / f"TIMSS-{subject}"
    media_dir.mkdir(parents=True, exist_ok=True)

    files = get_timss_media_file_list(subject, country, session)
    if not files:
        return 0

    # Build skip set from data-limitation-set
    limitation_dir = LIMITATION_SET_DIR / "media" / f"TIMSS-{subject}"
    skip_files: set[str] = set()
    if limitation_dir.exists():
        skip_files = {
            f.name for f in limitation_dir.iterdir()
            if f.is_file() and f.stat().st_size > 100
        }
        if skip_files:
            logger.info(f"  Skipping {len(skip_files)} files already in data-limitation-set")

    # Clean up 0-byte files
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
        for url in tqdm(files, desc=f"  TIMSS-{subject}/{country}", leave=False):
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
                as_completed(futures), total=len(futures),
                desc=f"  TIMSS-{subject}/{country}", leave=False,
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

    logger.info(
        f"  [TIMSS-{subject}/{country}] {downloaded} new, {skipped} skipped, {failed} failed"
    )
    return downloaded


def download_all_timss_media(
    output_dir: Path | None = None,
    subjects: list[str] | None = None,
    countries: list[str] | None = None,
    workers: int = 1,
) -> int:
    """Download TIMSS media for all (or specified) subjects/countries.

    Args:
        output_dir: Base dataset directory. Defaults to DATASET_DIR.
        subjects: List of TIMSS subjects (Math, Science). Defaults to both.
        countries: List of countries. Defaults to all per subject.
        workers: Number of parallel download threads.

    Returns:
        Total number of newly downloaded files.
    """
    if output_dir is None:
        output_dir = DATASET_DIR

    if subjects is None:
        subjects = TIMSS_SUBJECTS

    session = create_session()

    total_downloaded = 0
    for subject in subjects:
        if subject not in TIMSS_COUNTRIES:
            logger.warning(f"Unknown TIMSS subject: {subject}. Use Math or Science.")
            continue

        available_countries = TIMSS_COUNTRIES[subject]
        if countries:
            filtered = [c for c in countries if c in available_countries]
        else:
            filtered = available_countries

        logger.info(f"=== TIMSS-{subject} ===")
        logger.info(f"Countries: {', '.join(filtered)}")
        logger.info(f"Workers: {workers}")

        for country in filtered:
            logger.info(f"--- TIMSS-{subject}/{country} ---")
            total_downloaded += download_country_media(
                subject, country, output_dir, session, workers=workers
            )
            time.sleep(1)

    # Summary
    total_files = 0
    for subject in TIMSS_SUBJECTS:
        subject_dir = output_dir / "media" / f"TIMSS-{subject}"
        if subject_dir.exists():
            total_files += sum(
                1 for _ in subject_dir.rglob("*")
                if _.is_file() and _.stat().st_size > 100
            )
    logger.info(f"Done: {total_files} total TIMSS media files on disk")

    return total_downloaded
