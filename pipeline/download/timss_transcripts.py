"""Download TIMSS transcript files from TalkBank ClassBank."""

import logging
import time
import zipfile
from pathlib import Path

from tqdm import tqdm

from pipeline.auth import create_session
from pipeline.config import (
    DATASET_DIR,
    TIMSS_COUNTRIES,
    TIMSS_SUBJECTS,
    TIMSS_TRANSCRIPT_ZIP_URL,
)

logger = logging.getLogger(__name__)


def download_timss_corpus_transcripts(
    subject: str,
    country: str,
    output_dir: Path,
    session,
) -> bool:
    """Download and extract transcript zip for a TIMSS subject/country.

    Returns:
        True if download succeeded, False otherwise.
    """
    transcript_dir = output_dir / "transcripts" / f"TIMSS-{subject}"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    zip_url = TIMSS_TRANSCRIPT_ZIP_URL.format(subject=subject, country=country)
    logger.info(f"Downloading: TIMSS-{subject}/{country} from {zip_url}")

    try:
        resp = session.get(zip_url, timeout=120)

        content_type = resp.headers.get("content-type", "")
        is_zip = (
            "application/zip" in content_type
            or "application/octet-stream" in content_type
            or (resp.status_code == 200 and len(resp.content) > 500
                and resp.content[:4] == b'PK\x03\x04')
        )

        if is_zip:
            zip_path = transcript_dir / f"TIMSS-{subject}.zip"
            with open(zip_path, "wb") as f:
                f.write(resp.content)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(transcript_dir)
                logger.info(f"  Extracted TIMSS-{subject}/{country} transcripts")
                zip_path.unlink()
            except zipfile.BadZipFile:
                logger.warning(f"  Bad zip file for TIMSS-{subject}/{country}")
            return True
        else:
            if "authModals" in resp.text or "Login" in resp.text[:500]:
                logger.warning(
                    f"  Auth required for TIMSS-{subject}/{country} - check credentials"
                )
            else:
                logger.warning(
                    f"  Unexpected response for TIMSS-{subject}/{country}: "
                    f"HTTP {resp.status_code}"
                )
            return False

    except Exception as e:
        logger.error(f"  Error downloading TIMSS-{subject}/{country}: {e}")
        return False


def download_all_timss_transcripts(
    output_dir: Path | None = None,
    subjects: list[str] | None = None,
    countries: list[str] | None = None,
) -> dict[str, bool]:
    """Download TIMSS transcript zips for all (or specified) subjects/countries.

    Args:
        output_dir: Base dataset directory. Defaults to DATASET_DIR.
        subjects: List of TIMSS subjects. Defaults to both Math and Science.
        countries: List of countries. Defaults to all per subject.

    Returns:
        Dict mapping "TIMSS-{subject}/{country}" to success status.
    """
    if output_dir is None:
        output_dir = DATASET_DIR

    if subjects is None:
        subjects = TIMSS_SUBJECTS

    session = create_session()

    results: dict[str, bool] = {}
    for subject in subjects:
        if subject not in TIMSS_COUNTRIES:
            logger.warning(f"Unknown TIMSS subject: {subject}. Use Math or Science.")
            continue

        available_countries = TIMSS_COUNTRIES[subject]
        if countries:
            filtered = [c for c in countries if c in available_countries]
            for c in countries:
                if c not in available_countries:
                    logger.warning(
                        f"  Country '{c}' not available for TIMSS-{subject}. "
                        f"Available: {', '.join(available_countries)}"
                    )
        else:
            filtered = available_countries

        logger.info(f"=== TIMSS-{subject} ===")
        logger.info(f"Countries: {', '.join(filtered)}")

        for country in tqdm(filtered, desc=f"TIMSS-{subject} transcripts"):
            key = f"TIMSS-{subject}/{country}"
            results[key] = download_timss_corpus_transcripts(
                subject, country, output_dir, session
            )
            time.sleep(1)

    success = sum(1 for v in results.values() if v)
    logger.info(f"Done: {success}/{len(results)} TIMSS country datasets downloaded successfully")
    return results
