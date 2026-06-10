"""Download CHAT-format transcript files from TalkBank ClassBank."""

import logging
import time
import zipfile
from pathlib import Path

from tqdm import tqdm

from pipeline.auth import create_session
from pipeline.config import DATASET_DIR, ENGLISH_CORPORA, TRANSCRIPT_ZIP_URL

logger = logging.getLogger(__name__)


def download_corpus_transcripts(
    corpus: str,
    output_dir: Path,
    session,
) -> bool:
    """Download and extract transcript zip for a single corpus.

    Returns:
        True if download succeeded, False otherwise.
    """
    transcript_dir = output_dir / "transcripts" / corpus
    transcript_dir.mkdir(parents=True, exist_ok=True)

    zip_url = TRANSCRIPT_ZIP_URL.format(corpus=corpus)
    logger.info(f"Downloading: {corpus} from {zip_url}")

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
            zip_path = transcript_dir / f"{corpus}.zip"
            with open(zip_path, "wb") as f:
                f.write(resp.content)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(transcript_dir)
                logger.info(f"  Extracted {corpus} transcripts")
                zip_path.unlink()
            except zipfile.BadZipFile:
                logger.warning(f"  Bad zip file for {corpus}")
            return True
        else:
            if "authModals" in resp.text or "Login" in resp.text[:500]:
                logger.warning(f"  Auth required for {corpus} - check credentials")
            else:
                logger.warning(f"  Unexpected response for {corpus}: HTTP {resp.status_code}")
            return False

    except Exception as e:
        logger.error(f"  Error downloading {corpus}: {e}")
        return False


def download_all_transcripts(
    output_dir: Path | None = None,
    corpora: list[str] | None = None,
) -> dict[str, bool]:
    """Download transcript zips for all (or specified) corpora.

    Args:
        output_dir: Base dataset directory. Defaults to DATASET_DIR.
        corpora: List of corpus names. Defaults to ENGLISH_CORPORA.

    Returns:
        Dict mapping corpus name to success status.
    """
    if output_dir is None:
        output_dir = DATASET_DIR

    if corpora is None:
        corpora = ENGLISH_CORPORA

    logger.info(f"Output: {output_dir}")
    logger.info(f"Corpora: {len(corpora)}")

    session = create_session()

    results: dict[str, bool] = {}
    for corpus in tqdm(corpora, desc="Downloading transcripts"):
        results[corpus] = download_corpus_transcripts(corpus, output_dir, session)
        time.sleep(1)

    success = sum(1 for v in results.values() if v)
    logger.info(f"Done: {success}/{len(corpora)} corpora downloaded successfully")
    return results
