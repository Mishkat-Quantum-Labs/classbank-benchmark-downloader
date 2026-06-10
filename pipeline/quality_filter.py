"""Quality filtering — moves warn-status files to data-limitation-set.

Separates files that failed verification (status="warn") from the main dataset
into the data-limitation-set/ directory to keep dataset/ clean with only
fully-verified (pass) files.
"""

import json
import logging
import shutil
from pathlib import Path

from pipeline.config import DATASET_DIR, LIMITATION_SET_DIR

logger = logging.getLogger(__name__)


def separate_warn_files(dataset_dir: Path | None = None) -> dict[str, int]:
    """Move warn-status files to data-limitation-set/.

    Reads verification.json from the parsed output directory, identifies files
    with status="warn", and moves their transcripts (.cha), parsed JSON, and
    associated media files to the data-limitation-set directory.

    Args:
        dataset_dir: Dataset directory. Defaults to DATASET_DIR.

    Returns:
        Dict with counts: moved_transcripts, moved_parsed, moved_media.

    Raises:
        FileNotFoundError: If verification.json doesn't exist.
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    base_dir = dataset_dir.parent
    limit_set = LIMITATION_SET_DIR

    # Read verification report
    verification_path = dataset_dir / "parsed" / "verification.json"
    if not verification_path.exists():
        raise FileNotFoundError(
            f"verification.json not found at {verification_path}. "
            "Run preprocess_transcripts.py first."
        )

    data = json.loads(verification_path.read_text(encoding="utf-8"))
    warn_files = [f for f in data["files"] if f["status"] == "warn"]

    if not warn_files:
        logger.info("No warn files found. Dataset is clean.")
        return {"moved_transcripts": 0, "moved_parsed": 0, "moved_media": 0}

    logger.info(f"Found {len(warn_files)} warn files to move")

    moved_transcripts = 0
    moved_parsed = 0
    moved_media = 0

    for wf in warn_files:
        source_rel = wf["source"]
        output_rel = wf["output"]
        corpus = wf["corpus"]

        # 1. Move transcript .cha file
        src_transcript = base_dir / source_rel
        transcript_rel = source_rel.replace("dataset/transcripts/", "")
        dst_transcript = limit_set / "transcripts" / transcript_rel
        if src_transcript.exists():
            dst_transcript.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_transcript), str(dst_transcript))
            moved_transcripts += 1

        # 2. Move parsed .json file
        src_parsed = base_dir / output_rel
        parsed_rel = output_rel.replace("dataset/parsed/", "")
        dst_parsed = limit_set / "parsed" / parsed_rel
        if src_parsed.exists():
            dst_parsed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_parsed), str(dst_parsed))
            moved_parsed += 1

        # 3. Move corresponding media file (if exists)
        stem = Path(source_rel).stem
        media_corpus_dir = dataset_dir / "media" / corpus
        if media_corpus_dir.exists():
            for media_file in media_corpus_dir.iterdir():
                if media_file.is_file() and media_file.stem == stem:
                    dst_media = limit_set / "media" / corpus / media_file.name
                    dst_media.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(media_file), str(dst_media))
                    moved_media += 1

    logger.info(
        f"Moved: {moved_transcripts} transcripts, {moved_parsed} parsed, {moved_media} media"
    )
    logger.info(f"Destination: {limit_set}")

    return {
        "moved_transcripts": moved_transcripts,
        "moved_parsed": moved_parsed,
        "moved_media": moved_media,
    }
