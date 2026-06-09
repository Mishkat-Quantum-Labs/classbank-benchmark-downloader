"""
Separate Warn Files
====================

Moves files classified as "warn" by the preprocessor verification into
the data-limitation-set/ directory. This keeps the main dataset/ clean
with only fully-verified (pass) files.

Moves transcripts (.cha), parsed JSON, and associated media files.

Usage:
    python separate_warn_files.py [--dataset-dir ./dataset]
"""

import json
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Move warn-status files to data-limitation-set/"
    )
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                        help="Dataset directory (default: ./dataset)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    base_dir = dataset_dir.parent
    limit_set = base_dir / "data-limitation-set" / "dataset"

    # Read verification report
    verification_path = dataset_dir / "parsed" / "verification.json"
    if not verification_path.exists():
        print("[ERROR] verification.json not found. Run preprocess_transcripts.py first.")
        return 1

    data = json.loads(verification_path.read_text(encoding="utf-8"))
    warn_files = [f for f in data["files"] if f["status"] == "warn"]

    if not warn_files:
        print("[INFO] No warn files found. Dataset is clean.")
        return 0

    print(f"[INFO] Found {len(warn_files)} warn files to move")

    moved_transcripts = 0
    moved_parsed = 0
    moved_media = 0

    for wf in warn_files:
        source_rel = wf["source"]   # e.g. dataset/transcripts/APT/01-Character.cha
        output_rel = wf["output"]   # e.g. dataset/parsed/APT/01-Character.json
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

    print(f"[DONE] Moved: {moved_transcripts} transcripts, {moved_parsed} parsed, {moved_media} media")
    print(f"       Destination: {limit_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
