"""
CHAT Transcript Parser for Benchmarking
========================================

Parses CHAT (.cha) transcription files into structured formats suitable
for speech recognition and NLP benchmarking tasks.

Outputs:
- JSON Lines format (one utterance per line)
- CSV format with columns for benchmarking

Usage:
    python parse_chat.py --input-dir ./dataset/transcripts --output-dir ./dataset/parsed
"""

import os
import re
import json
import csv
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """Represents a single utterance from a CHAT transcript."""
    corpus: str
    filename: str
    utterance_id: int
    speaker_id: str
    speaker_role: str
    text: str
    text_clean: str  # Cleaned text without CHAT codes
    start_time_ms: Optional[int]
    end_time_ms: Optional[int]
    duration_ms: Optional[int]
    media_file: Optional[str]


@dataclass
class TranscriptMetadata:
    """Metadata from a CHAT transcript header."""
    corpus: str
    filename: str
    participants: dict  # speaker_id -> {name, role}
    languages: list
    media_file: Optional[str]
    date: Optional[str]
    situation: Optional[str]


def clean_chat_text(text: str) -> str:
    """Remove CHAT coding conventions to get clean text for ASR benchmarking."""
    # Remove timing bullets: \x15...\x15
    text = re.sub(r'\x15\d+_\d+\x15', '', text)

    # Remove retracing markers [/], [//], [///]
    text = re.sub(r'\[/+\]', '', text)

    # Remove error markers [*]
    text = re.sub(r'\[\*\]', '', text)

    # Remove comments [% ...]
    text = re.sub(r'\[%[^\]]*\]', '', text)

    # Remove actions [=! ...]
    text = re.sub(r'\[=![^\]]*\]', '', text)

    # Remove explanations [= ...]
    text = re.sub(r'\[=[^\]]*\]', '', text)

    # Remove overlap markers [<] [>]
    text = re.sub(r'\[<\]|\[>\]', '', text)

    # Remove best guess markers [?]
    text = re.sub(r'\[\?\]', '', text)

    # Remove dependent tier markers
    text = re.sub(r'\[:[^\]]*\]', '', text)

    # Remove angle brackets (used for overlap/retracing scope)
    text = re.sub(r'[<>]', '', text)

    # Remove special form markers (prefixes like & for fragments)
    text = re.sub(r'&\w+', '', text)

    # Remove CHAT terminator symbols
    text = re.sub(r'[.!?]+$', '', text)  # sentence-final punctuation in CHAT
    text = text.replace('+...', '').replace('+/.', '').replace('+//.', '')
    text = text.replace('+!?', '').replace('+".', '')
    text = text.replace('+,', '').replace('+"', '')

    # Remove pause markers (..) (...)
    text = re.sub(r'\(\.\.*\)', '', text)

    # Remove word-level suffixes like @s @l @wp etc.
    text = re.sub(r'@\w+', '', text)

    # Remove 0 (zero-marked elements)
    text = re.sub(r'\b0\w*', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_timestamps(line: str):
    """Extract bullet timestamps from a CHAT line."""
    # CHAT uses bullet markers: \x15start_end\x15
    matches = re.findall(r'\x15(\d+)_(\d+)\x15', line)
    if matches:
        start = int(matches[0][0])
        end = int(matches[-1][1])
        return start, end
    return None, None


def parse_chat_file(filepath: Path, corpus_name: str) -> tuple:
    """Parse a single CHAT file into structured data."""
    metadata = TranscriptMetadata(
        corpus=corpus_name,
        filename=filepath.name,
        participants={},
        languages=[],
        media_file=None,
        date=None,
        situation=None,
    )
    utterances = []

    try:
        # Try multiple encodings
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            logger.warning(f"Cannot read {filepath}: encoding issues")
            return metadata, []

        lines = content.split('\n')
        utterance_id = 0
        current_speaker = None
        current_text = ""

        for line in lines:
            line = line.rstrip()

            # Parse header lines
            if line.startswith('@'):
                if line.startswith('@Participants:'):
                    # Parse participant list
                    parts = line[len('@Participants:'):].strip()
                    for part in parts.split(','):
                        part = part.strip()
                        tokens = part.split()
                        if len(tokens) >= 2:
                            spk_id = tokens[0]
                            role = tokens[-1] if len(tokens) >= 2 else "Unknown"
                            name = " ".join(tokens[1:-1]) if len(tokens) > 2 else tokens[0]
                            metadata.participants[spk_id] = {
                                "name": name,
                                "role": role,
                            }

                elif line.startswith('@Media:'):
                    media_info = line[len('@Media:'):].strip()
                    metadata.media_file = media_info.split(',')[0].strip()

                elif line.startswith('@Languages:'):
                    metadata.languages = [
                        l.strip() for l in line[len('@Languages:'):].strip().split(',')
                    ]

                elif line.startswith('@Date:'):
                    metadata.date = line[len('@Date:'):].strip()

                elif line.startswith('@Situation:'):
                    metadata.situation = line[len('@Situation:'):].strip()

            # Parse main tier (speaker utterances)
            elif line.startswith('*'):
                # Save previous utterance if exists
                if current_speaker and current_text:
                    start_ms, end_ms = extract_timestamps(current_text)
                    clean_text = clean_chat_text(current_text)

                    if clean_text:  # Only add non-empty utterances
                        role = metadata.participants.get(current_speaker, {}).get("role", "Unknown")
                        utt = Utterance(
                            corpus=corpus_name,
                            filename=filepath.name,
                            utterance_id=utterance_id,
                            speaker_id=current_speaker,
                            speaker_role=role,
                            text=current_text.strip(),
                            text_clean=clean_text,
                            start_time_ms=start_ms,
                            end_time_ms=end_ms,
                            duration_ms=(end_ms - start_ms) if start_ms is not None and end_ms is not None else None,
                            media_file=metadata.media_file,
                        )
                        utterances.append(utt)
                        utterance_id += 1

                # Start new utterance
                colon_idx = line.find(':')
                if colon_idx > 0:
                    current_speaker = line[1:colon_idx].strip()
                    current_text = line[colon_idx + 1:].strip()
                else:
                    current_speaker = None
                    current_text = ""

            # Continuation lines (start with tab)
            elif line.startswith('\t') and current_speaker:
                current_text += " " + line.strip()

            # Dependent tiers (%) - skip but could be useful
            elif line.startswith('%'):
                pass

        # Don't forget the last utterance
        if current_speaker and current_text:
            start_ms, end_ms = extract_timestamps(current_text)
            clean_text = clean_chat_text(current_text)
            if clean_text:
                role = metadata.participants.get(current_speaker, {}).get("role", "Unknown")
                utt = Utterance(
                    corpus=corpus_name,
                    filename=filepath.name,
                    utterance_id=utterance_id,
                    speaker_id=current_speaker,
                    speaker_role=role,
                    text=current_text.strip(),
                    text_clean=clean_text,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    duration_ms=(end_ms - start_ms) if start_ms is not None and end_ms is not None else None,
                    media_file=metadata.media_file,
                )
                utterances.append(utt)

    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}")

    return metadata, utterances


def main():
    parser = argparse.ArgumentParser(
        description="Parse CHAT transcripts into benchmarking-ready formats."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./dataset/transcripts",
        help="Directory containing downloaded CHAT transcripts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset/parsed",
        help="Output directory for parsed data",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "csv", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--time-aligned-only",
        action="store_true",
        help="Only output utterances that have timestamp alignment",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find all .cha files
    cha_files = list(input_dir.rglob("*.cha"))
    logger.info(f"Found {len(cha_files)} CHAT files to parse")

    all_utterances = []
    all_metadata = []

    for cha_file in cha_files:
        # Determine corpus name from directory structure
        rel = cha_file.relative_to(input_dir)
        corpus_name = rel.parts[0] if len(rel.parts) > 1 else "unknown"

        metadata, utterances = parse_chat_file(cha_file, corpus_name)
        all_metadata.append(metadata)
        all_utterances.extend(utterances)

    logger.info(f"Parsed {len(all_utterances)} utterances from {len(cha_files)} files")

    # Filter time-aligned only if requested
    if args.time_aligned_only:
        all_utterances = [u for u in all_utterances if u.start_time_ms is not None]
        logger.info(f"  Time-aligned utterances: {len(all_utterances)}")

    # Output as JSONL
    if args.format in ("jsonl", "both"):
        jsonl_path = output_dir / "utterances.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for utt in all_utterances:
                f.write(json.dumps(asdict(utt), ensure_ascii=False) + "\n")
        logger.info(f"JSONL output: {jsonl_path}")

    # Output as CSV
    if args.format in ("csv", "both"):
        csv_path = output_dir / "utterances.csv"
        df = pd.DataFrame([asdict(u) for u in all_utterances])
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV output: {csv_path}")

    # Output metadata
    meta_path = output_dir / "file_metadata.json"
    meta_list = []
    for m in all_metadata:
        meta_list.append({
            "corpus": m.corpus,
            "filename": m.filename,
            "participants": m.participants,
            "languages": m.languages,
            "media_file": m.media_file,
            "date": m.date,
            "situation": m.situation,
        })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata output: {meta_path}")

    # Print summary statistics
    logger.info("=" * 60)
    logger.info("PARSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files parsed: {len(cha_files)}")
    logger.info(f"Total utterances: {len(all_utterances)}")
    logger.info(f"Time-aligned utterances: {sum(1 for u in all_utterances if u.start_time_ms is not None)}")
    logger.info(f"Unique speakers: {len(set(u.speaker_id for u in all_utterances))}")

    # Per-corpus stats
    corpus_stats = {}
    for u in all_utterances:
        if u.corpus not in corpus_stats:
            corpus_stats[u.corpus] = {"utterances": 0, "time_aligned": 0}
        corpus_stats[u.corpus]["utterances"] += 1
        if u.start_time_ms is not None:
            corpus_stats[u.corpus]["time_aligned"] += 1

    logger.info("\nPer-corpus statistics:")
    for corpus, stats in sorted(corpus_stats.items()):
        logger.info(f"  {corpus}: {stats['utterances']} utterances ({stats['time_aligned']} time-aligned)")


if __name__ == "__main__":
    main()
