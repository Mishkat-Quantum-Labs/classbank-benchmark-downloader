"""
TalkBank ClassBank - Metadata Downloader
=========================================

Queries the TalkBankDB API for corpus metadata and generates:
- corpus_manifest.json: Full dataset manifest with metadata
- transcripts_index.csv: Index of all transcript files (if transcripts exist)
- README.md: Dataset documentation

Usage:
    python download_metadata.py [--output-dir ./dataset] [--corpora APT,Bradford]
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TALKBANK_API_URL = "https://sla2.talkbank.org:1515"

ENGLISH_CORPORA = [
    "APT", "Bradford", "CarlaJim", "CogInst", "Crowley", "Curtis",
    "DISPEL", "Frederiksen", "Graesser", "Horowitz", "JLS", "Looney",
    "MacWhinney", "Moschkovich", "Person", "Rahm", "Roth", "Stevens",
    "TIMSS-Math", "TIMSS-Science", "Warren",
]

CORPUS_DESCRIPTIONS = {
    "APT": "Academically productive talk",
    "Bradford": "School lessons on cultural literacy",
    "CarlaJim": "Math lessons",
    "CogInst": "Problem-based learning in medical school",
    "Crowley": "Exploration of electricity in a science museum",
    "Curtis": "Second-grade geometry lessons",
    "DISPEL": "Tutorial game environment for dysrhythmic phonation",
    "Frederiksen": "Statistics tutoring",
    "Graesser": "Research methodology tutoring",
    "Horowitz": "Lessons on camels",
    "JLS": "Lessons on statistical graphing",
    "Looney": "Classroom interactions",
    "MacWhinney": "Lectures on Psychology Research Methods",
    "Moschkovich": "Math word problem solving",
    "Person": "Statistics tutoring",
    "Rahm": "Museum lessons on the color of the sky",
    "Roth": "Geography lesson",
    "Stevens": "Architecture discussions",
    "TIMSS-Math": "Math classroom recordings from multiple countries",
    "TIMSS-Science": "Science classroom recordings from multiple countries",
    "Warren": "Teachers' discussion of gravity",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TalkBankDB API
# ---------------------------------------------------------------------------


def query_transcript_metadata(corpus):
    """Query TalkBankDB for transcript metadata."""
    url = f"{TALKBANK_API_URL}/getTranscriptSummary"
    query = {
        "corpusName": "class",
        "corpora": [["class", corpus]],
        "lang": {},
        "media": {},
        "age": {},
        "gender": {},
        "designType": {},
        "activityType": {},
        "groupType": {},
        "respType": "JSON",
    }
    try:
        resp = requests.post(url, json={"queryVals": query}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"API query failed for {corpus}: {e}")
        return {"colHeadings": [], "data": []}


def collect_corpus_metadata(corpus):
    """Collect metadata for a single corpus."""
    result = query_transcript_metadata(corpus)

    metadata = {
        "corpus": corpus,
        "description": CORPUS_DESCRIPTIONS.get(corpus, ""),
        "transcripts": [],
        "total_transcripts": 0,
        "media_types": [],
        "languages": ["eng"],
    }

    if result.get("data"):
        headers = result.get("colHeadings", [])
        media_types = set()
        languages = set()

        for row in result["data"]:
            entry = dict(zip(headers, row)) if headers else {}
            metadata["transcripts"].append(entry)
            if "media" in entry:
                media_types.add(entry["media"])
            if "languages" in entry:
                languages.add(entry["languages"])

        metadata["total_transcripts"] = len(result["data"])
        metadata["media_types"] = list(media_types)
        if languages:
            metadata["languages"] = list(languages)

    return metadata


# ---------------------------------------------------------------------------
# Output Generation
# ---------------------------------------------------------------------------


def generate_manifest(all_metadata, output_dir):
    """Generate dataset manifest JSON."""
    manifest = {
        "dataset_name": "TalkBank ClassBank English Classroom Recordings",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "source": "https://class.talkbank.org",
        "license": "CC BY-NC-SA 3.0",
        "description": (
            "English classroom recordings and CHAT transcriptions from TalkBank's "
            "ClassBank collection. Includes science, mathematics, medicine, and reading "
            "classroom interactions ranging from third grade to medical school."
        ),
        "corpora": all_metadata,
        "total_corpora": len(all_metadata),
        "total_transcripts": sum(m["total_transcripts"] for m in all_metadata),
    }

    manifest_path = output_dir / "metadata" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest saved: {manifest_path}")
    return manifest


def generate_transcripts_index(output_dir):
    """Generate a CSV index of all downloaded transcript files."""
    transcript_dir = output_dir / "transcripts"
    if not transcript_dir.exists():
        logger.info("No transcripts directory found, skipping index generation.")
        return

    records = []
    for cha_file in transcript_dir.rglob("*.cha"):
        rel_path = cha_file.relative_to(transcript_dir)
        corpus = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
        records.append({
            "corpus": corpus,
            "filename": cha_file.name,
            "relative_path": str(rel_path),
            "size_bytes": cha_file.stat().st_size,
        })

    if records:
        df = pd.DataFrame(records)
        index_path = output_dir / "metadata" / "transcripts_index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(index_path, index=False)
        logger.info(f"Transcript index saved: {len(records)} files indexed")


def generate_readme(manifest, output_dir):
    """Generate README for the dataset directory."""
    corpora_table = "\n".join(
        f"| {m['corpus']} | {m.get('description', '')} |"
        for m in manifest["corpora"]
    )

    readme = f"""# TalkBank ClassBank English Classroom Recordings Dataset

## Overview

This dataset contains English classroom recordings and their CHAT-format transcriptions
from TalkBank's ClassBank collection (https://class.talkbank.org).

**Created:** {manifest['created']}
**License:** {manifest['license']}
**Total Corpora:** {manifest['total_corpora']}
**Total Transcripts:** {manifest['total_transcripts']}

## Description

ClassBank provides transcribed filmed interactions in various classrooms. Topics include
science, mathematics, medicine, and reading. Learners range from third grade to medical
school. All interactions are in English (except TIMSS Japanese classroom data which is
excluded from this collection).

## Dataset Structure

```
dataset/
├── metadata/
│   ├── corpus_manifest.json      # Full dataset manifest with metadata
│   └── transcripts_index.csv     # Index of all transcript files
├── transcripts/
│   ├── <corpus_name>/
│   │   └── *.cha                 # CHAT format transcription files
│   └── ...
├── media/
│   ├── <corpus_name>/
│   │   └── *.mp4 / *.mp3        # Audio/video recordings
│   └── ...
└── README.md
```

## Corpora Included

| Corpus | Description |
|--------|-------------|
{corpora_table}

## Transcript Format (CHAT)

Transcripts use the CHAT (Codes for the Human Analysis of Transcripts) format developed
by Brian MacWhinney at Carnegie Mellon University. Key features:

- **Headers:** Lines beginning with `@` contain metadata (participants, date, etc.)
- **Main tiers:** Lines beginning with `*` contain speaker utterances (e.g., `*TEA:` for teacher)
- **Dependent tiers:** Lines beginning with `%` contain annotations (morphology, timing, etc.)
- **Timestamps:** Media alignment info in `%tim` tier or bullet markers

For full CHAT documentation: https://talkbank.org/0info/manuals/CHAT.pdf

## Usage for Benchmarking

This dataset can be used for:
- Speech recognition benchmarking (with audio-aligned transcripts)
- Speaker diarization evaluation
- Classroom discourse analysis
- Turn-taking and overlap detection
- Educational NLP tasks

## Citation

If you use this data, please cite:

```
MacWhinney, B. (2000). The CHILDES Project: Tools for Analyzing Talk (3rd ed.).
Mahwah, NJ: Lawrence Erlbaum Associates.
```

Additionally, cite the specific corpus contributor(s) as documented on each
corpus's page at https://class.talkbank.org.

## License

This data is shared under the Creative Commons BY-NC-SA 3.0 license.
See: https://creativecommons.org/licenses/by-nc-sa/3.0/
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    logger.info(f"README saved: {readme_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download metadata from TalkBankDB API and generate dataset docs."
    )
    parser.add_argument("--output-dir", type=str, default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--corpora", type=str, default=None,
                        help="Comma-separated corpora (default: all)")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API queries, generate minimal metadata only")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    corpora = [c.strip() for c in args.corpora.split(",")] if args.corpora else ENGLISH_CORPORA

    logger.info(f"Output: {output_dir}")
    logger.info(f"Corpora: {len(corpora)}")

    # Collect metadata
    all_metadata = []
    if not args.skip_api:
        logger.info("Querying TalkBankDB API for metadata...")
        for corpus in tqdm(corpora, desc="Querying metadata"):
            meta = collect_corpus_metadata(corpus)
            all_metadata.append(meta)
            time.sleep(0.5)
    else:
        for corpus in corpora:
            all_metadata.append({
                "corpus": corpus,
                "description": CORPUS_DESCRIPTIONS.get(corpus, ""),
                "transcripts": [],
                "total_transcripts": 0,
                "media_types": [],
                "languages": ["eng"],
            })

    # Generate outputs
    manifest = generate_manifest(all_metadata, output_dir)
    generate_transcripts_index(output_dir)
    generate_readme(manifest, output_dir)

    logger.info("Metadata generation complete.")


if __name__ == "__main__":
    main()
