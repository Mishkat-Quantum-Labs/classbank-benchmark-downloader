"""Download metadata from TalkBankDB API and generate dataset documentation."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from pipeline.config import (
    CORPUS_DESCRIPTIONS,
    DATASET_DIR,
    ENGLISH_CORPORA,
    TALKBANK_API_URL,
)

logger = logging.getLogger(__name__)


# ── TalkBankDB API ──────────────────────────────────────────────────────────


def query_transcript_metadata(corpus: str) -> dict:
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


def collect_corpus_metadata(corpus: str) -> dict:
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
        media_types: set[str] = set()
        languages: set[str] = set()

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


# ── Output Generation ───────────────────────────────────────────────────────


def generate_manifest(all_metadata: list[dict], output_dir: Path) -> dict:
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


def generate_transcripts_index(output_dir: Path) -> None:
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


def generate_readme(manifest: dict, output_dir: Path) -> None:
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
\u251c\u2500\u2500 metadata/
\u2502   \u251c\u2500\u2500 corpus_manifest.json      # Full dataset manifest with metadata
\u2502   \u2514\u2500\u2500 transcripts_index.csv     # Index of all transcript files
\u251c\u2500\u2500 transcripts/
\u2502   \u251c\u2500\u2500 <corpus_name>/
\u2502   \u2502   \u2514\u2500\u2500 *.cha                 # CHAT format transcription files
\u2502   \u2514\u2500\u2500 ...
\u251c\u2500\u2500 media/
\u2502   \u251c\u2500\u2500 <corpus_name>/
\u2502   \u2502   \u2514\u2500\u2500 *.mp4 / *.mp3        # Audio/video recordings
\u2502   \u2514\u2500\u2500 ...
\u2514\u2500\u2500 README.md
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


# ── Main Entry Point ────────────────────────────────────────────────────────


def download_all_metadata(
    output_dir: Path | None = None,
    corpora: list[str] | None = None,
    skip_api: bool = False,
) -> dict:
    """Collect metadata and generate dataset documentation.

    Args:
        output_dir: Base dataset directory. Defaults to DATASET_DIR.
        corpora: List of corpus names. Defaults to ENGLISH_CORPORA.
        skip_api: If True, skip API queries and generate minimal metadata only.

    Returns:
        The manifest dict.
    """
    if output_dir is None:
        output_dir = DATASET_DIR

    if corpora is None:
        corpora = ENGLISH_CORPORA

    logger.info(f"Output: {output_dir}")
    logger.info(f"Corpora: {len(corpora)}")

    all_metadata = []
    if not skip_api:
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

    manifest = generate_manifest(all_metadata, output_dir)
    generate_transcripts_index(output_dir)
    generate_readme(manifest, output_dir)

    logger.info("Metadata generation complete.")
    return manifest
