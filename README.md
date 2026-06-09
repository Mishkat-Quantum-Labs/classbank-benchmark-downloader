# TalkBank ClassBank Benchmark Dataset

Downloads English classroom recordings (audio/video) and CHAT transcriptions from [TalkBank's ClassBank](https://class.talkbank.org), converts them to standardized JSON, and organizes them for speech/NLP benchmarking.

## Prerequisites

1. **Python 3.8+**
2. **Free TalkBank account** — Register at https://class.talkbank.org (click Login > New User)

## Setup

```bash
pip install -r requirements.txt
```

## Authentication

All download scripts require a free TalkBank account. Credentials are loaded from a `.env` file in the project root.

### Create your `.env` file

```bash
cp .env.example .env
```

Then edit `.env` with your credentials:

```env
TALKBANK_EMAIL=your@email.com
TALKBANK_PASSWORD=yourpassword
```

> **Note:** The `.env` file is gitignored and will never be committed. Do not share it.

## Pipeline Overview

```
1. Download transcripts (.cha)     → dataset/transcripts/
2. Download media (audio/video)    → dataset/media/
3. Preprocess (CHA → JSON)        → dataset/parsed/
4. Verification & separation       → warn files → data-limitation-set/
```

## Scripts

| Script | Purpose | Auth Required |
|--------|---------|---------------|
| `download_metadata.py` | Queries TalkBankDB API, generates manifest and index | No |
| `download_transcripts.py` | Downloads CHAT-format transcript archives | Yes |
| `download_media.py` | Downloads audio/video recordings | Yes |
| `download_timss_transcripts.py` | Downloads TIMSS transcript archives (per-country) | Yes |
| `download_timss_media.py` | Downloads TIMSS video recordings (per-country) | Yes |
| `preprocess_transcripts.py` | Converts .cha → JSON with 6-check verification | No |

## Usage

### Full pipeline

```bash
# Step 1: Metadata (no auth needed)
python download_metadata.py

# Step 2: Transcripts
python download_transcripts.py

# Step 3: Media
python download_media.py

# Step 4: TIMSS transcripts (USA only)
python download_timss_transcripts.py

# Step 5: TIMSS media (USA only)
python download_timss_media.py

# Step 6: Convert to JSON and verify
python preprocess_transcripts.py
```

### Download specific corpora only

Each script accepts `--corpora` to target specific collections:

```bash
python download_transcripts.py --corpora Bradford,APT
python download_media.py --corpora Crowley,Roth,Warren
python download_metadata.py --corpora Bradford,Stevens,Curtis
```

### TIMSS data (separate scripts)

```bash
python download_timss_transcripts.py
python download_timss_media.py
python download_timss_media.py --subjects Math --countries USA,Australia
```

### Custom output directory

```bash
python download_transcripts.py --output-dir /path/to/my/data
python download_media.py --output-dir /path/to/my/data
```

### Parallel downloads

```bash
python download_media.py --workers 4
python download_timss_media.py --workers 4
```

## Smart Skip Logic

The download scripts automatically skip files that:
- Already exist in `dataset/media/` (size > 100 bytes)
- Were moved to `data-limitation-set/` (files with quality issues)

This prevents re-downloading files that were intentionally separated. Partially downloaded `.tmp` files will resume from where they left off.

## Dataset Structure

```
benchmark_data_p/
├── dataset/                      # Clean, verified data (303 files, all pass)
│   ├── transcripts/              # Source .cha files
│   │   ├── APT/                  # 32 files
│   │   ├── Bradford/             # 7 files
│   │   ├── Curtis/               # Multi-day recordings
│   │   ├── TIMSS-Math/           # 22 files
│   │   └── ... (38 corpora)
│   ├── parsed/                   # Converted JSON output
│   │   ├── {Corpus}/{file}.json  # Per-file JSON
│   │   ├── manifest.json         # Index of all converted files
│   │   └── verification.json     # Per-file quality report
│   └── media/                    # Audio/video recordings
│       ├── APT/                  # *.mp4
│       ├── Bradford/             # *.mp3
│       └── ...
├── data-limitation-set/          # Files with source data limitations (41 files)
│   └── dataset/
│       ├── transcripts/          # .cha files with timestamp issues
│       ├── parsed/               # Their JSON (valid, but incomplete timestamps)
│       └── media/                # Associated media (30 files)
├── download_media.py
├── download_timss_media.py
├── download_transcripts.py
├── download_timss_transcripts.py
├── download_metadata.py
├── preprocess_transcripts.py
└── tests/
```

## Output JSON Schema

Each converted file in `dataset/parsed/` follows this structure:

```json
{
  "metadata": {
    "session_id": "roth-roth-a1b2c3d4",
    "duration_seconds": 1234.567,
    "language": "en",
    "model": "cha_parser_v1",
    "created_at": "2026-06-09T10:31:01Z",
    "source": "dataset/transcripts/Roth/roth.cha",
    "corpus": "Roth"
  },
  "segments": [
    {
      "start": 0.0,
      "end": 3.361,
      "speaker": "teacher",
      "text": "this is the photograph of the Saanich Peninsula"
    }
  ],
  "full_text": "this is the photograph of the Saanich Peninsula..."
}
```

## Verification (6 Quality Checks)

Every converted file is verified against its source:

| Check | Pass Criteria |
|-------|---------------|
| Utterance count | Source vs output: exact match |
| Speaker mapping | All source speakers present |
| Timestamp coverage | > 90% of segments have start+end |
| Word count | ≤ 10% difference |
| Timestamp validity | No negative times, end ≥ start |
| Timestamp order | Non-decreasing start times |

Files that **pass** all checks stay in `dataset/`. Files with **warnings** (typically low timestamp coverage due to source data) are moved to `data-limitation-set/`.

## Data Limitation Set

The 41 files in `data-limitation-set/` were successfully converted but have incomplete timestamps in the **source .cha files** (not a pipeline bug). They include:

- **11 files with 0% timestamp coverage** — text-only transcripts (Curtis/overview, Person/b04)
- **30 files with 62-90% coverage** — partially timed by original annotators (APT, TIMSS-Science, Stevens, etc.)

The JSON output is correct — utterances without timestamps have `null` for start/end. They're separated because time-aligned downstream tools won't work reliably with them.

## Available Corpora

| Corpus | Description |
|--------|-------------|
| APT | Academically productive talk |
| Bradford | School lessons on cultural literacy |
| CarlaJim | Math lessons |
| CogInst | Problem-based learning in medical school |
| Crowley | Exploration of electricity in a science museum |
| Curtis | Second-grade geometry lessons |
| DISPEL | Tutorial game environment for dysrhythmic phonation |
| Frederiksen | Statistics tutoring |
| Graesser | Research methodology tutoring |
| Horowitz | Lessons on camels |
| JLS | Lessons on statistical graphing |
| Looney | Classroom interactions |
| MacWhinney | Lectures on Psychology Research Methods |
| Moschkovich | Math word problem solving |
| Person | Statistics tutoring |
| Rahm | Museum lessons on the color of the sky |
| Roth | Geography lesson |
| Stevens | Architecture discussions |
| TIMSS-Math | Math classroom recordings from multiple countries |
| TIMSS-Science | Science classroom recordings from multiple countries |
| Warren | Teachers' discussion of gravity |

## Benchmarking Use Cases

- **ASR**: Time-aligned utterances with media as ground truth
- **Speaker Diarization**: Multiple speakers per recording with IDs
- **Classroom Discourse Analysis**: Speaker role annotations (Teacher, Student, etc.)
- **Turn-taking Detection**: Timestamps for overlap and gap analysis
- **Educational NLP**: Topic modeling, QA, summarization

## License & Citation

Data is shared under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).

```
MacWhinney, B. (2000). The CHILDES Project: Tools for Analyzing Talk (3rd ed.).
Mahwah, NJ: Lawrence Erlbaum Associates.
```

## Source

- TalkBank: https://talkbank.org
- ClassBank: https://class.talkbank.org
- TBDBpy API: https://github.com/TalkBank/TBDBpy
