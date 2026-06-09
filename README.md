# TalkBank ClassBank Benchmark Dataset Downloader

Downloads English classroom recordings (audio/video) and CHAT transcriptions from [TalkBank's ClassBank](https://class.talkbank.org) for speech/NLP benchmarking.

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

## Scripts

| Script | Purpose | Auth Required |
|--------|---------|---------------|
| `download_metadata.py` | Queries TalkBankDB API, generates manifest and index | No |
| `download_transcripts.py` | Downloads CHAT-format transcript archives | Yes |
| `download_media.py` | Downloads audio/video recordings | Yes |
| `download_timss_transcripts.py` | Downloads TIMSS transcript archives (per-country) | Yes |
| `download_timss_media.py` | Downloads TIMSS video recordings (per-country) | Yes |
| `parse_chat.py` | Parses CHAT transcripts into JSONL/CSV | No |

## Usage

### Download everything (all 21 corpora)

```bash
# Step 1: Metadata (no auth needed)
python download_metadata.py

# Step 2: Transcripts (non-TIMSS corpora)
python download_transcripts.py

# Step 3: Media (non-TIMSS corpora)
python download_media.py

# Step 4: TIMSS transcripts (English/USA only)
python download_timss_transcripts.py

# Step 5: TIMSS media (English/USA only)
python download_timss_media.py

# Step 6: Parse into benchmark format
python parse_chat.py
```

### Download specific corpora only

Each script accepts `--corpora` to target specific collections:

```bash
# Only Bradford and APT transcripts
python download_transcripts.py --corpora Bradford,APT

# Only media for small corpora
python download_media.py --corpora Crowley,Roth,Warren

# Metadata for specific corpora
python download_metadata.py --corpora Bradford,Stevens,Curtis
```

### TIMSS data (separate scripts)

TIMSS data uses a different URL structure (per-country downloads). By default, only English (USA) data is downloaded:

```bash
# Download TIMSS transcripts (USA only by default)
python download_timss_transcripts.py

# Download TIMSS media (USA only by default)
python download_timss_media.py

# Only TIMSS-Math
python download_timss_media.py --subjects Math

# Specify different countries
python download_timss_media.py --countries USA,Australia --subjects Math
```

Available TIMSS countries:
- **TIMSS-Math**: Australia, Czech, HongKong, Japan, Netherlands, Switzerland, USA
- **TIMSS-Science**: Australia, Czech, Japan, Netherlands, USA

### Custom output directory

All scripts default to `./dataset`. Override with `--output-dir`:

```bash
python download_transcripts.py --output-dir /path/to/my/data
python download_media.py --output-dir /path/to/my/data
python download_metadata.py --output-dir /path/to/my/data
```

### Parse transcripts

```bash
# Parse all transcripts into JSONL + CSV
python parse_chat.py --input-dir ./dataset/transcripts --output-dir ./dataset/parsed

# Only time-aligned utterances (for ASR benchmarking)
python parse_chat.py --input-dir ./dataset/transcripts --output-dir ./dataset/parsed --time-aligned-only

# Output only JSONL (no CSV)
python parse_chat.py --format jsonl
```

### Metadata without API queries

If the TalkBankDB API is down or you just want a skeleton:

```bash
python download_metadata.py --skip-api
```

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

## Dataset Structure (after download)

```
dataset/
├── metadata/
│   ├── corpus_manifest.json      # Full dataset manifest
│   └── transcripts_index.csv     # Index of all .cha files
├── transcripts/
│   ├── APT/
│   ├── Bradford/
│   ├── Curtis/
│   ├── TIMSS-Math/               # USA transcripts
│   ├── TIMSS-Science/            # USA transcripts
│   └── ... (other corpora)
├── media/
│   ├── APT/                      # *.mp4
│   ├── Bradford/                 # *.mp3
│   ├── TIMSS-Math/               # USA videos
│   ├── TIMSS-Science/            # USA videos
│   └── ...
└── parsed/
    ├── utterances.jsonl
    ├── utterances.csv
    └── file_metadata.json
```

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
