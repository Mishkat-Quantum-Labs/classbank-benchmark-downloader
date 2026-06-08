# TalkBank ClassBank Benchmark Dataset Downloader

Downloads English classroom recordings (audio/video) and CHAT transcriptions from [TalkBank's ClassBank](https://class.talkbank.org) for speech/NLP benchmarking.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download metadata (no auth needed for API queries)
python download_metadata.py --output-dir ./dataset

# 3. Download transcripts (requires free TalkBank account)
python download_transcripts.py --output-dir ./dataset

# 4. Download media files (requires free TalkBank account)
python download_media.py --output-dir ./dataset

# 5. Parse transcripts into benchmark-ready format
python parse_chat.py --input-dir ./dataset/transcripts --output-dir ./dataset/parsed
```

## Prerequisites

1. **Python 3.8+**
2. **Free TalkBank account** — Register at https://class.talkbank.org (click Login > New User)

You can provide credentials via:
- Command line: `--email your@email.com --password yourpass`
- Environment variables: `TALKBANK_EMAIL` and `TALKBANK_PASSWORD`
- Interactive prompt (default)

## Scripts

| Script | Purpose |
|--------|---------|
| `download_metadata.py` | Queries TalkBankDB API for corpus metadata, generates manifest and README |
| `download_transcripts.py` | Downloads CHAT-format transcript archives for all 21 English corpora |
| `download_media.py` | Downloads audio/video recordings with parallel downloads |
| `download_classbank.py` | All-in-one script (transcripts + media + metadata) |
| `parse_chat.py` | Parses CHAT transcripts into JSONL/CSV for benchmarking |

## Usage Options

### Download only transcripts (no media)
```bash
python download_transcripts.py --output-dir ./dataset --corpora Bradford,Stevens
```

### Download media for specific corpora with 10 parallel downloads
```bash
python download_media.py --corpora APT,Bradford --parallel 10
```

### Generate metadata without API queries
```bash
python download_metadata.py --skip-api
```

### Parse only time-aligned utterances (for ASR benchmarking)
```bash
python parse_chat.py --input-dir ./dataset/transcripts --output-dir ./dataset/parsed --time-aligned-only
```

## Dataset Structure (after download)

```
dataset/
├── metadata/
│   ├── corpus_manifest.json      # Full dataset manifest with API metadata
│   └── transcripts_index.csv     # Index of all .cha files
├── transcripts/
│   ├── APT/                      # Academically Productive Talk
│   ├── Bradford/                 # Cultural literacy interviews
│   ├── Curtis/                   # Second-grade geometry (160 transcripts)
│   └── ... (21 corpora total)
├── media/
│   ├── APT/                      # *.mp4, *.mp3
│   ├── Bradford/                 # *.mp3 (audio only)
│   └── ...
└── parsed/
    ├── utterances.jsonl          # One utterance per line
    ├── utterances.csv            # Tabular format
    └── file_metadata.json        # Per-file metadata
```

## Corpora Included (21 English ClassBank collections)

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

## Parsed Output Format

### JSONL (utterances.jsonl)
```json
{
  "corpus": "Bradford",
  "filename": "14.cha",
  "utterance_id": 0,
  "speaker_id": "CHI",
  "speaker_role": "Target_Child",
  "text_clean": "he was the first president",
  "start_time_ms": 12345,
  "end_time_ms": 15678,
  "duration_ms": 3333,
  "media_file": "14.mp3"
}
```

## Benchmarking Use Cases

- **ASR (Automatic Speech Recognition)**: Use time-aligned utterances with media files as ground truth
- **Speaker Diarization**: Multiple speakers per recording with speaker IDs
- **Classroom Discourse Analysis**: Rich speaker role annotations (Teacher, Student, etc.)
- **Turn-taking Detection**: Timestamps enable overlap and gap analysis
- **Educational NLP**: Transcript content for topic modeling, QA, summarization

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
