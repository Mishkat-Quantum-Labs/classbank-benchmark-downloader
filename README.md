# Origin ASR Benchmark

Downloads English classroom recordings and CHAT transcriptions from [TalkBank ClassBank](https://class.talkbank.org), converts them to standardized JSON, and benchmarks STT engines against the ground truth.

## Setup

```bash
git clone https://github.com/Mishkat-Quantum-Labs/classbank-benchmark-downloader.git
cd classbank-benchmark-downloader

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Credentials

Register a free account at https://class.talkbank.org, then:

```bash
cp .env.example .env
```

Fill in at minimum:

```env
# Required for downloading data
TALKBANK_EMAIL=your@email.com
TALKBANK_PASSWORD=yourpassword

# Required for benchmarking
GEMINI_API_KEY=your-gemini-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

## End-to-End Pipeline

Run these in order:

```bash
# 1. Download transcripts (.cha files)
python run_download_transcripts.py
python run_download_timss.py

# 2. Download media (audio/video, ~10 GB)
python run_download_media.py

# 3. Convert .cha → JSON with verification
python preprocess_transcripts.py

# 4. Separate warn-status files
python run_quality_filter.py

# 5. Download metadata & generate docs
python run_download_metadata.py

# 6. Run benchmark against STT engines
python run_benchmark.py
```

After this you'll have:
- `dataset/` — 303 verified files (transcripts + JSON + media)
- `data-limitation-set/` — 41 files with incomplete timestamps (valid but separated)
- `benchmark_results/` — STT engine evaluation reports

## Project Structure

```
├── pipeline/                     # Data acquisition & quality package
│   ├── config.py                 # Paths, URLs, corpora, credentials
│   ├── auth.py                   # TalkBank authentication
│   ├── quality_filter.py         # Separates warn files
│   └── download/
│       ├── media.py              # Non-TIMSS media download
│       ├── transcripts.py        # Non-TIMSS transcript download
│       ├── timss_media.py        # TIMSS media download
│       ├── timss_transcripts.py  # TIMSS transcript download
│       ├── metadata.py           # TalkBankDB API + docs generation
│       └── helpers.py            # Shared download/scraping utilities
│
├── benchmark/                    # ASR benchmarking package
│   ├── config.py                 # Engine settings, API keys
│   ├── engines/                  # STT engine implementations
│   │   ├── gemini_engine.py
│   │   └── elevenlabs_engine.py
│   ├── evaluation.py             # WER/CER metrics
│   ├── llm_client.py             # LLM for speaker classification
│   └── prompts.py                # Prompt templates
│
├── preprocess_transcripts.py     # CHA→JSON conversion + verification
│
├── run_download_media.py         # Entry: download media
├── run_download_transcripts.py   # Entry: download transcripts
├── run_download_timss.py         # Entry: download TIMSS data
├── run_download_metadata.py      # Entry: metadata & docs
├── run_quality_filter.py         # Entry: separate warn files
├── run_benchmark.py              # Entry: run STT benchmark
│
├── dataset/                      # Clean verified data
│   ├── transcripts/{Corpus}/     # Source .cha files
│   ├── parsed/{Corpus}/          # Converted JSON
│   └── media/{Corpus}/           # Audio/video
│
├── data-limitation-set/          # Files with source data limitations
├── benchmark_results/            # STT evaluation outputs
└── tests/                        # pytest suite
```

## Output JSON Schema

Each file in `dataset/parsed/` follows:

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

## Verification Checks

Every converted file passes 6 quality checks:

| Check | Pass | Warn | Fail |
|-------|------|------|------|
| Utterance count diff | 0 | 1–2 | ≥3 |
| Speaker mapping | All present | Missing roles | — |
| Timestamp coverage | >90% | ≤90% | — |
| Word count diff | ≤10% | 10–25% | >25% |
| Timestamp validity | All valid | — | Any violation |
| Timestamp order | Monotonic | — | Not monotonic |

Files with **pass** stay in `dataset/`. Files with **warn** go to `data-limitation-set/`.

## Corpora

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
| TIMSS-Math | Math classroom recordings (USA) |
| TIMSS-Science | Science classroom recordings (USA) |
| Warren | Teachers' discussion of gravity |

## License & Citation

Data shared under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).

```
MacWhinney, B. (2000). The CHILDES Project: Tools for Analyzing Talk (3rd ed.).
Mahwah, NJ: Lawrence Erlbaum Associates.
```
