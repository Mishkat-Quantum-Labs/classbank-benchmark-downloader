# CHA-to-JSON Pipeline Report

## What This Is

A preprocessing pipeline that converts CHAT-format (.cha) classroom transcription files from TalkBank's ClassBank dataset into standardized JSON. The JSON output is compatible with our transcription pipeline and can be used identically alongside ASR-derived transcripts.

## Source Data

- **Source**: TalkBank ClassBank (https://talkbank.org)
- **Format**: CHAT (.cha) transcription files
- **Location**: `dataset/transcripts/`
- **Total files discovered**: 322 across 37 corpora

## Directory Structure

```
benchmark_data_p/
├── dataset/
│   ├── transcripts/          # Source .cha files (281 clean files, 34 corpora)
│   ├── parsed/               # Output JSON files
│   │   ├── {Corpus}/         # Per-corpus subdirectories
│   │   │   └── {file}.json   # Converted transcript
│   │   ├── manifest.json     # Index of all converted files
│   │   └── verification.json # Per-file quality report
│   └── media/                # Associated audio/video files
├── data-limitation-set/      # Files with source data limitations (moved out)
│   └── dataset/
│       ├── transcripts/      # 41 .cha files with timestamp issues
│       ├── parsed/           # Their converted JSON (still valid, just incomplete timestamps)
│       └── media/            # Associated media (30 files)
├── preprocess_transcripts.py # The conversion pipeline script
└── tests/                    # Test suite (pytest + hypothesis)
```

## How It Works

The pipeline runs as a single script with 5 stages:

1. **Discovery** — recursively finds all .cha files, groups by corpus (parent directory)
2. **Parsing** — reads each .cha file using `pylangacq` with encoding error tolerance
3. **Transformation** — cleans CHAT annotations, converts timestamps to seconds, maps speakers to roles, builds JSON
4. **Verification** — runs 6 quality checks per file, classifies as pass/warn/fail
5. **Reporting** — writes manifest.json and verification.json

### Running It

```bash
python preprocess_transcripts.py
```

## Output JSON Schema

Each converted file follows this structure:

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

## Verification: 6 Quality Checks

Every converted file is verified against its source:

| Check | What It Measures | Pass Criteria |
|-------|-----------------|---------------|
| Utterance count | Source utterances vs output segments | Exact match |
| Speaker mapping | All source speakers present in output | None missing |
| Timestamp coverage | % of segments with start+end times | > 90% |
| Word count | Cleaned word count: source vs output | ≤ 10% difference |
| Timestamp validity | No negative times, end ≥ start | Zero violations |
| Timestamp order | Segments sorted by start time | Non-decreasing |

## Results

### Clean Dataset (281 files)

| Metric | Value |
|--------|-------|
| Files | 281 |
| Corpora | 34 |
| Total segments | 65,455 |
| Total duration | ~58.9 hours |
| Pass rate | **100%** |
| Warnings | 0 |
| Failures | 0 |

All 281 files pass every verification check.

### Data Limitation Set (41 files moved out)

These files were successfully converted but flagged because the **source .cha files have incomplete or missing timestamps**. This is a source data characteristic, not a pipeline bug.

**11 files with 0% timestamp coverage** — text-only transcripts, never time-aligned:
- Curtis/overview/ (10 files): area-hands, area-interview, bookshelf, islands, measure, rulers, tapemeasures, units, usingtape, zeropoint
- Person/b04 (1 file)

**30 files with 62-90% timestamp coverage** — partially timed by original annotators:
- APT/ (19 files): 01-Character, 01-Sleep, 02-Galimoto, 02-SubtractComp, 04-Grounded, 04-Uniforms, 04-Vascular, 05-BarkBeetles, 05-Decomposers, 05-Hurricanes, 05-VolumeMethod, 05-WildHorses, 06-Loyalists, 06-Slavery, 07-EmmettTill, 07-Maine, 08-Border, 08-Maleeka, 11-PolynomialsA
- TIMSS-Science/ (5 files): CC11241819, CC11241837, CC11241849, CC11241853, CC12021339
- Stevens/ (2 files): firm, library
- Person/a26 (1 file)
- Crowley/trolley (1 file)
- Curtis/dec13/dec13g (1 file)
- DISPEL/civ6 (1 file)

The JSON output for these files is correct — utterances without timestamps simply have `null` for start/end. They were separated because they won't work reliably with time-aligned downstream tools.

## Text Cleaning

The pipeline removes these CHAT annotations from utterance text:

- Inline timestamp markers (`␕0_3361␕`)
- Stress/syllable markers (U+0001, U+0002)
- Pause markers: `(.)`, `(0.3)`, `(1.4)`
- Terminator codes: `+/.`, `+//?`, `+...`, `+//.`
- Elongation colons: `a:nd` → `and`, `be:` → `be`
- Incomplete-word parentheses: `an(d)` → `and`
- Extra whitespace collapsed and trimmed

## Speaker Mapping

Speaker codes from CHAT `@Participants` header are mapped to lowercase roles:
- `TEA` → `teacher`
- `STU` → `student`
- If no role defined, uses the code lowercased (e.g., `CHI` → `chi`)

## Dependencies

```
pylangacq>=0.18.0
tqdm>=4.65.0
pytest>=7.0.0
hypothesis>=6.0.0
```

## Tests

```bash
python -m pytest tests/ -v
```

30 tests covering speaker mapping, verification classification, and edge cases. All pass.
