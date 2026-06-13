# Local Open-Source Models Setup Guide

This guide explains how to set up, download, and run the two open-source ASR engines locally:

1. **WhisperX** — Whisper large-v3 + Pyannote diarization
2. **DiCoW** — Diarization-Conditioned Whisper (v3 MLC) + DiariZen

Both support **code-switching** (multilingual within same utterance) and **speaker diarization**.

---

## Prerequisites

- Python 3.11+
- FFmpeg installed and on PATH
- ~10 GB disk space for models
- GPU recommended (CUDA 12.x) but CPU works (much slower)

### Install FFmpeg

**Windows:**
```bash
winget install ffmpeg
# or download from https://ffmpeg.org/download.html and add to PATH
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

---

## Option 1: WhisperX (Recommended Starting Point)

WhisperX provides Whisper large-v3 with word-level timestamps and speaker diarization via Pyannote.

### 1. Install Dependencies

```bash
# From project root
pip install -e ".[whisperx]"
```

Or manually:
```bash
pip install whisperx torch torchaudio
```

For CPU-only (no CUDA):
```bash
pip install whisperx torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Get HuggingFace Token (Required for Diarization)

Speaker diarization uses Pyannote models which require accepting a license:

1. Create account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens → create a **read** token
3. Accept the user agreement at: https://huggingface.co/pyannote/speaker-diarization-community-1
4. Add token to your `.env`:

```env
HF_TOKEN=hf_your_token_here
```

### 3. Model Download (Automatic on First Run)

Models download automatically on first `transcribe()` call:

| Model | Size | Purpose |
|-------|------|---------|
| `faster-whisper-large-v3` | ~3 GB | ASR transcription |
| `wav2vec2` alignment model | ~1.2 GB | Word-level timestamps |
| `pyannote/speaker-diarization-community-1` | ~50 MB | Speaker diarization |

Models are cached in `~/.cache/huggingface/` (reused across runs).

### 4. Run Benchmark with WhisperX

```python
from benchmark.engines import WhisperXEngine

# CPU mode
engine = WhisperXEngine(
    model_name="large-v3",
    device="cpu",
    batch_size=4,          # reduce for low memory
    compute_type="int8",   # int8 for CPU, float16 for GPU
)

# GPU mode (much faster)
engine = WhisperXEngine(
    model_name="large-v3",
    device="cuda",
    batch_size=16,
    compute_type="float16",
)

# Transcribe a file
result = engine.transcribe("dataset/media/DISPEL/aoe2.mp4")
print(result["dialogue"])
print(f"WER data - segments: {len(result['segments'])}")
print(f"Speakers detected: {result['speakers']}")
print(f"Time taken: {result['stt_time']}s")
```

### 5. Performance Expectations

| Device | Model | Speed (5-min audio) | Quality |
|--------|-------|-------------------|---------|
| GPU (RTX 3090) | large-v3 | ~30 seconds | Best |
| GPU (RTX 3060) | large-v3 | ~60 seconds | Best |
| CPU (modern) | large-v3 | ~15-25 minutes | Best |
| CPU (modern) | medium | ~5-8 minutes | Good |
| CPU (modern) | small | ~2-3 minutes | Moderate |

---

## Option 2: DiCoW (Diarization-Conditioned Whisper)

DiCoW integrates diarization directly into Whisper's decoding process. Better for overlapping speech.

### 1. Install Dependencies

```bash
# From project root
pip install -e ".[dicow]"
```

Or manually:
```bash
pip install transformers torch torchaudio diarizen
```

For CPU-only:
```bash
pip install transformers torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diarizen
```

### 2. Model Download (Automatic on First Run)

Models download automatically:

| Model | Size | Purpose |
|-------|------|---------|
| `BUT-FIT/DiCoW_v3_MLC` | ~3.1 GB | Diarization-conditioned Whisper |
| `BUT-FIT/diarizen-wavlm-large-s80-md` | ~1.2 GB | DiariZen speaker diarization |

### 3. Run Benchmark with DiCoW

```python
from benchmark.engines import DiCoWEngine

# CPU mode
engine = DiCoWEngine(device="cpu")

# GPU mode
engine = DiCoWEngine(device="cuda")

# Transcribe a file
result = engine.transcribe("dataset/media/DISPEL/aoe2.mp4")
print(result["dialogue"])
print(f"Speakers detected: {result['speakers']}")
print(f"Time taken: {result['stt_time']}s")
```

### 4. Performance Expectations

| Device | Speed (5-min audio) | Notes |
|--------|-------------------|-------|
| GPU (RTX 3090) | ~45 seconds | Includes diarization + ASR |
| GPU (RTX 3060) | ~90 seconds | |
| CPU (modern) | ~20-30 minutes | Heavy model, GPU strongly recommended |

---

## Running the Full Benchmark

### Install Both Engines

```bash
pip install -e ".[local]"
```

### Add Engines to Benchmark Config

Update `benchmark/config.py` to include the local engines:

```python
ENGINES: dict[str, str] = {
    "gemini_pro": "Gemini 2.5 Pro",
    "gemini_31_pro": "Gemini 3.1 Pro",
    "gemini_35_flash": "Gemini 3.5 Flash",
    "elevenlabs_scribe_v2": "ElevenLabs Scribe v2",
    "whisperx_large_v3": "WhisperX large-v3",
    "dicow_v3_mlc": "DiCoW v3 MLC",
}
```

### Run Benchmark Script

```python
"""Example: run benchmark with local models."""
import json
from pathlib import Path
from benchmark.engines import WhisperXEngine, DiCoWEngine
from benchmark.evaluation import normalize_text, compute_wer, FileResult
from benchmark.statistics import compute_engine_report, report_to_dict

# Setup
PARSED_DIR = Path("dataset/parsed")
MEDIA_DIR = Path("dataset/media")
RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

def run_local_benchmark(engine, engine_name: str, corpus: str = "DISPEL"):
    """Run benchmark for a single engine on one corpus."""
    parsed_files = list((PARSED_DIR / corpus).glob("*.json"))
    file_results = []

    for parsed_file in parsed_files:
        # Load ground truth
        with open(parsed_file) as f:
            ground_truth = json.load(f)

        file_id = f"{corpus}/{parsed_file.stem}"
        reference_text = ground_truth.get("full_text", "")

        # Find corresponding media file
        media_extensions = [".mp4", ".mp3", ".wav", ".m4a"]
        media_path = None
        for ext in media_extensions:
            candidate = MEDIA_DIR / corpus / f"{parsed_file.stem}{ext}"
            if candidate.exists():
                media_path = str(candidate)
                break

        if not media_path:
            print(f"  ⚠ No media file for {file_id}, skipping")
            continue

        # Transcribe
        print(f"  Transcribing {file_id}...")
        try:
            result = engine.transcribe(media_path)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

        # Build hypothesis text from segments
        hypothesis_text = " ".join(seg["text"] for seg in result.get("segments", []))

        # Normalize
        norm_ref = normalize_text(reference_text)
        norm_hyp = normalize_text(hypothesis_text)

        file_results.append(FileResult(
            file_id=file_id,
            corpus=corpus,
            normalized_reference=norm_ref,
            normalized_hypothesis=norm_hyp,
            stt_time=result.get("stt_time", 0),
            audio_duration=0,  # can compute from media if needed
            der=None,
        ))

        per_file_wer = compute_wer([norm_ref], [norm_hyp])
        print(f"  ✓ {file_id} — WER: {per_file_wer:.2%}")

    # Aggregate report
    report = compute_engine_report(file_results, engine_name, len(parsed_files))
    return report_to_dict(report)


if __name__ == "__main__":
    # Run WhisperX
    print("=" * 60)
    print("Running WhisperX (large-v3) benchmark...")
    print("=" * 60)
    whisperx_engine = WhisperXEngine(model_name="large-v3", device="cpu")
    whisperx_results = run_local_benchmark(whisperx_engine, "whisperx_large_v3")

    # Run DiCoW
    print("=" * 60)
    print("Running DiCoW v3 MLC benchmark...")
    print("=" * 60)
    dicow_engine = DiCoWEngine(device="cpu")
    dicow_results = run_local_benchmark(dicow_engine, "dicow_v3_mlc")

    # Save results
    results = {
        "whisperx_large_v3": whisperx_results,
        "dicow_v3_mlc": dicow_results,
    }
    output_path = RESULTS_DIR / "local_models_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
```

Save this as `run_benchmark_local.py` and run:

```bash
python run_benchmark_local.py
```

---

## Troubleshooting

### "HF_TOKEN not set" error
Set your HuggingFace token:
```bash
set HF_TOKEN=hf_your_token_here       # Windows CMD
$env:HF_TOKEN="hf_your_token_here"    # PowerShell
export HF_TOKEN=hf_your_token_here    # Linux/macOS
```

### Out of memory (CPU)
Reduce batch size and use smaller model:
```python
engine = WhisperXEngine(model_name="small", device="cpu", batch_size=2)
```

### Out of memory (GPU)
Reduce batch size:
```python
engine = WhisperXEngine(model_name="large-v3", device="cuda", batch_size=4)
```

### FFmpeg not found
Ensure FFmpeg is installed and on PATH:
```bash
ffmpeg -version
```

### DiCoW / DiariZen import errors
The `diarizen` package may require specific transformers version:
```bash
pip install transformers>=4.40.0 diarizen
```

### Slow CPU inference
Expected. Use `medium` or `small` Whisper model for faster (less accurate) results:
```python
engine = WhisperXEngine(model_name="medium", device="cpu", batch_size=4)
```

---

## Model Comparison

| | WhisperX (large-v3) | DiCoW v3 MLC |
|---|---|---|
| **Base model** | Whisper large-v3 (faster-whisper) | Whisper large-v3 (fine-tuned) |
| **Diarization** | Pyannote (post-hoc) | DiariZen (conditioned) |
| **Code-switching** | ✅ 99 languages | ✅ 99 languages |
| **Overlapping speech** | Moderate | Better (diarization-guided) |
| **Maturity** | Battle-tested, widely used | Newer, research-stage |
| **CPU viable** | Yes (with smaller model) | Barely (very slow) |
| **Expected WER (classroom)** | 30-40% | 25-35% |
| **Install complexity** | Easy | Medium (diarizen package) |
