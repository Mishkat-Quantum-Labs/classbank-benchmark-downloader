# Google Colab Setup Guide

Complete guide to running the ClassBank ASR Benchmark on Google Colab — from cloning the private repo to running open-source models with GPU acceleration.

---

## Quick Start Notebook

Copy-paste each cell into a new Colab notebook. Select **Runtime → Change runtime type → T4 GPU** (free tier).

---

## Cell 1: Authenticate with GitHub (Private Repo)

```python
# ============================================================
# CELL 1: GitHub Authentication & Clone
# ============================================================
# Option A: Using Personal Access Token (PAT) — Recommended
# 1. Go to: https://github.com/settings/tokens
# 2. Generate new token (classic) → select "repo" scope
# 3. Copy the token

import os
from getpass import getpass

# Enter your GitHub PAT when prompted
GITHUB_TOKEN = getpass("Enter your GitHub Personal Access Token: ")

# Clone the private repo
!git clone https://{GITHUB_TOKEN}@github.com/Mishkat-Quantum-Labs/classbank-benchmark-downloader.git
%cd classbank-benchmark-downloader

print("✓ Repository cloned successfully!")
```

**Alternative: Using SSH key (advanced)**
```python
# If you prefer SSH, upload your private key to Colab:
from google.colab import files
uploaded = files.upload()  # Upload your id_ed25519 or id_rsa

!mkdir -p ~/.ssh
!mv id_ed25519 ~/.ssh/id_ed25519
!chmod 600 ~/.ssh/id_ed25519
!ssh-keyscan github.com >> ~/.ssh/known_hosts

!git clone git@github.com:Mishkat-Quantum-Labs/classbank-benchmark-downloader.git
%cd classbank-benchmark-downloader
```

---

## Cell 2: Install Dependencies

```python
# ============================================================
# CELL 2: Install Dependencies
# ============================================================
# Install the project with local model dependencies
!pip install -e ".[local]" -q

# Install ffmpeg (required for audio processing)
!apt-get install -y ffmpeg -qq

# Verify installations
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

import whisperx
print(f"✓ WhisperX installed")

print("\n✓ All dependencies installed!")
```

---

## Cell 3: Set Up Credentials

```python
# ============================================================
# CELL 3: Set Up Credentials
# ============================================================
from getpass import getpass
import os

# HuggingFace token (REQUIRED for WhisperX diarization)
# Get yours at: https://huggingface.co/settings/tokens
# ALSO accept license at: https://huggingface.co/pyannote/speaker-diarization-community-1
HF_TOKEN = getpass("Enter your HuggingFace token (hf_...): ")
os.environ["HF_TOKEN"] = HF_TOKEN

# Optional: For cloud engines (Gemini/ElevenLabs) — skip if only running local models
USE_CLOUD_ENGINES = False  # Set to True if you want Gemini/ElevenLabs too

if USE_CLOUD_ENGINES:
    os.environ["GEMINI_API_KEY"] = getpass("Gemini API Key: ")
    os.environ["ELEVENLABS_API_KEY"] = getpass("ElevenLabs API Key: ")
    os.environ["AWS_ACCESS_KEY_ID"] = getpass("AWS Access Key ID: ")
    os.environ["AWS_SECRET_ACCESS_KEY"] = getpass("AWS Secret Access Key: ")

# Write .env file
with open(".env", "w") as f:
    f.write(f"HF_TOKEN={HF_TOKEN}\n")
    if USE_CLOUD_ENGINES:
        f.write(f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')}\n")
        f.write(f"ELEVENLABS_API_KEY={os.environ.get('ELEVENLABS_API_KEY', '')}\n")
        f.write(f"AWS_ACCESS_KEY_ID={os.environ.get('AWS_ACCESS_KEY_ID', '')}\n")
        f.write(f"AWS_SECRET_ACCESS_KEY={os.environ.get('AWS_SECRET_ACCESS_KEY', '')}\n")

print("✓ Credentials configured!")
```

---

## Cell 4: Download Dataset

```python
# ============================================================
# CELL 4: Download Dataset (Transcripts + Media)
# ============================================================
# This downloads classroom recordings and ground-truth transcriptions
# from TalkBank ClassBank. Requires TalkBank credentials.

from getpass import getpass
import os

# TalkBank credentials (register free at https://class.talkbank.org)
TALKBANK_EMAIL = getpass("TalkBank email: ")
TALKBANK_PASSWORD = getpass("TalkBank password: ")

os.environ["TALKBANK_EMAIL"] = TALKBANK_EMAIL
os.environ["TALKBANK_PASSWORD"] = TALKBANK_PASSWORD

# Append to .env
with open(".env", "a") as f:
    f.write(f"TALKBANK_EMAIL={TALKBANK_EMAIL}\n")
    f.write(f"TALKBANK_PASSWORD={TALKBANK_PASSWORD}\n")

# Download transcripts
print("Downloading transcripts...")
!python run_download_transcripts.py
!python run_download_timss.py

# Download media (this is ~10GB, takes a while)
print("\nDownloading media files...")
!python run_download_media.py

# Preprocess: convert .cha → JSON
print("\nPreprocessing transcripts...")
!python preprocess_transcripts.py

# Quality filter
print("\nRunning quality filter...")
!python run_quality_filter.py

print("\n✓ Dataset ready!")
!echo "Parsed files:" && ls dataset/parsed/ | head -20
```

**Alternative: If you already have the dataset, upload it:**
```python
# Upload a zip of the dataset directory
from google.colab import files
uploaded = files.upload()  # Upload dataset.zip

!unzip -q dataset.zip -d .
print("✓ Dataset uploaded!")
```

---

## Cell 5: Run Benchmark — WhisperX (GPU)

```python
# ============================================================
# CELL 5: Run WhisperX Benchmark (GPU)
# ============================================================
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Run on a specific corpus (DISPEL is small, good for testing)
!python run_benchmark_local.py \
    --engine whisperx \
    --device {device} \
    --model large-v3 \
    --batch-size 16 \
    --corpus DISPEL
```

---

## Cell 6: Run Benchmark — DiCoW (GPU)

```python
# ============================================================
# CELL 6: Run DiCoW Benchmark (GPU)
# ============================================================
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

!python run_benchmark_local.py \
    --engine dicow \
    --device {device} \
    --corpus DISPEL
```

---

## Cell 7: Run Both Engines on All Corpora

```python
# ============================================================
# CELL 7: Full Benchmark — Both Engines, All Corpora
# ============================================================
# WARNING: This takes a long time! Start with one corpus first.

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

!python run_benchmark_local.py \
    --engine both \
    --device {device} \
    --model large-v3 \
    --batch-size 16
```

---

## Cell 8: View Results

```python
# ============================================================
# CELL 8: View Results
# ============================================================
import json
from pathlib import Path

results_file = Path("benchmark_results/local_models_results.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Run at: {results.get('run_at', 'N/A')}")
    print(f"Device: {results.get('device', 'N/A')}")
    print(f"Total time: {results.get('total_time_sec', 0):.1f}s")

    for engine_name, report in results.get("engine_reports", {}).items():
        print(f"\n{'─' * 40}")
        print(f"  Engine: {engine_name}")
        print(f"  WER:    {report['wer'] * 100:.2f}%")
        print(f"  CER:    {report['cer'] * 100:.2f}%")
        print(f"  Files:  {report['successful_files']}/{report['total_files']}")

        if report.get("corpus_wer"):
            print(f"  Per-corpus:")
            for corpus, data in report["corpus_wer"].items():
                print(f"    {corpus}: {data['wer'] * 100:.2f}% ({data['files']} files)")

        if report.get("per_file"):
            print(f"  Per-file:")
            for pf in report["per_file"]:
                print(f"    {pf['file_id']}: WER={pf['wer'] * 100:.2f}%")
else:
    print("No results file found. Run the benchmark first!")
```

---

## Cell 9: Compare with Cloud Engines (Optional)

```python
# ============================================================
# CELL 9: Run Cloud Engines for Comparison (Optional)
# ============================================================
# Only run this if you set USE_CLOUD_ENGINES = True in Cell 3

!python run_benchmark.py
```

---

## Cell 10: Download Results

```python
# ============================================================
# CELL 10: Download Results to Local Machine
# ============================================================
from google.colab import files
import shutil

# Zip results
shutil.make_archive("benchmark_results", "zip", "benchmark_results")
files.download("benchmark_results.zip")
print("✓ Results downloaded!")
```

---

## Troubleshooting

### "Repository not found" when cloning
- Ensure your PAT has `repo` scope
- Check the token hasn't expired
- Verify you have access to the Mishkat-Quantum-Labs organization

### "HF_TOKEN not set" or Pyannote access denied
- Get token at: https://huggingface.co/settings/tokens
- Accept license at: https://huggingface.co/pyannote/speaker-diarization-community-1
- Accept license at: https://huggingface.co/pyannote/segmentation-3.0

### CUDA out of memory
Reduce batch size or use smaller model:
```python
!python run_benchmark_local.py --engine whisperx --device cuda --model medium --batch-size 4
```

### Colab disconnects mid-run
- Use Colab Pro for longer runtimes
- Or process one corpus at a time: `--corpus DISPEL`
- Results save after each engine completes

### WhisperX alignment fails for non-English
This is expected for languages without a phoneme alignment model. The engine will skip alignment and proceed with segment-level timestamps. WER is still computed correctly.

### Slow download of TalkBank media
The media download is ~10GB. If Colab is slow:
1. Download on your local machine first
2. Upload `dataset.zip` to Google Drive
3. Mount Drive and unzip:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/dataset.zip .
!unzip -q dataset.zip
```

---

## Runtime Estimates (T4 GPU)

| Engine | Corpus | Files | Estimated Time |
|--------|--------|-------|---------------|
| WhisperX (large-v3) | DISPEL (4 files) | 4 | ~5-8 minutes |
| DiCoW v3 MLC | DISPEL (4 files) | 4 | ~8-12 minutes |
| WhisperX (large-v3) | All corpora | ~300 | ~6-10 hours |
| DiCoW v3 MLC | All corpora | ~300 | ~10-15 hours |

For quick testing, start with `--corpus DISPEL` (4 files, ~12 min total audio).

---

## Colab Pro Tips

1. **Save to Drive**: Mount Google Drive and save results there so they persist across sessions
2. **Use T4 GPU**: Free tier gives T4 which is sufficient for both models
3. **Background execution**: Keep the tab open or use Colab Pro for background execution
4. **Checkpointing**: Results are saved per-engine, so if Colab disconnects you keep partial results
