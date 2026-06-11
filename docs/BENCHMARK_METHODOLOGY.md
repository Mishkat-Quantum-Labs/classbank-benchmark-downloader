# Benchmark Methodology

How this ASR benchmark computes metrics, which tools it uses, and why.

## Metrics Overview

| Metric | What it Measures | Range |
|--------|-----------------|-------|
| WER | Word-level transcription accuracy | 0% (perfect) → 100%+ (terrible) |
| CER | Character-level transcription accuracy | 0% → 100%+ |
| Semantic WER | Meaning preservation (ignores formatting) | 0% (perfect) → 100% |
| DER | Speaker attribution accuracy | 0% (perfect) → 100%+ |
| RTF | Processing speed vs audio length | <1 = faster than real-time |

---

## 1. WER (Word Error Rate)

**Formula:** `WER = (Substitutions + Deletions + Insertions) / Reference Words`

**Tool:** [`jiwer`](https://github.com/jitsi/jiwer) — Python package using RapidFuzz (C++) for Levenshtein edit distance.

**Used by:** ASR Leaderboard (86 systems, 12 datasets), AssemblyAI SDK, Gladia.

**Verification:**
- ASR Leaderboard paper: https://arxiv.org/html/2510.06961v4
- AssemblyAI guide: https://www.assemblyai.com/blog/how-to-evaluate-speech-recognition-models
- Gladia docs: https://docs.gladia.io/chapters/pre-recorded-stt/benchmarking

---

## 2. Text Normalization

**Tool:** [`whisper-normalizer`](https://pypi.org/project/whisper-normalizer/) — OpenAI's `EnglishTextNormalizer`.

**What it does:**
- Lowercase
- Remove punctuation
- Expand contractions (`don't` → `do not`)
- Expand titles (`Dr.` → `doctor`)
- Remove filler words (`um`, `uh`)
- Remove bracketed annotations (`[laughter]`)

**Applied identically to both reference AND hypothesis before any metric computation.**

**Used by:** ASR Leaderboard ("closely following Whisper normalizer"), AssemblyAI ("use the open-source Whisper Normalizer"), OpenAI Whisper evaluation.

**Verification:**
- OpenAI Whisper paper (Section 3.2): https://cdn.openai.com/papers/whisper.pdf
- ASR Leaderboard: "We normalize all text prior to computing WER. This normalization removes punctuation and casing, and applies an English text normalization pipeline closely following that of Whisper."
- AssemblyAI: "AssemblyAI recommends using the open-source Whisper Normalizer for English transcription"

---

## 3. Semantic WER (LLM-as-Judge)

**Tool:** Claude Sonnet 4.5 via AWS Bedrock (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`)

**Configuration:**
- Temperature: 0 (deterministic)
- Max tokens: 4096
- Prompt: Adapted from Pipecat's open-source STT benchmark

**How it works:**
1. Send reference + hypothesis to Claude
2. Claude normalizes, aligns word-by-word
3. For each difference, Claude asks: "Would an LLM interpret these differently?"
4. Only counts meaning-changing errors (wrong names, numbers, nonsense words)
5. Ignores: punctuation, contractions, singular/plural, filler words, articles

**Does NOT count as error:** `"don't"` vs `"do not"`, `"3"` vs `"three"`, `"license"` vs `"licenses"`

**Counts as error:** `"card"` vs `"car"`, `"lentil"` vs `"landon"`, `"hours"` vs `"was"`

**Used by:** Pipecat STT Benchmark (same model, same prompt, same config).

**Verification:**
- Pipecat source: https://github.com/pipecat-ai/stt-benchmark/blob/main/src/stt_benchmark/evaluation/semantic_wer.py
- Pipecat docs: https://github.com/pipecat-ai/stt-benchmark/blob/main/docs/analysis.md

**Fallback:** If AWS credentials are not configured, falls back to word-overlap heuristic (unigram + bigram recall).

---

## 4. DER (Diarization Error Rate)

**Tool:** [`pyannote.metrics`](https://pyannote.github.io/pyannote-metrics/reference.html) — `DiarizationErrorRate` class.

**Configuration:**
- Speaker mapping: Hungarian algorithm (optimal)
- Collar: 0.25 seconds (forgives timing differences at speaker turn boundaries)
- Overlap handling: Included (`skip_overlap=False`)

**Formula:** `DER = (False Alarm + Missed Speech + Speaker Confusion) / Total Reference Speech Duration`

**Components:**
- **False alarm:** Non-speech marked as speech
- **Missed speech:** Speech marked as non-speech
- **Speaker confusion:** Correct speech region, wrong speaker assigned

**Used by:** AssemblyAI SDK, NIST evaluations, all academic diarization papers.

**Verification:**
- pyannote docs: https://pyannote.github.io/pyannote-metrics/reference.html
- AssemblyAI SDK (uses pyannote.metrics + pyannote.core): https://github.com/AssemblyAI-Solutions/stt-benchmarking-sdk
- NIST SCTK: https://github.com/usnistgov/SCTK
- dscore (DER standard tool): https://github.com/nryant/dscore

---

## 5. Statistical Analysis

### Bootstrap Confidence Interval (95%)

Resamples per-file WER values 1000 times to estimate the range the true mean WER falls in.

### Wilcoxon Signed-Rank Test

Non-parametric paired test to determine if one engine is statistically significantly better than another. Requires ≥10 paired samples. Significance threshold: p < 0.05.

**Used by:** Academic ASR papers for pairwise engine comparison.

**Verification:**
- Speechmatics recommends χ² or significance testing: https://docs.speechmatics.com/speech-to-text/accuracy-benchmarking
- Standard practice in paired ASR evaluations

---

## Pipeline Flow

```
Audio File + Ground Truth JSON
        │
        ├─→ STT Engine (Gemini / ElevenLabs) → Hypothesis JSON
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  EVALUATION                                                  │
│                                                              │
│  1. normalize(reference)    ← whisper-normalizer             │
│  2. normalize(hypothesis)   ← whisper-normalizer             │
│  3. WER/CER                 ← jiwer                          │
│  4. Semantic WER            ← Claude Sonnet 4.5 (Bedrock)    │
│  5. DER                     ← pyannote.metrics               │
│  6. Aggregate + CI          ← numpy + scipy                  │
│  7. Pairwise tests          ← scipy.stats.wilcoxon           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
   benchmark_results/*.json
```

---

## Dependencies

```
jiwer>=3.0.0              # WER/CER computation
whisper-normalizer>=0.1.10 # Text normalization (industry standard)
pyannote.metrics>=4.0.0    # DER computation (industry standard)
scipy>=1.10.0              # Wilcoxon test
numpy>=1.24.0              # Bootstrap CI
boto3>=1.28.0              # AWS Bedrock (Semantic WER)
```
