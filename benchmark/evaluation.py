"""Core evaluation — text normalization, WER/CER computation, dataclasses.

WER methodology matches the HuggingFace Open ASR Leaderboard exactly:
- Text normalization via OpenAI Whisper's EnglishTextNormalizer
- WER computed using jiwer across all utterances (not per-file averaging)
- Single WER number = jiwer.wer(all_references, all_hypotheses)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

logger = logging.getLogger("benchmark")

# Industry-standard normalizer (same as OpenAI Whisper / HF ASR Leaderboard)
_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Normalize text using Whisper's English normalizer (industry standard).

    Handles: lowercase, remove punctuation, expand contractions,
    expand titles (Dr.→doctor), remove fillers (um/uh), remove brackets.
    Applied identically to both reference and hypothesis.
    """
    if not text:
        return ""
    text = text.replace("[inaudible]", "").replace("[INAUDIBLE]", "")
    return _normalizer(text)


@dataclass
class FileResult:
    """Per-file data collected during evaluation."""

    file_id: str
    corpus: str
    normalized_reference: str
    normalized_hypothesis: str
    stt_time: float = 0.0
    audio_duration: float = 0.0
    der: float | None = None


@dataclass
class EngineReport:
    """Report for one engine — HF ASR Leaderboard style."""

    engine: str
    total_files: int
    successful_files: int
    failed_files: int

    # Primary metric (HF style): single WER across all utterances
    wer: float = 0.0
    cer: float = 0.0

    # Per-corpus WER breakdown
    corpus_wer: dict = field(default_factory=dict)

    # Timing
    total_stt_time: float = 0.0
    total_audio_duration: float = 0.0
    rtfx: float = 0.0

    # DER (if applicable)
    mean_der: float | None = None

    # Per-file details
    file_results: list[FileResult] = field(default_factory=list)


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Compute WER exactly like HF ASR Leaderboard.

    Passes all references and hypotheses as lists to jiwer in one call.
    This gives a single corpus-level WER number.
    """
    if not references or not hypotheses:
        return 0.0

    # Filter out empty references (jiwer requires non-empty)
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        return 0.0

    refs, hyps = zip(*pairs)
    # Replace empty hypotheses with a space (jiwer requirement)
    hyps = [h if h.strip() else " " for h in hyps]

    return jiwer.wer(list(refs), list(hyps))


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Compute CER across all utterances."""
    if not references or not hypotheses:
        return 0.0

    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        return 0.0

    refs, hyps = zip(*pairs)
    hyps = [h if h.strip() else " " for h in hyps]

    return jiwer.cer(list(refs), list(hyps))
