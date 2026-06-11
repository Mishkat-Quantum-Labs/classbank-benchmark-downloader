"""Core evaluation — text normalization, WER/CER computation, dataclasses."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

logger = logging.getLogger("benchmark")

# Industry-standard normalizer (same as OpenAI Whisper / ASR Leaderboard)
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
class FileMetrics:
    """Metrics for a single file."""

    file_id: str
    corpus: str
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    reference_words: int
    hypothesis_words: int
    stt_time: float = 0.0
    audio_duration: float = 0.0
    rtf: float = 0.0
    semantic_wer: float | None = None
    der: float | None = None


@dataclass
class EngineReport:
    """Aggregated report for one engine."""

    engine: str
    total_files: int
    successful_files: int
    failed_files: int
    file_metrics: list[FileMetrics] = field(default_factory=list)

    mean_wer: float = 0.0
    median_wer: float = 0.0
    std_wer: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    mean_cer: float = 0.0
    median_cer: float = 0.0
    mean_rtf: float = 0.0
    total_stt_time: float = 0.0
    total_audio_duration: float = 0.0

    total_insertions: int = 0
    total_deletions: int = 0
    total_substitutions: int = 0

    mean_semantic_wer: float | None = None
    median_semantic_wer: float | None = None
    mean_der: float | None = None
    median_der: float | None = None

    corpus_wer: dict = field(default_factory=dict)


def compute_file_metrics(
    reference_text: str,
    hypothesis_text: str,
    file_id: str,
    corpus: str,
    stt_time: float = 0.0,
    audio_duration: float = 0.0,
    **kwargs,
) -> Optional[FileMetrics]:
    """Compute WER/CER for a single file."""
    ref = normalize_text(reference_text)
    hyp = normalize_text(hypothesis_text)

    if not ref:
        logger.warning("[Eval] Empty reference for %s — skipping", file_id)
        return None

    if not hyp:
        ref_words = len(ref.split())
        return FileMetrics(
            file_id=file_id, corpus=corpus,
            wer=1.0, cer=1.0,
            insertions=0, deletions=ref_words, substitutions=0,
            reference_words=ref_words, hypothesis_words=0,
            stt_time=stt_time, audio_duration=audio_duration,
            rtf=stt_time / audio_duration if audio_duration > 0 else 0.0,
        )

    wer_out = jiwer.process_words(ref, hyp)
    cer_out = jiwer.process_characters(ref, hyp)

    ref_words = wer_out.hits + wer_out.substitutions + wer_out.deletions
    hyp_words = wer_out.hits + wer_out.substitutions + wer_out.insertions
    rtf = stt_time / audio_duration if audio_duration > 0 else 0.0

    return FileMetrics(
        file_id=file_id, corpus=corpus,
        wer=wer_out.wer, cer=cer_out.cer,
        insertions=wer_out.insertions,
        deletions=wer_out.deletions,
        substitutions=wer_out.substitutions,
        reference_words=ref_words,
        hypothesis_words=hyp_words,
        stt_time=stt_time, audio_duration=audio_duration, rtf=rtf,
    )
