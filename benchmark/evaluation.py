"""Evaluation module — computes WER/CER metrics and statistical analysis.

Implements the text normalization pipeline and metric computation from
the BENCHMARKING-PLAN.md specification.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jiwer
import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger("benchmark")

# ── Text Normalization Pipeline ─────────────────────────────────────────────
# CRITICAL: Apply IDENTICAL normalization to BOTH reference AND hypothesis.

TRANSFORMS = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def normalize_text(text: str) -> str:
    """Apply standard text normalization for WER computation.

    Removes punctuation, lowercases, strips extra spaces, and removes
    common markers that shouldn't affect WER calculation.

    Args:
        text: Raw text from reference or hypothesis.

    Returns:
        Normalized text string ready for WER computation.
    """
    if not text:
        return ""

    # Remove [inaudible] markers
    import re
    text = re.sub(r"\[inaudible\]", "", text, flags=re.IGNORECASE)

    # Remove filler markers that may appear in reference
    text = re.sub(r"\b(um|uh|hmm|mhm|uh-huh|ah)\b", "", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Per-File Metrics ────────────────────────────────────────────────────────


@dataclass
class FileMetrics:
    """Metrics for a single file comparison."""

    file_id: str
    corpus: str
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    insertion_rate: float
    deletion_rate: float
    substitution_rate: float
    reference_words: int
    hypothesis_words: int
    stt_time: float = 0.0
    audio_duration: float = 0.0
    rtf: float = 0.0  # Real-Time Factor


@dataclass
class EngineReport:
    """Aggregated evaluation report for one engine."""

    engine: str
    total_files: int
    successful_files: int
    failed_files: int
    file_metrics: list[FileMetrics] = field(default_factory=list)

    # Aggregates (computed after all files)
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

    # Error type breakdown
    total_insertions: int = 0
    total_deletions: int = 0
    total_substitutions: int = 0
    mean_insertion_rate: float = 0.0
    mean_deletion_rate: float = 0.0
    mean_substitution_rate: float = 0.0

    # Per-corpus breakdown
    corpus_wer: dict = field(default_factory=dict)


def compute_file_metrics(
    reference_text: str,
    hypothesis_text: str,
    file_id: str,
    corpus: str,
    stt_time: float = 0.0,
    audio_duration: float = 0.0,
) -> Optional[FileMetrics]:
    """Compute WER/CER metrics for a single file.

    Args:
        reference_text: Ground truth full text (pre-normalized or raw).
        hypothesis_text: Engine output full text.
        file_id: Identifier for this file (e.g., "APT/02-Phonics").
        corpus: Corpus name.
        stt_time: Time taken by STT engine (seconds).
        audio_duration: Duration of the audio (seconds).

    Returns:
        FileMetrics dataclass, or None if comparison is not possible.
    """
    # Normalize both sides identically
    ref_normalized = normalize_text(reference_text)
    hyp_normalized = normalize_text(hypothesis_text)

    if not ref_normalized:
        logger.warning("[Eval] Empty reference text for %s — skipping", file_id)
        return None

    if not hyp_normalized:
        # Hypothesis is empty — 100% deletion
        ref_words = len(ref_normalized.split())
        return FileMetrics(
            file_id=file_id,
            corpus=corpus,
            wer=1.0,
            cer=1.0,
            insertions=0,
            deletions=ref_words,
            substitutions=0,
            insertion_rate=0.0,
            deletion_rate=1.0,
            substitution_rate=0.0,
            reference_words=ref_words,
            hypothesis_words=0,
            stt_time=stt_time,
            audio_duration=audio_duration,
            rtf=stt_time / audio_duration if audio_duration > 0 else 0.0,
        )

    # Compute WER with detailed measures
    measures = jiwer.compute_measures(
        ref_normalized,
        hyp_normalized,
        truth_transform=TRANSFORMS,
        hypothesis_transform=TRANSFORMS,
    )

    # Compute CER
    cer = jiwer.cer(
        ref_normalized,
        hyp_normalized,
        truth_transform=jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ]),
        hypothesis_transform=jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ]),
    )

    ref_words = measures["truth_count"] if "truth_count" in measures else len(ref_normalized.split())
    n = ref_words if ref_words > 0 else 1

    rtf = stt_time / audio_duration if audio_duration > 0 else 0.0

    return FileMetrics(
        file_id=file_id,
        corpus=corpus,
        wer=measures["wer"],
        cer=cer,
        insertions=measures["insertions"],
        deletions=measures["deletions"],
        substitutions=measures["substitutions"],
        insertion_rate=measures["insertions"] / n,
        deletion_rate=measures["deletions"] / n,
        substitution_rate=measures["substitutions"] / n,
        reference_words=ref_words,
        hypothesis_words=measures.get("hypothesis_count", len(hyp_normalized.split())),
        stt_time=stt_time,
        audio_duration=audio_duration,
        rtf=rtf,
    )


def bootstrap_ci(values: list[float], n_iterations: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: List of per-file WER values.
        n_iterations: Number of bootstrap samples.
        ci: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return (mean, mean)

    rng = np.random.default_rng(seed=42)
    arr = np.array(values)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_iterations)
    ])

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return (round(lower, 4), round(upper, 4))


def compute_engine_report(file_metrics: list[FileMetrics], engine: str, total_attempted: int) -> EngineReport:
    """Aggregate per-file metrics into an engine-level report.

    Args:
        file_metrics: List of FileMetrics for all successfully evaluated files.
        engine: Engine key/name.
        total_attempted: Total number of files attempted.

    Returns:
        EngineReport with all aggregates computed.
    """
    report = EngineReport(
        engine=engine,
        total_files=total_attempted,
        successful_files=len(file_metrics),
        failed_files=total_attempted - len(file_metrics),
        file_metrics=file_metrics,
    )

    if not file_metrics:
        return report

    wers = [m.wer for m in file_metrics]
    cers = [m.cer for m in file_metrics]

    report.mean_wer = round(float(np.mean(wers)), 4)
    report.median_wer = round(float(np.median(wers)), 4)
    report.std_wer = round(float(np.std(wers)), 4)
    report.mean_cer = round(float(np.mean(cers)), 4)
    report.median_cer = round(float(np.median(cers)), 4)

    # 95% CI via bootstrap
    ci_lower, ci_upper = bootstrap_ci(wers)
    report.ci_95_lower = ci_lower
    report.ci_95_upper = ci_upper

    # RTF
    rtfs = [m.rtf for m in file_metrics if m.rtf > 0]
    report.mean_rtf = round(float(np.mean(rtfs)), 4) if rtfs else 0.0
    report.total_stt_time = round(sum(m.stt_time for m in file_metrics), 2)
    report.total_audio_duration = round(sum(m.audio_duration for m in file_metrics), 2)

    # Error type breakdown
    report.total_insertions = sum(m.insertions for m in file_metrics)
    report.total_deletions = sum(m.deletions for m in file_metrics)
    report.total_substitutions = sum(m.substitutions for m in file_metrics)
    report.mean_insertion_rate = round(float(np.mean([m.insertion_rate for m in file_metrics])), 4)
    report.mean_deletion_rate = round(float(np.mean([m.deletion_rate for m in file_metrics])), 4)
    report.mean_substitution_rate = round(float(np.mean([m.substitution_rate for m in file_metrics])), 4)

    # Per-corpus breakdown
    corpus_metrics: dict[str, list[float]] = {}
    for m in file_metrics:
        if m.corpus not in corpus_metrics:
            corpus_metrics[m.corpus] = []
        corpus_metrics[m.corpus].append(m.wer)

    report.corpus_wer = {
        corpus: {
            "mean_wer": round(float(np.mean(wers)), 4),
            "median_wer": round(float(np.median(wers)), 4),
            "count": len(wers),
        }
        for corpus, wers in sorted(corpus_metrics.items())
    }

    return report


def pairwise_comparison(
    system_a_wers: list[float],
    system_b_wers: list[float],
) -> dict:
    """Run Wilcoxon signed-rank test for pairwise engine comparison.

    Args:
        system_a_wers: Per-file WERs for system A.
        system_b_wers: Per-file WERs for system B (same order/files).

    Returns:
        Dict with statistic, p_value, and significance assessment.
    """
    if len(system_a_wers) != len(system_b_wers):
        raise ValueError("Both systems must have the same number of file results")

    if len(system_a_wers) < 10:
        return {
            "statistic": None,
            "p_value": None,
            "significant": False,
            "note": "Too few samples for statistical test (need ≥10)",
        }

    # Filter out pairs where both are identical (Wilcoxon requires differences)
    differences = [a - b for a, b in zip(system_a_wers, system_b_wers)]
    if all(d == 0 for d in differences):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "note": "No differences between systems",
        }

    stat, p_value = wilcoxon(system_a_wers, system_b_wers)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "note": "Wilcoxon signed-rank test (p < 0.05 = significant)",
    }


def report_to_dict(report: EngineReport) -> dict:
    """Convert EngineReport to a JSON-serializable dict."""
    return {
        "engine": report.engine,
        "total_files": report.total_files,
        "successful_files": report.successful_files,
        "failed_files": report.failed_files,
        "mean_wer": report.mean_wer,
        "median_wer": report.median_wer,
        "std_wer": report.std_wer,
        "ci_95": [report.ci_95_lower, report.ci_95_upper],
        "mean_cer": report.mean_cer,
        "median_cer": report.median_cer,
        "mean_rtf": report.mean_rtf,
        "total_stt_time_sec": report.total_stt_time,
        "total_audio_duration_sec": report.total_audio_duration,
        "error_breakdown": {
            "total_insertions": report.total_insertions,
            "total_deletions": report.total_deletions,
            "total_substitutions": report.total_substitutions,
            "mean_insertion_rate": report.mean_insertion_rate,
            "mean_deletion_rate": report.mean_deletion_rate,
            "mean_substitution_rate": report.mean_substitution_rate,
        },
        "corpus_wer": report.corpus_wer,
        "per_file": [
            {
                "file_id": m.file_id,
                "corpus": m.corpus,
                "wer": m.wer,
                "cer": m.cer,
                "insertions": m.insertions,
                "deletions": m.deletions,
                "substitutions": m.substitutions,
                "reference_words": m.reference_words,
                "hypothesis_words": m.hypothesis_words,
                "stt_time": m.stt_time,
                "rtf": m.rtf,
            }
            for m in report.file_metrics
        ],
    }
