"""Statistical analysis — bootstrap CI, pairwise tests, aggregate reports."""

import numpy as np
from scipy.stats import wilcoxon

from benchmark.evaluation import EngineReport, FileMetrics


def bootstrap_ci(values: list[float], n_iterations: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    if len(values) < 2:
        mean = float(np.mean(values)) if values else 0.0
        return (mean, mean)

    rng = np.random.default_rng(seed=42)
    arr = np.array(values)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_iterations)
    ])

    alpha = (1 - ci) / 2
    return (
        round(float(np.percentile(boot_means, alpha * 100)), 4),
        round(float(np.percentile(boot_means, (1 - alpha) * 100)), 4),
    )


def pairwise_comparison(system_a_wers: list[float], system_b_wers: list[float]) -> dict:
    """Wilcoxon signed-rank test between two engines."""
    if len(system_a_wers) != len(system_b_wers):
        raise ValueError("Both systems must have same number of results")

    if len(system_a_wers) < 10:
        return {"statistic": None, "p_value": None, "significant": False,
                "note": "Too few samples (need >=10)"}

    if all(a == b for a, b in zip(system_a_wers, system_b_wers)):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "note": "No differences between systems"}

    stat, p_value = wilcoxon(system_a_wers, system_b_wers)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def compute_engine_report(file_metrics: list[FileMetrics], engine: str, total_attempted: int) -> EngineReport:
    """Aggregate per-file metrics into engine-level report."""
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

    ci_lower, ci_upper = bootstrap_ci(wers)
    report.ci_95_lower = ci_lower
    report.ci_95_upper = ci_upper

    rtfs = [m.rtf for m in file_metrics if m.rtf > 0]
    report.mean_rtf = round(float(np.mean(rtfs)), 4) if rtfs else 0.0
    report.total_stt_time = round(sum(m.stt_time for m in file_metrics), 2)
    report.total_audio_duration = round(sum(m.audio_duration for m in file_metrics), 2)

    report.total_insertions = sum(m.insertions for m in file_metrics)
    report.total_deletions = sum(m.deletions for m in file_metrics)
    report.total_substitutions = sum(m.substitutions for m in file_metrics)

    sem_wers = [m.semantic_wer for m in file_metrics if m.semantic_wer is not None]
    if sem_wers:
        report.mean_semantic_wer = round(float(np.mean(sem_wers)), 4)
        report.median_semantic_wer = round(float(np.median(sem_wers)), 4)

    ders = [m.der for m in file_metrics if m.der is not None]
    if ders:
        report.mean_der = round(float(np.mean(ders)), 4)
        report.median_der = round(float(np.median(ders)), 4)

    corpus_metrics: dict[str, list[float]] = {}
    for m in file_metrics:
        corpus_metrics.setdefault(m.corpus, []).append(m.wer)

    report.corpus_wer = {
        corpus: {
            "mean_wer": round(float(np.mean(w)), 4),
            "median_wer": round(float(np.median(w)), 4),
            "count": len(w),
        }
        for corpus, w in sorted(corpus_metrics.items())
    }

    return report


def report_to_dict(report: EngineReport) -> dict:
    """Convert EngineReport to JSON-serializable dict."""
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
        "mean_semantic_wer": report.mean_semantic_wer,
        "median_semantic_wer": report.median_semantic_wer,
        "mean_der": report.mean_der,
        "median_der": report.median_der,
        "mean_rtf": report.mean_rtf,
        "total_stt_time_sec": report.total_stt_time,
        "total_audio_duration_sec": report.total_audio_duration,
        "error_breakdown": {
            "total_insertions": report.total_insertions,
            "total_deletions": report.total_deletions,
            "total_substitutions": report.total_substitutions,
        },
        "corpus_wer": report.corpus_wer,
        "per_file": [
            {
                "file_id": m.file_id,
                "corpus": m.corpus,
                "wer": m.wer,
                "cer": m.cer,
                "semantic_wer": m.semantic_wer,
                "der": m.der,
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
