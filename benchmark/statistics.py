"""Aggregate reports — HF ASR Leaderboard style.

Single WER number per engine, computed across all utterances in one jiwer call.
Per-corpus breakdown uses the same approach.
"""

import numpy as np

from benchmark.evaluation import EngineReport, FileResult, compute_wer, compute_cer


def compute_engine_report(
    file_results: list[FileResult],
    engine: str,
    total_attempted: int,
) -> EngineReport:
    """Aggregate file results into engine report (HF Leaderboard style).

    WER = jiwer.wer(all_references, all_hypotheses) in one call.
    """
    report = EngineReport(
        engine=engine,
        total_files=total_attempted,
        successful_files=len(file_results),
        failed_files=total_attempted - len(file_results),
        file_results=file_results,
    )

    if not file_results:
        return report

    # ── Corpus-level WER (HF style: one jiwer call across all files) ───
    all_refs = [f.normalized_reference for f in file_results]
    all_hyps = [f.normalized_hypothesis for f in file_results]

    report.wer = round(compute_wer(all_refs, all_hyps), 4)
    report.cer = round(compute_cer(all_refs, all_hyps), 4)

    # ── Timing ─────────────────────────────────────────────────────────
    report.total_stt_time = round(sum(f.stt_time for f in file_results), 2)
    report.total_audio_duration = round(sum(f.audio_duration for f in file_results), 2)
    if report.total_stt_time > 0:
        report.rtfx = round(report.total_audio_duration / report.total_stt_time, 1)

    # ── DER ────────────────────────────────────────────────────────────
    ders = [f.der for f in file_results if f.der is not None]
    if ders:
        report.mean_der = round(float(np.mean(ders)), 4)

    # ── Per-corpus WER breakdown ───────────────────────────────────────
    corpus_files: dict[str, list[FileResult]] = {}
    for f in file_results:
        corpus_files.setdefault(f.corpus, []).append(f)

    report.corpus_wer = {}
    for corpus, files in sorted(corpus_files.items()):
        refs = [f.normalized_reference for f in files]
        hyps = [f.normalized_hypothesis for f in files]
        report.corpus_wer[corpus] = {
            "wer": round(compute_wer(refs, hyps), 4),
            "files": len(files),
        }

    return report


def report_to_dict(report: EngineReport) -> dict:
    """Convert EngineReport to JSON-serializable dict."""
    return {
        "engine": report.engine,
        "total_files": report.total_files,
        "successful_files": report.successful_files,
        "failed_files": report.failed_files,
        "wer": report.wer,
        "cer": report.cer,
        "rtfx": report.rtfx,
        "total_stt_time_sec": report.total_stt_time,
        "total_audio_duration_sec": report.total_audio_duration,
        "mean_der": report.mean_der,
        "corpus_wer": report.corpus_wer,
        "per_file": [
            {
                "file_id": f.file_id,
                "corpus": f.corpus,
                "wer": round(compute_wer([f.normalized_reference], [f.normalized_hypothesis]), 4),
                "der": f.der,
                "stt_time": f.stt_time,
                "audio_duration": f.audio_duration,
            }
            for f in report.file_results
        ],
    }
