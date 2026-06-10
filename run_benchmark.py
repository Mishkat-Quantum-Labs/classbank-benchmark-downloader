"""Origin ASR Benchmark Pipeline.

Runs STT engines on ClassBank media files, compares against ground truth,
and generates evaluation reports with WER/CER metrics.

Usage:
    # Run all engines on all files
    python run_benchmark.py

    # Run specific engine(s)
    python run_benchmark.py --engines gemini_pro,elevenlabs_scribe_v2

    # Run on specific corpus only
    python run_benchmark.py --corpora APT,Bradford

    # Evaluate only (skip transcription, use existing hypothesis files)
    python run_benchmark.py --evaluate-only

    # Limit number of files (for testing)
    python run_benchmark.py --limit 5
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from benchmark.config import (
    ENGINES,
    GEMINI_MODEL_MAP,
    MEDIA_DIR,
    PARSED_DIR,
    RESULTS_DIR,
)
from benchmark.evaluation import (
    FileMetrics,
    bootstrap_ci,
    compute_engine_report,
    compute_file_metrics,
    pairwise_comparison,
    report_to_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ── File Discovery ──────────────────────────────────────────────────────────


def discover_media_files(corpora: list[str] | None = None) -> list[tuple[str, Path]]:
    """Find all media files with matching ground truth.

    Returns:
        List of (corpus, media_path) tuples where ground truth exists.
    """
    files = []

    if not MEDIA_DIR.exists():
        logger.error("Media directory not found: %s", MEDIA_DIR)
        sys.exit(1)

    for corpus_dir in sorted(MEDIA_DIR.iterdir()):
        if not corpus_dir.is_dir():
            continue
        corpus = corpus_dir.name

        # Filter by corpus if specified
        if corpora and corpus not in corpora:
            continue

        # Check if ground truth exists for this corpus
        parsed_corpus_dir = PARSED_DIR / corpus
        if not parsed_corpus_dir.exists():
            logger.warning("No ground truth for corpus '%s' — skipping", corpus)
            continue

        for media_file in sorted(corpus_dir.iterdir()):
            if media_file.suffix.lower() not in (".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac"):
                continue

            # Check matching ground truth JSON exists
            gt_path = parsed_corpus_dir / f"{media_file.stem}.json"
            if not gt_path.exists():
                logger.debug("No ground truth for %s/%s — skipping", corpus, media_file.name)
                continue

            files.append((corpus, media_file))

    logger.info("Discovered %d media files with ground truth", len(files))
    return files


def load_ground_truth(corpus: str, file_stem: str) -> dict | None:
    """Load ground truth JSON for a file.

    Returns:
        Parsed JSON dict with metadata, segments, full_text — or None.
    """
    gt_path = PARSED_DIR / corpus / f"{file_stem}.json"
    if not gt_path.exists():
        return None
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Engine Factory ──────────────────────────────────────────────────────────


def get_engine(engine_key: str):
    """Create an engine instance by key.

    Returns:
        An engine object with a .transcribe(audio_path) method.
    """
    if engine_key in GEMINI_MODEL_MAP:
        from benchmark.engines.gemini_engine import GeminiEngine
        return GeminiEngine(engine_key)
    elif engine_key == "elevenlabs_scribe_v2":
        from benchmark.engines.elevenlabs_engine import ElevenLabsEngine
        return ElevenLabsEngine()
    else:
        raise ValueError(f"Unknown engine: {engine_key}. Available: {list(ENGINES.keys())}")


# ── Transcription Stage ─────────────────────────────────────────────────────


def run_transcription(
    engine_key: str,
    files: list[tuple[str, Path]],
) -> int:
    """Run an STT engine on all media files and save hypothesis JSONs.

    Args:
        engine_key: Engine identifier.
        files: List of (corpus, media_path) tuples.

    Returns:
        Number of successfully transcribed files.
    """
    engine = get_engine(engine_key)
    output_dir = RESULTS_DIR / engine_key
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    errors = []

    for corpus, media_path in tqdm(files, desc=f"[{engine_key}] Transcribing", unit="file"):
        # Create corpus subdirectory
        corpus_output_dir = output_dir / corpus
        corpus_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = corpus_output_dir / f"{media_path.stem}.json"

        # Skip if already transcribed
        if output_path.exists():
            logger.debug("Already transcribed: %s/%s — skipping", corpus, media_path.name)
            success_count += 1
            continue

        try:
            result = engine.transcribe(str(media_path))

            # Save hypothesis JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                f.write("\n")

            success_count += 1
            logger.info(
                "[%s] ✓ %s/%s (%.1fs, %d segments)",
                engine_key,
                corpus,
                media_path.name,
                result.get("stt_time", 0),
                len(result.get("segments", [])),
            )

        except Exception as e:
            errors.append((f"{corpus}/{media_path.name}", str(e)))
            logger.error("[%s] ✗ %s/%s — %s", engine_key, corpus, media_path.name, e)

    # Write error log
    if errors:
        error_log_path = output_dir / "errors.json"
        with open(error_log_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        logger.warning("[%s] %d errors saved to %s", engine_key, len(errors), error_log_path)

    return success_count


# ── Evaluation Stage ────────────────────────────────────────────────────────


def run_evaluation(
    engine_key: str,
    files: list[tuple[str, Path]],
) -> dict:
    """Evaluate an engine's hypothesis transcripts against ground truth.

    Args:
        engine_key: Engine identifier.
        files: List of (corpus, media_path) tuples.

    Returns:
        Engine report as a dict.
    """
    hypothesis_dir = RESULTS_DIR / engine_key
    if not hypothesis_dir.exists():
        logger.error("No hypothesis results found for engine '%s'", engine_key)
        return {}

    file_metrics: list[FileMetrics] = []
    skipped = 0

    for corpus, media_path in tqdm(files, desc=f"[{engine_key}] Evaluating", unit="file"):
        # Load hypothesis
        hyp_path = hypothesis_dir / corpus / f"{media_path.stem}.json"
        if not hyp_path.exists():
            skipped += 1
            continue

        with open(hyp_path, "r", encoding="utf-8") as f:
            hypothesis = json.load(f)

        # Load ground truth
        ground_truth = load_ground_truth(corpus, media_path.stem)
        if not ground_truth:
            skipped += 1
            continue

        # Extract full text from both
        ref_text = ground_truth.get("full_text", "")
        hyp_text = hypothesis.get("full_text", "")

        # If hypothesis doesn't have full_text, build from segments
        if not hyp_text and "segments" in hypothesis:
            hyp_text = " ".join(
                seg.get("text", "") for seg in hypothesis["segments"]
            )

        # Get timing info
        stt_time = hypothesis.get("stt_time", 0.0)
        audio_duration = ground_truth.get("metadata", {}).get("duration_seconds", 0.0)

        # Compute metrics
        file_id = f"{corpus}/{media_path.stem}"
        metrics = compute_file_metrics(
            reference_text=ref_text,
            hypothesis_text=hyp_text,
            file_id=file_id,
            corpus=corpus,
            stt_time=stt_time,
            audio_duration=audio_duration,
        )

        if metrics:
            file_metrics.append(metrics)

    # Compute aggregate report
    total_attempted = len(files) - skipped
    report = compute_engine_report(file_metrics, engine_key, total_attempted)

    logger.info(
        "[%s] Evaluation complete: %d files, mean WER=%.2f%%, median WER=%.2f%%",
        engine_key,
        report.successful_files,
        report.mean_wer * 100,
        report.median_wer * 100,
    )

    return report_to_dict(report)


# ── Report Generation ───────────────────────────────────────────────────────


def generate_comparison_report(engine_reports: dict[str, dict]) -> dict:
    """Generate a cross-engine comparison report with statistical tests.

    Args:
        engine_reports: Dict mapping engine_key → report dict.

    Returns:
        Comparison report dict with rankings and pairwise tests.
    """
    if len(engine_reports) < 2:
        return {"note": "Need at least 2 engines for comparison"}

    # Rank engines by mean WER
    rankings = sorted(
        engine_reports.items(),
        key=lambda x: x[1].get("mean_wer", 1.0),
    )

    comparison = {
        "rankings": [
            {
                "rank": i + 1,
                "engine": key,
                "mean_wer": report.get("mean_wer", 0),
                "median_wer": report.get("median_wer", 0),
                "ci_95": report.get("ci_95", [0, 0]),
                "mean_cer": report.get("mean_cer", 0),
                "mean_rtf": report.get("mean_rtf", 0),
            }
            for i, (key, report) in enumerate(rankings)
        ],
        "pairwise_tests": [],
    }

    # Pairwise Wilcoxon tests
    engine_keys = list(engine_reports.keys())
    for i in range(len(engine_keys)):
        for j in range(i + 1, len(engine_keys)):
            key_a = engine_keys[i]
            key_b = engine_keys[j]

            # Get per-file WERs (need same files for comparison)
            per_file_a = {
                m["file_id"]: m["wer"]
                for m in engine_reports[key_a].get("per_file", [])
            }
            per_file_b = {
                m["file_id"]: m["wer"]
                for m in engine_reports[key_b].get("per_file", [])
            }

            # Find common files
            common_files = sorted(set(per_file_a.keys()) & set(per_file_b.keys()))
            if len(common_files) < 10:
                continue

            wers_a = [per_file_a[f] for f in common_files]
            wers_b = [per_file_b[f] for f in common_files]

            test_result = pairwise_comparison(wers_a, wers_b)
            test_result["system_a"] = key_a
            test_result["system_b"] = key_b
            test_result["common_files"] = len(common_files)
            comparison["pairwise_tests"].append(test_result)

    return comparison


# ── Main Pipeline ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Origin ASR Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=None,
        help=f"Comma-separated engine keys. Available: {list(ENGINES.keys())}",
    )
    parser.add_argument(
        "--corpora",
        type=str,
        default=None,
        help="Comma-separated corpus names to benchmark (default: all)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip transcription, only run evaluation on existing results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files per engine (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports (default: benchmark_results/)",
    )

    args = parser.parse_args()

    # Parse engine list
    if args.engines:
        engine_keys = [e.strip() for e in args.engines.split(",")]
        for key in engine_keys:
            if key not in ENGINES:
                logger.error("Unknown engine '%s'. Available: %s", key, list(ENGINES.keys()))
                sys.exit(1)
    else:
        engine_keys = list(ENGINES.keys())

    # Parse corpora filter
    corpora = None
    if args.corpora:
        corpora = [c.strip() for c in args.corpora.split(",")]

    # Discover files
    files = discover_media_files(corpora)
    if not files:
        logger.error("No media files found with matching ground truth")
        sys.exit(1)

    # Apply limit
    if args.limit:
        files = files[: args.limit]
        logger.info("Limited to %d files", len(files))

    # Output directory
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ── Transcription Stage ─────────────────────────────────────────────
    if not args.evaluate_only:
        print("\n" + "=" * 60)
        print("  STAGE 1: TRANSCRIPTION")
        print("=" * 60)

        for engine_key in engine_keys:
            print(f"\n→ Running {ENGINES[engine_key]} ({engine_key})…")
            success = run_transcription(engine_key, files)
            print(f"  ✓ {success}/{len(files)} files transcribed")

    # ── Evaluation Stage ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2: EVALUATION")
    print("=" * 60)

    engine_reports: dict[str, dict] = {}

    for engine_key in engine_keys:
        print(f"\n→ Evaluating {ENGINES[engine_key]} ({engine_key})…")
        report = run_evaluation(engine_key, files)
        if report:
            engine_reports[engine_key] = report

            # Save individual engine report
            report_path = output_dir / f"{engine_key}_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
            print(f"  Report saved: {report_path}")

    # ── Comparison Report ───────────────────────────────────────────────
    if len(engine_reports) >= 2:
        print("\n" + "=" * 60)
        print("  STAGE 3: COMPARISON")
        print("=" * 60)

        comparison = generate_comparison_report(engine_reports)
        comparison_path = output_dir / "comparison_report.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
            f.write("\n")
        print(f"\n  Comparison report saved: {comparison_path}")

        # Print ranking table
        print("\n  Engine Rankings (by WER):")
        print("  " + "-" * 50)
        for r in comparison.get("rankings", []):
            print(
                f"  #{r['rank']} {r['engine']:25s} WER={r['mean_wer']*100:.2f}% "
                f"(CI: [{r['ci_95'][0]*100:.2f}%, {r['ci_95'][1]*100:.2f}%])"
            )

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Engines tested: {len(engine_reports)}")
    print(f"  Files per engine: {len(files)}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results directory: {output_dir}")

    # Save combined results
    combined = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engines_tested": list(engine_reports.keys()),
        "files_per_engine": len(files),
        "total_time_sec": round(elapsed, 1),
        "engine_reports": engine_reports,
    }
    if len(engine_reports) >= 2:
        combined["comparison"] = comparison

    combined_path = output_dir / "benchmark_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
        f.write("\n")
    print(f"  Combined results: {combined_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
