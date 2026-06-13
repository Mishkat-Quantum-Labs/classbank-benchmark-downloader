"""Run ASR benchmark using local open-source models (WhisperX + DiCoW).

Usage:
    python run_benchmark_local.py                    # Run both engines
    python run_benchmark_local.py --engine whisperx  # WhisperX only
    python run_benchmark_local.py --engine dicow     # DiCoW only
    python run_benchmark_local.py --device cuda      # Use GPU
    python run_benchmark_local.py --corpus DISPEL    # Specific corpus only
    python run_benchmark_local.py --model medium     # Smaller Whisper model (faster)

Requires:
    pip install -e ".[local]"
    HF_TOKEN environment variable set (for WhisperX diarization)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from benchmark.config import PARSED_DIR, MEDIA_DIR, RESULTS_DIR
from benchmark.evaluation import normalize_text, compute_wer, FileResult
from benchmark.statistics import compute_engine_report, report_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def find_media_file(corpus: str, stem: str) -> str | None:
    """Find media file matching a parsed JSON stem."""
    extensions = [".mp4", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"]
    for ext in extensions:
        candidate = MEDIA_DIR / corpus / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def run_engine_benchmark(engine, engine_name: str, corpora: list[str]) -> dict:
    """Run benchmark for a single engine across specified corpora."""
    file_results = []
    total_attempted = 0

    for corpus in corpora:
        corpus_parsed_dir = PARSED_DIR / corpus
        if not corpus_parsed_dir.exists():
            logger.warning("Corpus directory not found: %s", corpus_parsed_dir)
            continue

        parsed_files = sorted(corpus_parsed_dir.glob("*.json"))
        if not parsed_files:
            logger.warning("No parsed JSON files in: %s", corpus_parsed_dir)
            continue

        logger.info("Processing corpus: %s (%d files)", corpus, len(parsed_files))

        for parsed_file in parsed_files:
            total_attempted += 1
            file_id = f"{corpus}/{parsed_file.stem}"

            # Load ground truth
            with open(parsed_file, encoding="utf-8") as f:
                ground_truth = json.load(f)

            reference_text = ground_truth.get("full_text", "")
            if not reference_text.strip():
                logger.warning("  ⚠ Empty reference text for %s, skipping", file_id)
                continue

            # Find media file
            media_path = find_media_file(corpus, parsed_file.stem)
            if not media_path:
                logger.warning("  ⚠ No media file for %s, skipping", file_id)
                continue

            # Transcribe
            logger.info("  Transcribing %s...", file_id)
            try:
                result = engine.transcribe(media_path)
            except Exception as e:
                logger.error("  ✗ Failed %s: %s", file_id, e)
                continue

            # Build hypothesis text
            hypothesis_text = " ".join(
                seg["text"] for seg in result.get("segments", [])
            )

            # Normalize texts
            norm_ref = normalize_text(reference_text)
            norm_hyp = normalize_text(hypothesis_text)

            # Compute per-file WER for logging
            per_file_wer = compute_wer([norm_ref], [norm_hyp])
            logger.info(
                "  ✓ %s — WER: %.2f%% | Speakers: %d | Time: %.1fs",
                file_id, per_file_wer * 100, result.get("speakers", 0),
                result.get("stt_time", 0),
            )

            file_results.append(FileResult(
                file_id=file_id,
                corpus=corpus,
                normalized_reference=norm_ref,
                normalized_hypothesis=norm_hyp,
                stt_time=result.get("stt_time", 0),
                audio_duration=0,
                der=None,
            ))

    # Aggregate report
    report = compute_engine_report(file_results, engine_name, total_attempted)
    return report_to_dict(report)


def main():
    parser = argparse.ArgumentParser(
        description="Run ASR benchmark with local open-source models"
    )
    parser.add_argument(
        "--engine", choices=["whisperx", "dicow", "both"], default="both",
        help="Which engine to run (default: both)"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device for inference (default: cpu)"
    )
    parser.add_argument(
        "--model", default="large-v3",
        help="Whisper model size for WhisperX (default: large-v3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for WhisperX (default: 4, reduce if low memory)"
    )
    parser.add_argument(
        "--corpus", nargs="*", default=None,
        help="Specific corpora to benchmark (default: all available)"
    )
    args = parser.parse_args()

    # Determine corpora
    if args.corpus:
        corpora = args.corpus
    else:
        # Auto-discover corpora that have both parsed JSON and media
        corpora = []
        if PARSED_DIR.exists():
            for corpus_dir in sorted(PARSED_DIR.iterdir()):
                if corpus_dir.is_dir() and any(corpus_dir.glob("*.json")):
                    corpora.append(corpus_dir.name)

    if not corpora:
        logger.error("No corpora found in %s", PARSED_DIR)
        sys.exit(1)

    logger.info("Corpora to benchmark: %s", corpora)
    logger.info("Device: %s", args.device)

    results = {}
    t0 = time.time()

    # ── WhisperX ────────────────────────────────────────────────────────
    if args.engine in ("whisperx", "both"):
        try:
            from benchmark.engines import WhisperXEngine

            compute_type = "float16" if args.device == "cuda" else "int8"
            logger.info("=" * 60)
            logger.info("WhisperX (%s) — device=%s, batch_size=%d",
                        args.model, args.device, args.batch_size)
            logger.info("=" * 60)

            engine = WhisperXEngine(
                model_name=args.model,
                device=args.device,
                batch_size=args.batch_size,
                compute_type=compute_type,
            )
            results["whisperx_large_v3"] = run_engine_benchmark(
                engine, "whisperx_large_v3", corpora
            )
        except ImportError as e:
            logger.error("WhisperX not installed: %s", e)
            logger.error("Install with: pip install -e \".[whisperx]\"")
        except Exception as e:
            logger.error("WhisperX failed: %s", e)

    # ── DiCoW ───────────────────────────────────────────────────────────
    if args.engine in ("dicow", "both"):
        try:
            from benchmark.engines import DiCoWEngine

            logger.info("=" * 60)
            logger.info("DiCoW v3 MLC — device=%s", args.device)
            logger.info("=" * 60)

            engine = DiCoWEngine(device=args.device)
            results["dicow_v3_mlc"] = run_engine_benchmark(
                engine, "dicow_v3_mlc", corpora
            )
        except ImportError as e:
            logger.error("DiCoW not installed: %s", e)
            logger.error("Install with: pip install -e \".[dicow]\"")
        except Exception as e:
            logger.error("DiCoW failed: %s", e)

    # ── Save results ────────────────────────────────────────────────────
    total_time = time.time() - t0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "total_time_sec": round(total_time, 1),
        "engine_reports": results,
    }

    output_path = RESULTS_DIR / "local_models_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for engine_name, report in results.items():
        print(f"\n  {engine_name}:")
        print(f"    WER:  {report['wer'] * 100:.2f}%")
        print(f"    CER:  {report['cer'] * 100:.2f}%")
        print(f"    Files: {report['successful_files']}/{report['total_files']}")
        if report.get("corpus_wer"):
            for corpus, data in report["corpus_wer"].items():
                print(f"    {corpus}: {data['wer'] * 100:.2f}% ({data['files']} files)")

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
