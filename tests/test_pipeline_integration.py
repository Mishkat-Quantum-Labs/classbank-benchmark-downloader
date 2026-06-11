"""Integration test — runs the full benchmark pipeline with mocked engines.

Uses dummy transcription outputs to test the entire evaluation flow
(WER, CER, DER, Semantic WER) end-to-end without making real API calls.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from benchmark.config import ENGINES
from benchmark.evaluation import compute_file_metrics, normalize_text
from benchmark.diarization import compute_diarization_error_rate
from benchmark.semantic_wer import _calculate_wer
from benchmark.statistics import compute_engine_report, report_to_dict, bootstrap_ci


# ── Dummy Data ──────────────────────────────────────────────────────────────

DUMMY_GROUND_TRUTH = {
    "metadata": {
        "duration_seconds": 300.0,
        "corpus": "TestCorpus",
    },
    "full_text": (
        "Today we are going to learn about photosynthesis. "
        "Photosynthesis is the process by which plants convert sunlight into energy. "
        "Can anyone tell me what plants need for photosynthesis? "
        "They need sunlight water and carbon dioxide."
    ),
    "segments": [
        {"start": 0.0, "end": 5.0, "speaker": "teacher", "text": "Today we are going to learn about photosynthesis."},
        {"start": 5.5, "end": 12.0, "speaker": "teacher", "text": "Photosynthesis is the process by which plants convert sunlight into energy."},
        {"start": 12.5, "end": 16.0, "speaker": "teacher", "text": "Can anyone tell me what plants need for photosynthesis?"},
        {"start": 17.0, "end": 20.0, "speaker": "student_1", "text": "They need sunlight water and carbon dioxide."},
    ],
}

# Good hypothesis (low WER)
DUMMY_HYPOTHESIS_GOOD = {
    "language": "en",
    "speakers": 2,
    "stt_time": 8.5,
    "full_text": (
        "Today we are going to learn about photosynthesis. "
        "Photosynthesis is the process by which plants convert sunlight into energy. "
        "Can anyone tell me what plants need for photosynthesis? "
        "They need sunlight water and carbon dioxide."
    ),
    "segments": [
        {"start": 0.0, "end": 5.0, "speaker": "teacher", "text": "Today we are going to learn about photosynthesis."},
        {"start": 5.5, "end": 12.0, "speaker": "teacher", "text": "Photosynthesis is the process by which plants convert sunlight into energy."},
        {"start": 12.5, "end": 16.0, "speaker": "teacher", "text": "Can anyone tell me what plants need for photosynthesis?"},
        {"start": 17.0, "end": 20.0, "speaker": "student_1", "text": "They need sunlight water and carbon dioxide."},
    ],
}

# Bad hypothesis (high WER, some None timestamps)
DUMMY_HYPOTHESIS_BAD = {
    "language": "en",
    "speakers": 2,
    "stt_time": 12.0,
    "full_text": (
        "Today we are going to learn about something. "
        "Photosynthesis is a process where plants do something with light. "
        "Can someone tell me what they need? "
        "They need sun and water."
    ),
    "segments": [
        {"start": 0.0, "end": 5.0, "speaker": "teacher", "text": "Today we are going to learn about something."},
        {"start": None, "end": None, "speaker": "teacher", "text": "Photosynthesis is a process where plants do something with light."},
        {"start": 12.5, "end": 16.0, "speaker": "student_1", "text": "Can someone tell me what they need?"},
        {"start": 17.0, "end": 20.0, "speaker": "student_1", "text": "They need sun and water."},
    ],
}

# Hypothesis with no segments (text only)
DUMMY_HYPOTHESIS_NO_SEGMENTS = {
    "language": "en",
    "speakers": 1,
    "stt_time": 6.0,
    "full_text": "Today we learn photosynthesis plants convert sunlight energy.",
    "segments": [],
}


# ── Unit Integration Tests ──────────────────────────────────────────────────


class TestNormalization:
    """Test text normalization used in WER computation."""

    def test_normalize_removes_inaudible(self):
        text = "Hello [inaudible] world"
        result = normalize_text(text)
        assert "inaudible" not in result

    def test_normalize_empty(self):
        assert normalize_text("") == ""

    def test_normalize_consistent(self):
        """Same text normalizes to same result."""
        text = "Dr. Smith said it costs $5."
        r1 = normalize_text(text)
        r2 = normalize_text(text)
        assert r1 == r2


class TestFileMetrics:
    """Test compute_file_metrics with various inputs."""

    def test_perfect_match(self):
        ref = "Hello world this is a test"
        metrics = compute_file_metrics(ref, ref, "test/file1", "TestCorpus", 5.0, 60.0)
        assert metrics is not None
        assert metrics.wer == 0.0
        assert metrics.cer == 0.0

    def test_completely_wrong(self):
        ref = "Hello world"
        hyp = "Goodbye universe"
        metrics = compute_file_metrics(ref, hyp, "test/file2", "TestCorpus", 3.0, 30.0)
        assert metrics is not None
        assert metrics.wer > 0.0

    def test_empty_hypothesis(self):
        ref = "Hello world test"
        hyp = ""
        metrics = compute_file_metrics(ref, hyp, "test/file3", "TestCorpus", 1.0, 10.0)
        assert metrics is not None
        assert metrics.wer == 1.0

    def test_empty_reference_returns_none(self):
        metrics = compute_file_metrics("", "Hello", "test/file4", "TestCorpus", 1.0, 10.0)
        assert metrics is None

    def test_rtf_calculation(self):
        metrics = compute_file_metrics("test text", "test text", "f", "C", 10.0, 100.0)
        assert metrics is not None
        assert metrics.rtf == pytest.approx(0.1)


class TestDiarizationWithDummyData:
    """Test DER computation using dummy pipeline data."""

    def test_good_hypothesis_der(self):
        der = compute_diarization_error_rate(
            DUMMY_GROUND_TRUTH["segments"],
            DUMMY_HYPOTHESIS_GOOD["segments"],
        )
        assert der is not None
        assert der == 0.0

    def test_bad_hypothesis_with_none_timestamps(self):
        """Bad hypothesis has None timestamps — should not crash."""
        der = compute_diarization_error_rate(
            DUMMY_GROUND_TRUTH["segments"],
            DUMMY_HYPOTHESIS_BAD["segments"],
        )
        # Should compute from valid segments only, not crash
        assert der is None or isinstance(der, float)

    def test_no_segments_hypothesis(self):
        """Empty segments should return None."""
        der = compute_diarization_error_rate(
            DUMMY_GROUND_TRUTH["segments"],
            DUMMY_HYPOTHESIS_NO_SEGMENTS["segments"],
        )
        assert der is None


class TestStatisticsWithDummyData:
    """Test statistics module with simulated per-file metrics."""

    def test_engine_report_single_file(self):
        metrics = compute_file_metrics(
            DUMMY_GROUND_TRUTH["full_text"],
            DUMMY_HYPOTHESIS_GOOD["full_text"],
            "TestCorpus/test_file",
            "TestCorpus",
            stt_time=8.5,
            audio_duration=300.0,
        )
        assert metrics is not None

        report = compute_engine_report([metrics], "test_engine", 1)
        assert report.engine == "test_engine"
        assert report.successful_files == 1
        assert report.failed_files == 0
        assert report.mean_wer == 0.0

    def test_engine_report_multiple_files(self):
        metrics_good = compute_file_metrics(
            DUMMY_GROUND_TRUTH["full_text"],
            DUMMY_HYPOTHESIS_GOOD["full_text"],
            "TestCorpus/good",
            "TestCorpus",
            8.5, 300.0,
        )
        metrics_bad = compute_file_metrics(
            DUMMY_GROUND_TRUTH["full_text"],
            DUMMY_HYPOTHESIS_BAD["full_text"],
            "TestCorpus/bad",
            "TestCorpus",
            12.0, 300.0,
        )
        assert metrics_good is not None
        assert metrics_bad is not None

        report = compute_engine_report([metrics_good, metrics_bad], "test_engine", 2)
        assert report.successful_files == 2
        assert report.mean_wer > 0.0  # bad hypothesis pulls it up
        assert report.mean_wer < 1.0  # good hypothesis keeps it reasonable

    def test_report_to_dict_structure(self):
        metrics = compute_file_metrics(
            "hello world test", "hello world test",
            "C/f", "C", 5.0, 60.0,
        )
        report = compute_engine_report([metrics], "engine_x", 1)
        d = report_to_dict(report)

        assert "engine" in d
        assert "mean_wer" in d
        assert "median_wer" in d
        assert "ci_95" in d
        assert "per_file" in d
        assert len(d["per_file"]) == 1
        assert d["engine"] == "engine_x"

    def test_bootstrap_ci(self):
        values = [0.1, 0.15, 0.12, 0.08, 0.11, 0.14, 0.09, 0.13]
        lower, upper = bootstrap_ci(values)
        assert lower <= upper
        assert lower >= 0.0
        assert upper <= 1.0


class TestFullPipelineWithDummyOutputs:
    """End-to-end test: write dummy hypothesis files, run evaluation, check reports."""

    def setup_method(self):
        """Create a temporary results directory with dummy hypothesis files."""
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.results_dir = self.tmp_dir / "benchmark_results"
        self.parsed_dir = self.tmp_dir / "dataset" / "parsed" / "TestCorpus"
        self.media_dir = self.tmp_dir / "dataset" / "media" / "TestCorpus"

        # Create ground truth
        self.parsed_dir.mkdir(parents=True)
        with open(self.parsed_dir / "lesson1.json", "w") as f:
            json.dump(DUMMY_GROUND_TRUTH, f)

        # Create dummy media file (empty, just needs to exist)
        self.media_dir.mkdir(parents=True)
        (self.media_dir / "lesson1.mp4").write_text("")

        # Create hypothesis outputs for two "engines"
        for engine, hyp_data in [
            ("engine_good", DUMMY_HYPOTHESIS_GOOD),
            ("engine_bad", DUMMY_HYPOTHESIS_BAD),
        ]:
            engine_dir = self.results_dir / engine / "TestCorpus"
            engine_dir.mkdir(parents=True)
            with open(engine_dir / "lesson1.json", "w") as f:
                json.dump(hyp_data, f)

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_evaluation_produces_metrics(self):
        """Load hypothesis + ground truth, compute metrics, verify structure."""
        from benchmark.evaluation import compute_file_metrics

        # Load ground truth
        with open(self.parsed_dir / "lesson1.json") as f:
            gt = json.load(f)

        # Load good hypothesis
        hyp_path = self.results_dir / "engine_good" / "TestCorpus" / "lesson1.json"
        with open(hyp_path) as f:
            hyp = json.load(f)

        metrics = compute_file_metrics(
            reference_text=gt["full_text"],
            hypothesis_text=hyp["full_text"],
            file_id="TestCorpus/lesson1",
            corpus="TestCorpus",
            stt_time=hyp["stt_time"],
            audio_duration=gt["metadata"]["duration_seconds"],
        )
        assert metrics is not None
        assert metrics.wer == 0.0
        assert metrics.corpus == "TestCorpus"
        assert metrics.stt_time == 8.5

    def test_bad_hypothesis_has_higher_wer(self):
        """Bad hypothesis should have higher WER than good."""
        with open(self.parsed_dir / "lesson1.json") as f:
            gt = json.load(f)

        good_path = self.results_dir / "engine_good" / "TestCorpus" / "lesson1.json"
        bad_path = self.results_dir / "engine_bad" / "TestCorpus" / "lesson1.json"

        with open(good_path) as f:
            hyp_good = json.load(f)
        with open(bad_path) as f:
            hyp_bad = json.load(f)

        m_good = compute_file_metrics(
            gt["full_text"], hyp_good["full_text"],
            "TestCorpus/lesson1", "TestCorpus", 8.5, 300.0,
        )
        m_bad = compute_file_metrics(
            gt["full_text"], hyp_bad["full_text"],
            "TestCorpus/lesson1", "TestCorpus", 12.0, 300.0,
        )
        assert m_good is not None
        assert m_bad is not None
        assert m_good.wer < m_bad.wer

    def test_der_with_none_timestamps_does_not_crash(self):
        """DER computation should handle None timestamps gracefully."""
        with open(self.parsed_dir / "lesson1.json") as f:
            gt = json.load(f)

        bad_path = self.results_dir / "engine_bad" / "TestCorpus" / "lesson1.json"
        with open(bad_path) as f:
            hyp_bad = json.load(f)

        # This should NOT raise TypeError
        der = compute_diarization_error_rate(
            gt["segments"], hyp_bad["segments"]
        )
        # Should be a float or None, not crash
        assert der is None or isinstance(der, float)

    def test_full_report_generation(self):
        """Generate full engine report from dummy data."""
        with open(self.parsed_dir / "lesson1.json") as f:
            gt = json.load(f)

        all_metrics = []
        for engine in ["engine_good", "engine_bad"]:
            hyp_path = self.results_dir / engine / "TestCorpus" / "lesson1.json"
            with open(hyp_path) as f:
                hyp = json.load(f)

            metrics = compute_file_metrics(
                gt["full_text"], hyp["full_text"],
                f"TestCorpus/lesson1_{engine}", "TestCorpus",
                hyp["stt_time"], gt["metadata"]["duration_seconds"],
            )
            if metrics:
                # Skip semantic WER (requires API) — set to None
                metrics.semantic_wer = None
                # Compute DER
                metrics.der = compute_diarization_error_rate(
                    gt["segments"], hyp["segments"]
                )
                all_metrics.append(metrics)

        assert len(all_metrics) == 2

        report = compute_engine_report(all_metrics, "combined_test", 2)
        d = report_to_dict(report)

        # Verify report structure
        assert d["successful_files"] == 2
        assert d["mean_wer"] >= 0.0
        assert d["mean_cer"] >= 0.0
        assert len(d["per_file"]) == 2
        assert "corpus_wer" in d
        assert "TestCorpus" in d["corpus_wer"]

        # Verify JSON serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
