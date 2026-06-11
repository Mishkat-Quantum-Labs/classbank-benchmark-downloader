"""Unit tests for diarization error rate computation."""

import pytest
from benchmark.diarization import compute_diarization_error_rate


class TestComputeDER:
    """Tests for compute_diarization_error_rate with various edge cases."""

    def test_perfect_match(self):
        """Identical segments should give DER = 0.0."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
            {"start": 5.0, "end": 10.0, "speaker": "Student"},
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
            {"start": 5.0, "end": 10.0, "speaker": "Student"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        assert result == 0.0

    def test_completely_wrong_speakers(self):
        """Mismatched speakers with 3+ speakers should give DER > 0."""
        # With only 2 speakers, Hungarian algorithm can optimally remap labels.
        # Need 3+ speakers with wrong mapping to get DER > 0.
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
            {"start": 5.0, "end": 10.0, "speaker": "Student1"},
            {"start": 10.0, "end": 15.0, "speaker": "Student2"},
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": "Student1"},
            {"start": 5.0, "end": 10.0, "speaker": "Student2"},
            {"start": 10.0, "end": 15.0, "speaker": "Student1"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        assert result is not None
        assert result > 0.0

    def test_none_start_in_reference(self):
        """Segments with None start in reference should be skipped, not crash."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
            {"start": None, "end": 10.0, "speaker": "Student"},
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        # Should not crash — returns a valid DER or None
        assert result is None or isinstance(result, float)

    def test_none_end_in_hypothesis(self):
        """Segments with None end in hypothesis should be skipped, not crash."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
        ]
        hyp = [
            {"start": 0.0, "end": None, "speaker": "Teacher"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        # All hyp segments are invalid → no valid hypothesis → None
        assert result is None

    def test_none_start_and_end_in_hypothesis(self):
        """Segments with both None start and end should be skipped."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
            {"start": 5.0, "end": 10.0, "speaker": "Student"},
        ]
        hyp = [
            {"start": None, "end": None, "speaker": "Teacher"},
            {"start": 5.0, "end": 10.0, "speaker": "Student"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        # Should compute DER from the one valid hypothesis segment
        assert result is not None
        assert isinstance(result, float)

    def test_mixed_valid_and_none_segments(self):
        """Mix of valid and None segments should compute DER from valid ones."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
            {"start": 5.0, "end": 10.0, "speaker": "Student"},
            {"start": 10.0, "end": 15.0, "speaker": "Teacher"},
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
            {"start": None, "end": None, "speaker": "Student"},
            {"start": 10.0, "end": 15.0, "speaker": "Teacher"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        assert result is not None
        assert isinstance(result, float)

    def test_string_timestamps(self):
        """String timestamps that are valid floats should work."""
        ref = [
            {"start": "0.0", "end": "5.0", "speaker": "Teacher"},
        ]
        hyp = [
            {"start": "0.0", "end": "5.0", "speaker": "Teacher"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        assert result == 0.0

    def test_invalid_string_timestamps(self):
        """Non-numeric string timestamps should be skipped."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
        ]
        hyp = [
            {"start": "invalid", "end": "bad", "speaker": "Teacher"},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        assert result is None

    def test_empty_reference(self):
        """Empty reference should return None."""
        result = compute_diarization_error_rate([], [{"start": 0, "end": 5, "speaker": "T"}])
        assert result is None

    def test_empty_hypothesis(self):
        """Empty hypothesis should return None."""
        result = compute_diarization_error_rate([{"start": 0, "end": 5, "speaker": "T"}], [])
        assert result is None

    def test_missing_keys(self):
        """Segments missing required keys should return None."""
        ref = [{"start": 0.0, "end": 5.0}]  # missing "speaker"
        hyp = [{"start": 0.0, "end": 5.0, "speaker": "Teacher"}]
        result = compute_diarization_error_rate(ref, hyp)
        assert result is None

    def test_none_speaker_skipped(self):
        """Segments with None speaker should be skipped."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": None},
        ]
        result = compute_diarization_error_rate(ref, hyp)
        assert result is None

    def test_end_before_start_skipped(self):
        """Segments where end < start should be skipped."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Teacher"},
        ]
        hyp = [
            {"start": 10.0, "end": 2.0, "speaker": "Teacher"},  # invalid: end < start
        ]
        result = compute_diarization_error_rate(ref, hyp)
        assert result is None
