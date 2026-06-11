"""Unit tests for semantic WER module — tests the non-API parts."""

import pytest
from benchmark.semantic_wer import _calculate_wer, SemanticWERResult, evaluate_semantic_wer


class TestCalculateWER:
    """Tests for the programmatic _calculate_wer function (no API calls)."""

    def test_perfect_transcription(self):
        """Zero errors should give WER = 0."""
        result = _calculate_wer(substitutions=0, deletions=0, insertions=0, reference_words=50)
        assert result["wer"] == 0.0
        assert result["total_errors"] == 0

    def test_all_substitutions(self):
        """All words wrong should give WER = 1.0."""
        result = _calculate_wer(substitutions=10, deletions=0, insertions=0, reference_words=10)
        assert result["wer"] == 1.0
        assert result["total_errors"] == 10

    def test_mixed_errors(self):
        """Mixed errors: (3 + 2 + 1) / 20 = 0.3."""
        result = _calculate_wer(substitutions=3, deletions=2, insertions=1, reference_words=20)
        assert result["wer"] == pytest.approx(0.3)
        assert result["total_errors"] == 6

    def test_wer_above_one(self):
        """WER can exceed 1.0 when insertions are high."""
        result = _calculate_wer(substitutions=5, deletions=5, insertions=10, reference_words=10)
        assert result["wer"] == 2.0
        assert result["total_errors"] == 20

    def test_zero_reference_words_no_errors(self):
        """Zero reference words with zero errors = 0.0."""
        result = _calculate_wer(substitutions=0, deletions=0, insertions=0, reference_words=0)
        assert result["wer"] == 0.0

    def test_zero_reference_words_with_insertions(self):
        """Zero reference words with insertions = inf."""
        result = _calculate_wer(substitutions=0, deletions=0, insertions=5, reference_words=0)
        assert result["wer"] == float("inf")

    def test_result_keys(self):
        """Result should contain all expected keys."""
        result = _calculate_wer(substitutions=1, deletions=2, insertions=3, reference_words=10)
        assert "wer" in result
        assert "wer_percentage" in result
        assert "substitutions" in result
        assert "deletions" in result
        assert "insertions" in result
        assert "reference_words" in result
        assert "total_errors" in result


class TestEvaluateSemanticWEREdgeCases:
    """Tests for evaluate_semantic_wer with empty/edge case inputs (no API calls)."""

    def test_both_empty(self):
        """Both empty strings should return WER = 0."""
        result = evaluate_semantic_wer("", "")
        assert result.wer == 0.0
        assert result.reference_words == 0

    def test_both_whitespace(self):
        """Both whitespace-only strings should return WER = 0."""
        result = evaluate_semantic_wer("   ", "   ")
        assert result.wer == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        """Empty reference with hypothesis should return WER = inf."""
        result = evaluate_semantic_wer("", "hello world")
        assert result.wer == float("inf")
        assert result.insertions == 2

    def test_nonempty_reference_empty_hypothesis(self):
        """Non-empty reference with empty hypothesis should return WER = 1.0."""
        result = evaluate_semantic_wer("hello world foo", "")
        assert result.wer == 1.0
        assert result.deletions == 3
        assert result.reference_words == 3
