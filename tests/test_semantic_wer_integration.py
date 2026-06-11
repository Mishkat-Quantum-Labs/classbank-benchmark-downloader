"""Integration tests for semantic WER — mock Bedrock to verify tool-use loop.

Tests the actual multi-turn conversation flow with mocked API responses
to confirm:
1. Extended thinking is disabled for Claude Sonnet 4.5
2. All tool_use blocks get tool_result responses
3. Unknown tools get error results
4. Retry logic works on ValidationException
"""

import pytest
from unittest.mock import patch, MagicMock

from benchmark.semantic_wer import (
    evaluate_semantic_wer,
    compute_semantic_wer,
    SemanticWERResult,
)


def _make_converse_response(content_blocks, stop_reason="end_turn"):
    """Helper to build a mock Converse API response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content_blocks,
            }
        },
        "stopReason": stop_reason,
    }


class TestToolUseLoop:
    """Test the multi-turn tool-use conversation loop."""

    @patch("benchmark.semantic_wer._get_bedrock_client")
    def test_successful_tool_call(self, mock_get_client):
        """Model calls calculate_wer tool successfully."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First response: model calls the calculate_wer tool
        tool_use_response = _make_converse_response(
            [
                {"text": "Let me analyze these texts..."},
                {
                    "toolUse": {
                        "toolUseId": "tool_123",
                        "name": "calculate_wer",
                        "input": {
                            "substitutions": 2,
                            "deletions": 1,
                            "insertions": 1,
                            "reference_words": 10,
                            "normalized_reference": "hello world test",
                            "normalized_hypothesis": "hello world check",
                        },
                    }
                },
            ],
            stop_reason="tool_use",
        )

        # Second response: model acknowledges the result
        final_response = _make_converse_response(
            [{"text": "The WER is 40%."}],
            stop_reason="end_turn",
        )

        mock_client.converse.side_effect = [tool_use_response, final_response]

        result = evaluate_semantic_wer("hello world test foo bar baz one two three four", "hello world check foo bar baz one two three four")

        assert isinstance(result, SemanticWERResult)
        assert result.wer == 0.4  # (2+1+1)/10
        assert result.substitutions == 2
        assert result.deletions == 1
        assert result.insertions == 1
        assert result.reference_words == 10

    @patch("benchmark.semantic_wer._get_bedrock_client")
    def test_thinking_disabled_in_request(self, mock_get_client):
        """Verify that extended thinking is disabled in the API request."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Model calls the tool
        tool_use_response = _make_converse_response(
            [
                {
                    "toolUse": {
                        "toolUseId": "tool_456",
                        "name": "calculate_wer",
                        "input": {
                            "substitutions": 0,
                            "deletions": 0,
                            "insertions": 0,
                            "reference_words": 5,
                        },
                    }
                },
            ],
            stop_reason="tool_use",
        )

        final_response = _make_converse_response(
            [{"text": "Perfect match."}],
            stop_reason="end_turn",
        )

        mock_client.converse.side_effect = [tool_use_response, final_response]

        evaluate_semantic_wer("one two three four five", "one two three four five")

        # Check the first converse call had thinking disabled
        first_call_kwargs = mock_client.converse.call_args_list[0][1]
        assert "additionalModelRequestFields" in first_call_kwargs
        assert first_call_kwargs["additionalModelRequestFields"] == {
            "thinking": {"type": "disabled"}
        }

    @patch("benchmark.semantic_wer._get_bedrock_client")
    def test_unknown_tool_gets_error_result(self, mock_get_client):
        """Unknown tool_use blocks get an error tool_result back."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Model calls an unknown tool first, then calculate_wer
        unknown_tool_response = _make_converse_response(
            [
                {
                    "toolUse": {
                        "toolUseId": "tool_unknown",
                        "name": "some_other_tool",
                        "input": {"foo": "bar"},
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "tool_wer",
                        "name": "calculate_wer",
                        "input": {
                            "substitutions": 1,
                            "deletions": 0,
                            "insertions": 0,
                            "reference_words": 5,
                        },
                    }
                },
            ],
            stop_reason="tool_use",
        )

        final_response = _make_converse_response(
            [{"text": "Done."}],
            stop_reason="end_turn",
        )

        mock_client.converse.side_effect = [unknown_tool_response, final_response]

        result = evaluate_semantic_wer("one two three four five", "one two three four six")

        assert result.wer == 0.2  # 1/5
        assert result.substitutions == 1

        # Verify the tool_result message was sent with BOTH results
        second_call_kwargs = mock_client.converse.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # Last user message should contain tool_results
        last_user_msg = messages[-1]
        assert last_user_msg["role"] == "user"
        tool_results = last_user_msg["content"]
        assert len(tool_results) == 2
        # First is error for unknown tool
        assert tool_results[0]["toolResult"]["toolUseId"] == "tool_unknown"
        assert tool_results[0]["toolResult"]["status"] == "error"
        # Second is success for calculate_wer
        assert tool_results[1]["toolResult"]["toolUseId"] == "tool_wer"


class TestRetryLogic:
    """Test retry behavior in compute_semantic_wer."""

    @patch("benchmark.semantic_wer.evaluate_semantic_wer")
    def test_retries_on_validation_exception(self, mock_eval):
        """Should retry on ValidationException and succeed."""
        from botocore.exceptions import ClientError

        # First call raises ValidationException, second succeeds
        mock_eval.side_effect = [
            Exception("An error occurred (ValidationException) when calling the Converse operation: ..."),
            SemanticWERResult(
                wer=0.15, substitutions=3, deletions=0, insertions=0,
                reference_words=20, total_errors=3,
            ),
        ]

        result = compute_semantic_wer("some reference text", "some hypothesis text")
        assert result == 0.15
        assert mock_eval.call_count == 2

    @patch("benchmark.semantic_wer.evaluate_semantic_wer")
    def test_returns_none_after_max_retries(self, mock_eval):
        """Should return None after exhausting retries."""
        mock_eval.side_effect = Exception(
            "An error occurred (ValidationException) when calling the Converse operation"
        )

        result = compute_semantic_wer("some reference text", "some hypothesis text")
        assert result is None
        assert mock_eval.call_count == 3  # 3 attempts

    @patch("benchmark.semantic_wer.evaluate_semantic_wer")
    def test_no_retry_on_non_retryable_error(self, mock_eval):
        """Should not retry on non-retryable errors."""
        mock_eval.side_effect = Exception("AccessDeniedException: not authorized")

        result = compute_semantic_wer("some reference text", "some hypothesis text")
        assert result is None
        assert mock_eval.call_count == 1  # No retry
