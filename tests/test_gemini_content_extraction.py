"""Unit tests for Gemini response content extraction.

Tests _extract_text_from_content which handles the different formats
returned by langchain-google-genai for Gemini 2.x (string) vs 3.x (list).
"""

import pytest
from benchmark.engines.gemini_engine import _extract_text_from_content


class TestExtractTextFromContent:
    """Tests for _extract_text_from_content helper."""

    def test_plain_string(self):
        """Gemini 2.x returns content as plain string."""
        content = '{"task1_transcripts": [], "task2_speakers": []}'
        result = _extract_text_from_content(content)
        assert result == content

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert _extract_text_from_content("") == ""

    def test_none_input(self):
        """None should return empty string."""
        assert _extract_text_from_content(None) == ""

    def test_list_with_single_text_block(self):
        """Gemini 3.x basic format: list with one text block."""
        content = [{"type": "text", "text": '{"task1_transcripts": []}'}]
        result = _extract_text_from_content(content)
        assert result == '{"task1_transcripts": []}'

    def test_list_with_thinking_and_text_blocks(self):
        """Gemini 3.x with thinking: should extract only text blocks."""
        content = [
            {"type": "thinking", "thinking": "Let me analyze this audio recording..."},
            {"type": "text", "text": '{"task1_transcripts": [{"start": 0, "end": 5}]}'},
        ]
        result = _extract_text_from_content(content)
        assert result == '{"task1_transcripts": [{"start": 0, "end": 5}]}'

    def test_list_with_multiple_text_blocks(self):
        """Multiple text blocks should be concatenated."""
        content = [
            {"type": "text", "text": '{"task1'},
            {"type": "text", "text": '_transcripts": []}'},
        ]
        result = _extract_text_from_content(content)
        assert result == '{"task1_transcripts": []}'

    def test_list_with_plain_strings(self):
        """List of plain strings (older format) should be concatenated."""
        content = ["hello", " world"]
        result = _extract_text_from_content(content)
        assert result == "hello world"

    def test_empty_list(self):
        """Empty list should return empty string."""
        assert _extract_text_from_content([]) == ""

    def test_list_with_only_thinking_blocks(self):
        """List with only thinking blocks (no text) should return empty."""
        content = [
            {"type": "thinking", "thinking": "reasoning..."},
        ]
        result = _extract_text_from_content(content)
        assert result == ""

    def test_list_with_unknown_block_types(self):
        """Unknown block types should be ignored."""
        content = [
            {"type": "executable_code", "code": "print('hi')"},
            {"type": "text", "text": "actual content"},
        ]
        result = _extract_text_from_content(content)
        assert result == "actual content"

    def test_dict_without_type_key(self):
        """Dict blocks without 'type' key should be ignored."""
        content = [
            {"data": "something"},
            {"type": "text", "text": "real content"},
        ]
        result = _extract_text_from_content(content)
        assert result == "real content"

    def test_non_string_non_list_fallback(self):
        """Non-string, non-list types should be coerced to string."""
        result = _extract_text_from_content(12345)
        assert result == "12345"

    def test_text_block_with_empty_text(self):
        """Text block with empty string text."""
        content = [{"type": "text", "text": ""}]
        result = _extract_text_from_content(content)
        assert result == ""

    def test_realistic_gemini_31_pro_response(self):
        """Simulate a realistic Gemini 3.1 Pro response with thinking + JSON."""
        json_response = (
            '{"task1_transcripts": [{"start": 0.5, "end": 3.2, "text": "Hello class", "voice": 1}], '
            '"task2_speakers": [{"voice": 1, "name": "Teacher", "role": "teacher"}], '
            '"detected_language": "en"}'
        )
        content = [
            {"type": "thinking", "thinking": "I need to listen to this audio and transcribe it..."},
            {"type": "text", "text": json_response},
        ]
        result = _extract_text_from_content(content)
        assert result == json_response

        # Verify it's valid JSON
        import json
        parsed = json.loads(result)
        assert "task1_transcripts" in parsed
        assert "task2_speakers" in parsed
        assert parsed["detected_language"] == "en"
