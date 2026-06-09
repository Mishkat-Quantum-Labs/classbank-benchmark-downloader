"""Unit tests for extract_participants function."""

from unittest.mock import MagicMock

from preprocess_transcripts import extract_participants


class TestExtractParticipants:
    """Tests for the extract_participants function."""

    def test_maps_roles_to_lowercase(self):
        """Speaker roles from @Participants header are converted to lowercase."""
        reader = MagicMock()
        p1 = MagicMock()
        p1.code = "TEA"
        p1.role = "Teacher"
        p2 = MagicMock()
        p2.code = "STU"
        p2.role = "Student"
        reader.participants.return_value = [p1, p2]

        result = extract_participants(reader)

        assert result == {"TEA": "teacher", "STU": "student"}

    def test_empty_role_falls_back_to_code(self):
        """When role is empty, uses speaker code lowercased as fallback."""
        reader = MagicMock()
        p1 = MagicMock()
        p1.code = "CHI"
        p1.role = ""
        reader.participants.return_value = [p1]

        result = extract_participants(reader)

        assert result == {"CHI": "chi"}

    def test_no_participants_returns_empty_dict(self):
        """When no @Participants header exists, returns empty dict."""
        reader = MagicMock()
        reader.participants.return_value = []

        result = extract_participants(reader)

        assert result == {}

    def test_mixed_roles_and_empty(self):
        """Handles mix of defined roles and missing roles correctly."""
        reader = MagicMock()
        p1 = MagicMock()
        p1.code = "TEA"
        p1.role = "Teacher"
        p2 = MagicMock()
        p2.code = "STU"
        p2.role = ""
        p3 = MagicMock()
        p3.code = "OTH"
        p3.role = "Other"
        reader.participants.return_value = [p1, p2, p3]

        result = extract_participants(reader)

        assert result == {"TEA": "teacher", "STU": "stu", "OTH": "other"}

    def test_casefold_handles_non_ascii(self):
        """casefold() handles locale-independent case folding (e.g., German ß)."""
        reader = MagicMock()
        p1 = MagicMock()
        p1.code = "TEA"
        p1.role = "Lehrer"  # German for teacher
        reader.participants.return_value = [p1]

        result = extract_participants(reader)

        assert result == {"TEA": "lehrer"}

    def test_role_with_uppercase_code_fallback(self):
        """When role is absent, uppercase speaker code is lowercased."""
        reader = MagicMock()
        p1 = MagicMock()
        p1.code = "INV"
        p1.role = ""
        reader.participants.return_value = [p1]

        result = extract_participants(reader)

        assert result == {"INV": "inv"}
