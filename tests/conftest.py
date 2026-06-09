"""Shared test fixtures for CHA-to-JSON pipeline tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Sample participant data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_participants_with_roles():
    """Participant data with defined roles (code -> role mapping)."""
    return {
        "TEA": "Teacher",
        "STU": "Student",
        "ST2": "Student",
        "OBS": "Observer",
    }


@pytest.fixture
def sample_participants_no_roles():
    """Participant data where roles are missing (should fall back to code)."""
    return {
        "CHI": "",
        "MOT": "",
        "FAT": "",
    }


@pytest.fixture
def sample_participants_mixed():
    """Participant data with mix of defined and missing roles."""
    return {
        "TEA": "Teacher",
        "STU": "",
        "OTH": "Other",
    }


# ---------------------------------------------------------------------------
# Sample utterance data
# ---------------------------------------------------------------------------


@dataclass
class MockUtterance:
    """Mimics pylangacq Utterance structure for testing."""

    participant: str
    tiers: dict
    time_marks: Optional[tuple]
    tokens: list = None

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = []


@pytest.fixture
def sample_utterances_with_timestamps():
    """Utterances with valid time marks (ms integers)."""
    return [
        MockUtterance(
            participant="TEA",
            tiers={"TEA": "this is the photograph of the Saanich Peninsula ."},
            time_marks=(0, 3361),
        ),
        MockUtterance(
            participant="STU",
            tiers={"STU": "yes I can see it ."},
            time_marks=(3500, 5200),
        ),
        MockUtterance(
            participant="TEA",
            tiers={"TEA": "and basically what you see here ."},
            time_marks=(5300, 8100),
        ),
    ]


@pytest.fixture
def sample_utterances_no_timestamps():
    """Utterances with no time marks (None)."""
    return [
        MockUtterance(
            participant="TEA",
            tiers={"TEA": "welcome to class today ."},
            time_marks=None,
        ),
        MockUtterance(
            participant="STU",
            tiers={"STU": "thank you ."},
            time_marks=None,
        ),
    ]


@pytest.fixture
def sample_utterances_mixed_timestamps():
    """Utterances with a mix of present and absent time marks."""
    return [
        MockUtterance(
            participant="TEA",
            tiers={"TEA": "let us begin ."},
            time_marks=(0, 2000),
        ),
        MockUtterance(
            participant="STU",
            tiers={"STU": "okay ."},
            time_marks=None,
        ),
        MockUtterance(
            participant="TEA",
            tiers={"TEA": "open your books ."},
            time_marks=(3000, 5000),
        ),
    ]


# ---------------------------------------------------------------------------
# Sample text strings (cleaned and uncleaned)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_uncleaned_texts():
    """Raw CHAT-format text strings with various annotations."""
    return {
        "inline_timestamps": "\u00150_3361\u0015this is a test\u00153361_5000\u0015",
        "pause_markers": "well (.) let me think (0.3) about it",
        "terminator_codes": "she said +/. and then +//. nothing",
        "stress_markers": "the \u0001big\u0001 \u0002red\u0002 ball",
        "elongation": "a:nd the:n we we:nt ho:me",
        "incomplete_words": "an(d) the(n) we wen(t) home",
        "mixed_annotations": "\u00150_1000\u0015well (.) a:nd the(n) +/. \u0001big\u0001",
        "whitespace_issues": "  too   many    spaces   here  ",
        "clean_text": "this is already clean text",
    }


@pytest.fixture
def sample_cleaned_texts():
    """Expected cleaned output for corresponding uncleaned texts."""
    return {
        "inline_timestamps": "this is a test",
        "pause_markers": "well let me think about it",
        "terminator_codes": "she said and then nothing",
        "stress_markers": "the big red ball",
        "elongation": "and then we went home",
        "incomplete_words": "and then we went home",
        "mixed_annotations": "well and then big",
        "whitespace_issues": "too many spaces here",
        "clean_text": "this is already clean text",
    }


# ---------------------------------------------------------------------------
# Mock pylangacq Reader
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_reader_basic(sample_utterances_with_timestamps, sample_participants_with_roles):
    """A mock pylangacq Reader with basic valid data."""
    reader = MagicMock()
    reader.file_paths = ["/fake/path/corpus/test.cha"]
    reader.languages.return_value = [["eng"]]
    reader.utterances.return_value = sample_utterances_with_timestamps

    # Mock participants as named tuples
    mock_participants = []
    for code, role in sample_participants_with_roles.items():
        p = MagicMock()
        p.code = code
        p.name = code
        p.role = role
        mock_participants.append(p)
    reader.participants.return_value = [set(mock_participants)]

    # Mock headers
    mock_header = MagicMock()
    mock_header.participants = {
        code: {"role": role}
        for code, role in sample_participants_with_roles.items()
    }
    reader.headers.return_value = [mock_header]

    return reader


@pytest.fixture
def mock_reader_no_timestamps(sample_utterances_no_timestamps):
    """A mock pylangacq Reader with utterances that have no timestamps."""
    reader = MagicMock()
    reader.file_paths = ["/fake/path/corpus/notimed.cha"]
    reader.languages.return_value = [["eng"]]
    reader.utterances.return_value = sample_utterances_no_timestamps

    mock_participants = []
    for code in ["TEA", "STU"]:
        p = MagicMock()
        p.code = code
        p.name = code
        p.role = "Teacher" if code == "TEA" else "Student"
        mock_participants.append(p)
    reader.participants.return_value = [set(mock_participants)]

    mock_header = MagicMock()
    mock_header.participants = {
        "TEA": {"role": "Teacher"},
        "STU": {"role": "Student"},
    }
    reader.headers.return_value = [mock_header]

    return reader


@pytest.fixture
def mock_reader_empty():
    """A mock pylangacq Reader with no utterances."""
    reader = MagicMock()
    reader.file_paths = ["/fake/path/corpus/empty.cha"]
    reader.languages.return_value = [["eng"]]
    reader.utterances.return_value = []
    reader.participants.return_value = [set()]
    reader.headers.return_value = [MagicMock(participants={})]
    return reader


# ---------------------------------------------------------------------------
# Temporary directory fixtures for file I/O tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project_dir(tmp_path):
    """Create a temporary project directory with transcripts structure."""
    transcripts_dir = tmp_path / "dataset" / "transcripts"
    transcripts_dir.mkdir(parents=True)
    parsed_dir = tmp_path / "dataset" / "parsed"
    parsed_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def tmp_transcripts_dir(tmp_project_dir):
    """Return the transcripts subdirectory within the temp project."""
    return tmp_project_dir / "dataset" / "transcripts"


@pytest.fixture
def tmp_parsed_dir(tmp_project_dir):
    """Return the parsed output subdirectory within the temp project."""
    return tmp_project_dir / "dataset" / "parsed"


@pytest.fixture
def tmp_corpus_with_files(tmp_transcripts_dir):
    """Create a temporary corpus directory with sample .cha files."""
    corpus_dir = tmp_transcripts_dir / "TestCorpus"
    corpus_dir.mkdir()

    # Minimal valid CHAT file content
    cha_content = (
        "@UTF8\n"
        "@Begin\n"
        "@Languages:\teng\n"
        "@Participants:\tTEA Teacher Teacher, STU Student Student\n"
        "@ID:\teng|TestCorpus|TEA||female|||Teacher|||\n"
        "@ID:\teng|TestCorpus|STU||male|||Student|||\n"
        "*TEA:\tthis is a test .\n"
        "%mor:\tpro|this cop|be&3S det|a n|test .\n"
        "*STU:\tyes it is .\n"
        "%mor:\tadv|yes pro|it cop|be&3S .\n"
        "@End\n"
    )

    file1 = corpus_dir / "session1.cha"
    file1.write_text(cha_content, encoding="utf-8")

    file2 = corpus_dir / "session2.cha"
    file2.write_text(cha_content, encoding="utf-8")

    return corpus_dir


@pytest.fixture
def tmp_multiple_corpora(tmp_transcripts_dir):
    """Create multiple corpus directories with sample .cha files."""
    cha_content = (
        "@UTF8\n"
        "@Begin\n"
        "@Languages:\teng\n"
        "@Participants:\tTEA Teacher Teacher\n"
        "*TEA:\thello world .\n"
        "@End\n"
    )

    corpora = ["CorpusA", "CorpusB", "CorpusC"]
    for corpus_name in corpora:
        corpus_dir = tmp_transcripts_dir / corpus_name
        corpus_dir.mkdir()
        (corpus_dir / "file1.cha").write_text(cha_content, encoding="utf-8")
        (corpus_dir / "file2.cha").write_text(cha_content, encoding="utf-8")

    return tmp_transcripts_dir
