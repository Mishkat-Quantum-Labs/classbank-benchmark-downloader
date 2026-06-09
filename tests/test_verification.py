"""Tests for classify_verification function."""
import pytest

from preprocess_transcripts import classify_verification


def _make_checks(
    utterance_diff=0,
    missing_speakers=None,
    timestamp_coverage=100.0,
    word_diff_percent=0.0,
    validity_violations=None,
    monotonic=True,
):
    """Helper to build a checks dict with sensible defaults (all pass)."""
    return {
        "utterance_count": {"source": 10, "output": 10 - utterance_diff, "diff": utterance_diff},
        "speaker_mapping": {
            "expected": ["teacher", "student"],
            "found": ["teacher", "student"],
            "missing": missing_speakers or [],
        },
        "timestamp_coverage": {"percentage": timestamp_coverage},
        "word_count": {"source": 100, "output": 100, "diff_percent": word_diff_percent},
        "timestamp_validity": {"violations": validity_violations or []},
        "timestamp_order": {"monotonic": monotonic},
    }


class TestClassifyVerificationPass:
    """Tests for PASS classification (Req 9.1)."""

    def test_all_checks_perfect(self):
        checks = _make_checks()
        status, reasons = classify_verification(checks)
        assert status == "pass"
        assert reasons == []

    def test_pass_with_high_coverage(self):
        """Coverage above 90% contributes to pass."""
        checks = _make_checks(timestamp_coverage=95.0)
        status, reasons = classify_verification(checks)
        assert status == "pass"
        assert reasons == []

    def test_pass_with_10_percent_word_diff(self):
        """Exactly 10% word diff is still pass."""
        checks = _make_checks(word_diff_percent=10.0)
        status, reasons = classify_verification(checks)
        assert status == "pass"
        assert reasons == []


class TestClassifyVerificationFail:
    """Tests for FAIL classification (Req 9.3)."""

    def test_utterance_diff_3_or_more(self):
        checks = _make_checks(utterance_diff=3)
        status, reasons = classify_verification(checks)
        assert status == "fail"
        assert any("utterance_count" in r for r in reasons)

    def test_utterance_diff_5(self):
        checks = _make_checks(utterance_diff=5)
        status, reasons = classify_verification(checks)
        assert status == "fail"
        assert "utterance_count: diff=5 (threshold: 3)" in reasons

    def test_timestamp_coverage_below_50(self):
        checks = _make_checks(timestamp_coverage=45.0)
        status, reasons = classify_verification(checks)
        assert status == "fail"
        assert "timestamp_coverage: 45.0% (threshold: <50%)" in reasons

    def test_word_diff_exceeds_25(self):
        checks = _make_checks(word_diff_percent=30.0)
        status, reasons = classify_verification(checks)
        assert status == "fail"
        assert "word_count: diff=30.0% (threshold: >25%)" in reasons

    def test_timestamp_validity_violations(self):
        checks = _make_checks(validity_violations=[2, 5])
        status, reasons = classify_verification(checks)
        assert status == "fail"
        assert "timestamp_validity: violations at indices [2, 5]" in reasons

    def test_timestamp_not_monotonic(self):
        checks = _make_checks(monotonic=False)
        status, reasons = classify_verification(checks)
        assert status == "fail"
        assert "timestamp_order: not monotonic" in reasons

    def test_multiple_fail_conditions(self):
        """Multiple fail conditions should all appear in reasons."""
        checks = _make_checks(utterance_diff=5, timestamp_coverage=30.0, monotonic=False)
        status, reasons = classify_verification(checks)
        assert status == "fail"
        assert len(reasons) == 3


class TestClassifyVerificationWarn:
    """Tests for WARN classification (Req 9.2)."""

    def test_utterance_diff_1(self):
        checks = _make_checks(utterance_diff=1)
        status, reasons = classify_verification(checks)
        assert status == "warn"
        assert any("utterance_count" in r for r in reasons)

    def test_utterance_diff_2(self):
        checks = _make_checks(utterance_diff=2)
        status, reasons = classify_verification(checks)
        assert status == "warn"
        assert any("utterance_count" in r for r in reasons)

    def test_timestamp_coverage_50_to_90(self):
        checks = _make_checks(timestamp_coverage=75.0)
        status, reasons = classify_verification(checks)
        assert status == "warn"
        assert any("timestamp_coverage" in r for r in reasons)

    def test_timestamp_coverage_exactly_50(self):
        """50% is warn, not fail."""
        checks = _make_checks(timestamp_coverage=50.0)
        status, reasons = classify_verification(checks)
        assert status == "warn"

    def test_timestamp_coverage_exactly_90(self):
        """90% is warn boundary."""
        checks = _make_checks(timestamp_coverage=90.0)
        status, reasons = classify_verification(checks)
        assert status == "warn"

    def test_word_diff_above_10_at_25(self):
        """Word diff > 10% and <= 25% is warn."""
        checks = _make_checks(word_diff_percent=15.0)
        status, reasons = classify_verification(checks)
        assert status == "warn"
        assert any("word_count" in r for r in reasons)

    def test_word_diff_exactly_25(self):
        """Exactly 25% is still warn, not fail."""
        checks = _make_checks(word_diff_percent=25.0)
        status, reasons = classify_verification(checks)
        assert status == "warn"

    def test_missing_speakers_triggers_warn(self):
        checks = _make_checks(missing_speakers=["teacher"])
        status, reasons = classify_verification(checks)
        assert status == "warn"
        assert "speaker_mapping: missing roles ['teacher']" in reasons


class TestClassifyVerificationPriority:
    """Tests for priority: fail > warn > pass (Req 9.4)."""

    def test_fail_overrides_warn(self):
        """When both fail and warn conditions exist, status is fail."""
        checks = _make_checks(utterance_diff=5, timestamp_coverage=75.0)
        status, reasons = classify_verification(checks)
        assert status == "fail"
        # Only fail reasons returned, not warn reasons
        assert any("utterance_count" in r for r in reasons)

    def test_warn_overrides_pass(self):
        """When warn conditions exist but no fail, status is warn."""
        checks = _make_checks(utterance_diff=1)
        status, reasons = classify_verification(checks)
        assert status == "warn"


class TestClassifyVerificationReasons:
    """Tests for descriptive reason messages (Req 9.5)."""

    def test_pass_has_empty_reasons(self):
        checks = _make_checks()
        status, reasons = classify_verification(checks)
        assert reasons == []

    def test_warn_has_nonempty_reasons(self):
        checks = _make_checks(utterance_diff=1)
        status, reasons = classify_verification(checks)
        assert len(reasons) > 0

    def test_fail_has_nonempty_reasons(self):
        checks = _make_checks(utterance_diff=5)
        status, reasons = classify_verification(checks)
        assert len(reasons) > 0

    def test_reason_includes_check_name_and_value(self):
        checks = _make_checks(word_diff_percent=30.0)
        status, reasons = classify_verification(checks)
        assert any("word_count" in r and "30.0" in r for r in reasons)
