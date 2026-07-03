"""Tests for covxplore.status — pure, no network, no secrets."""

import pytest

from covxplore.status import TestStatus, is_failure_status, is_hard_fail, normalize_test_status


class TestTestStatusEnum:
    def test_members(self):
        assert TestStatus.PASSED == TestStatus.PASSED
        assert TestStatus.PASSED.value == "PASSED"

    def test_str_is_full_qualified_name(self):
        # (str, Enum) members: str() returns the qualified member name,
        # NOT the bare value. Use .value for the bare string.
        assert str(TestStatus.PASSED) == "TestStatus.PASSED"

    def test_value_access(self):
        assert TestStatus.PASSED.value == "PASSED"
        assert TestStatus.FAILED.value == "FAILED"

    def test_equality_with_value(self):
        assert TestStatus.PASSED == "PASSED"
        assert TestStatus.FAILED == "FAILED"

    def test_count(self):
        assert len(TestStatus) == 5


class TestNormalizeTestStatus:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, TestStatus.UNKNOWN),
            ("", TestStatus.UNKNOWN),
            ("garbage", TestStatus.UNKNOWN),
            ("passed", TestStatus.PASSED),
            ("PASSED", TestStatus.PASSED),
            ("success", TestStatus.PASSED),
            ("Success", TestStatus.PASSED),
            ("pass", TestStatus.PASSED),
            ("ok", TestStatus.PASSED),
            ("OK", TestStatus.PASSED),
            ("failed", TestStatus.FAILED),
            ("FAILED", TestStatus.FAILED),
            ("fail", TestStatus.FAILED),
            ("Fail", TestStatus.FAILED),
            ("runtime error", TestStatus.RUNTIME_ERROR),
            ("RUNTIME_ERROR", TestStatus.RUNTIME_ERROR),
            ("runtime_err", TestStatus.RUNTIME_ERROR),
            ("runtime failure", TestStatus.RUNTIME_ERROR),
            ("runtime failed", TestStatus.RUNTIME_ERROR),
            ("compile error", TestStatus.COMPILE_ERROR),
            ("compilation error", TestStatus.COMPILE_ERROR),
            # hyphen / underscore / mixed separators → token normalisation
            ("runtime-error", TestStatus.RUNTIME_ERROR),
            ("compile_error", TestStatus.COMPILE_ERROR),
            ("COMPILATION-ERROR", TestStatus.COMPILE_ERROR),
            # "run-time-error" → "run time error" (no match → UNKNOWN)
            ("run-time-error", TestStatus.UNKNOWN),
            # "RUNTIME  ERROR" → "runtime  error" (double space, no match)
            ("RUNTIME  ERROR", TestStatus.UNKNOWN),
        ],
    )
    def test_mappings(self, raw, expected):
        assert normalize_test_status(raw) == expected

    def test_leading_trailing_whitespace(self):
        assert normalize_test_status("  passed  ") == TestStatus.PASSED
        assert normalize_test_status("\tfailed\t") == TestStatus.FAILED


class TestIsFailureStatus:
    @pytest.mark.parametrize(
        "status,expected",
        [
            ("PASSED", False),
            ("passed", False),
            (TestStatus.PASSED, False),
            ("FAILED", True),
            ("failed", True),
            (TestStatus.FAILED, True),
            (TestStatus.RUNTIME_ERROR, True),
            ("runtime error", True),
            (TestStatus.COMPILE_ERROR, True),
            ("compile error", True),
            ("UNKNOWN", False),
            (TestStatus.UNKNOWN, False),
        ],
    )
    def test_failure_detection(self, status, expected):
        assert is_failure_status(status) == expected


class TestIsHardFail:
    @pytest.mark.parametrize(
        "status,expected",
        [
            ("PASSED", False),
            (TestStatus.RUNTIME_ERROR, False),
            ("runtime error", False),
            ("FAILED", True),
            (TestStatus.FAILED, True),
            ("compile error", True),
            (TestStatus.COMPILE_ERROR, True),
            ("UNKNOWN", True),
            (TestStatus.UNKNOWN, True),
        ],
    )
    def test_hard_fail_detection(self, status, expected):
        assert is_hard_fail(status) == expected
