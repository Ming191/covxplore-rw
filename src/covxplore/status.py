"""Canonical test execution statuses and normalization helpers."""
from __future__ import annotations

from enum import Enum


class TestStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILE_ERROR = "COMPILE_ERROR"
    INFRA_ERROR = "INFRA_ERROR"
    UNKNOWN = "UNKNOWN"


def normalize_test_status(raw_status: str | None) -> TestStatus:
    """Map backend/raw status strings to a canonical enum value."""
    if raw_status is None:
        return TestStatus.UNKNOWN

    token = raw_status.strip().lower().replace("-", " ").replace("_", " ")

    if token in {"success", "passed", "pass", "ok"}:
        return TestStatus.PASSED

    # Backend can return "failed" for non-success execution outcomes.
    if token in {"failed", "fail"}:
        return TestStatus.FAILED

    if token in {"runtime error", "runtime err", "runtime failure", "runtime failed"}:
        return TestStatus.RUNTIME_ERROR

    if token in {"compile error", "compilation error"}:
        return TestStatus.COMPILE_ERROR

    if token in {"infra error", "infrastructure error", "api error", "backend error"}:
        return TestStatus.INFRA_ERROR

    return TestStatus.UNKNOWN


def is_failure_status(status: str | TestStatus) -> bool:
    """Return True when status indicates a failed test execution."""
    normalized = status if isinstance(status, TestStatus) else normalize_test_status(status)
    return normalized in {
        TestStatus.FAILED,
        TestStatus.RUNTIME_ERROR,
        TestStatus.COMPILE_ERROR,
        TestStatus.INFRA_ERROR,
    }
