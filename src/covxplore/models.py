"""Compatibility exports for coverage data models and suite state."""

from __future__ import annotations

from covxplore.coverage_models import (
    ConditionKey,
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TraceSummary,
    UnvisitedBranch,
    UnvisitedMcdc,
    UnvisitedStatement,
)
from covxplore.test_suite import TestSuite

__all__ = [
    "ConditionKey",
    "ConditionTraceEntry",
    "CoverageDetail",
    "TestResult",
    "TestSuite",
    "TraceSummary",
    "UnvisitedBranch",
    "UnvisitedMcdc",
    "UnvisitedStatement",
]
