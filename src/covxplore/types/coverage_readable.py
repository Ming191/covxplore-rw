from __future__ import annotations

from typing import Protocol

from covxplore.types.test_result import TestResult
from covxplore.types.coverage_metrics import CoverageMetrics
from covxplore.types.coverage_gap_input import CoverageGapInput


class CoverageReadable(Protocol):
    """Structural interface for any object that can answer coverage queries."""

    def metrics(self, tests: list[TestResult]) -> CoverageMetrics: ...

    def gap_input(self, tests: list[TestResult], batch_count: int) -> CoverageGapInput: ...
