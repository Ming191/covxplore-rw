from __future__ import annotations

import time
from dataclasses import dataclass, field

from covxplore.coverage.state import CoverageState
from covxplore.types.test_result import TestResult


@dataclass
class TestSuite:
    function_path: str

    tests: list[TestResult] = field(default_factory=list)
    rejected_tests: list[TestResult] = field(default_factory=list)
    coverage: CoverageState = field(default_factory=CoverageState)
    iteration_count: int = 0
    fail_streak: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def add_result(self, result: TestResult, min_suite_size: int = 3) -> None:
        self.iteration_count += 1
        result.iteration = self.iteration_count

        self.coverage.add_result(result, prior_results=self.tests, min_suite_size=min_suite_size)

        self.tests.append(result)

    @property
    def redundancy_rate(self) -> float:
        if not self.tests:
            return 0.0
        redundant = sum(1 for t in self.tests if t.is_redundant)
        return redundant / len(self.tests)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.token_input for t in self.tests)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.token_output for t in self.tests)

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def non_redundant_tests(self) -> list[TestResult]:
        return [t for t in self.tests if not t.is_redundant]

    def mark_iter(self, failed: bool) -> None:
        self.fail_streak = self.fail_streak + 1 if failed else 0

    def consecutive_failures(self) -> int:
        """Count consecutive failing tool iterations, not individual test cases."""
        return self.fail_streak

    def to_dict(self) -> dict:
        metrics = self.coverage.metrics(self.tests)
        return {
            "function_path": self.function_path,
            "iteration_count": self.iteration_count,
            "fail_streak": self.fail_streak,
            "statement_coverage_pct": round(metrics.statement_pct, 4),
            "branch_coverage_pct": round(metrics.branch_pct, 4),
            "covered_statements": metrics.covered_statements,
            "covered_branches": metrics.covered_branches,
            "mcdc_coverage_pct": round(metrics.mcdc_pct, 4),
            "total_statements": metrics.total_statements,
            "total_branches": metrics.total_branches,
            "covered_mcdc_pairs": metrics.covered_mcdc_pairs,
            "total_mcdc_pairs": metrics.total_mcdc_pairs,
            "redundancy_rate": round(self.redundancy_rate, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "tests": [t.model_dump() for t in self.tests],
            "rejected_tests": [t.model_dump() for t in self.rejected_tests],
        }
