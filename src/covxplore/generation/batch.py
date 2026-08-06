from __future__ import annotations

from dataclasses import dataclass, field

from covxplore.status import TestStatus, is_failure_status, normalize_test_status
from covxplore.types.test_result import TestResult
from covxplore.types.test_suite import TestSuite


@dataclass
class BatchMergeSummary:
    accepted: list[TestResult] = field(default_factory=list)
    rejected_redundant: list[TestResult] = field(default_factory=list)

    def record_redundancy(self, suite: TestSuite) -> None:
        if not self.rejected_redundant:
            return
        if any(_is_coverage_result(result) and not result.is_redundant for result in self.accepted):
            return
        if any(is_failure_status(result.status) for result in self.accepted):
            return
        suite.coverage.consecutive_redundant += 1


def _is_coverage_result(result: TestResult) -> bool:
    return normalize_test_status(result.status) in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}


def merge_batch_results(
    suite: TestSuite,
    results: list[TestResult],
    min_suite_size: int = 1,
) -> BatchMergeSummary:
    """Greedily retain tests that add structural coverage."""
    ranked = sorted(
        enumerate(results),
        key=lambda item: (
            0 if _is_coverage_result(item[1]) else 1,
            -suite.coverage.structural_gain(item[1]),
            len(item[1].test_body),
            item[0],
        ),
    )

    summary = BatchMergeSummary()
    for _, result in ranked:
        gain = suite.coverage.structural_gain(result)
        if _is_coverage_result(result) and gain == 0 and suite.tests:
            result.is_redundant = True
            summary.rejected_redundant.append(result)
            suite.rejected_tests.append(result)
            continue

        suite.add_result(result, min_suite_size=min_suite_size)
        summary.accepted.append(result)

    return summary
