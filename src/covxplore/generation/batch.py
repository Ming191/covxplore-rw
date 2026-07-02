from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from covxplore.driver.executor import TestCaseExecutor
from covxplore.status import TestStatus, is_failure_status, normalize_test_status
from covxplore.types.test_result import TestResult
from covxplore.types.test_suite import TestSuite


@dataclass
class TestcaseCandidate:
    test_body: str
    test_name: str | None = None
    target_node_id: int | None = None
    target_polarity: str | None = None
    target_reason: str | None = None


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


def _marginal_gain(suite: TestSuite, result: TestResult) -> int:
    return len(result.condition_keys() - suite.coverage._covered_keys)


def _is_coverage_result(result: TestResult) -> bool:
    return normalize_test_status(result.status) in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}


def merge_batch_results(
    suite: TestSuite,
    results: list[TestResult],
    min_suite_size: int = 3,
) -> BatchMergeSummary:
    """Greedily add visible/progressing results from one generated batch."""
    indexed = list(enumerate(results))
    ranked = sorted(
        indexed,
        key=lambda item: (
            0 if _is_coverage_result(item[1]) else 1,
            -_marginal_gain(suite, item[1]),
            len(item[1].test_body),
            item[0],
        ),
    )

    summary = BatchMergeSummary()
    for _, result in ranked:
        gain = _marginal_gain(suite, result)
        if _is_coverage_result(result) and gain == 0:
            summary.rejected_redundant.append(result)
            continue

        suite.add_result(result, min_suite_size=min_suite_size)
        summary.accepted.append(result)

    return summary


async def async_execute_candidates(
    function_path: str,
    candidates: list[TestcaseCandidate],
    executor: TestCaseExecutor,
):
    return await asyncio.gather(
        *(
            asyncio.to_thread(
                executor.execute,
                function_path,
                candidate.test_body,
                candidate.test_name,
            )
            for candidate in candidates
        )
    )
