from __future__ import annotations

import asyncio

from covxplore.api_client import ExecuteResult
from covxplore.generation.batch import TestcaseCandidate, async_execute_candidates, merge_batch_results
from covxplore.types import ConditionKey, ConditionTraceEntry, TestResult, TestSuite


def _result(name: str, status: str, keys: list[tuple[int, bool]], body: str | None = None) -> TestResult:
    return TestResult(
        test_name=name,
        test_body=body or name,
        status=status,
        condition_trace=[
            ConditionTraceEntry(
                node_id=node_id,
                condition=f"c{node_id}",
                true_branch_visited=polarity,
                false_branch_visited=not polarity,
            )
            for node_id, polarity in keys
        ],
    )


def test_merge_recomputes_gain_and_rejects_redundant_overlap():
    suite = TestSuite(function_path="/f.cpp::f()")
    first = _result("first", "PASSED", [(1, True)], "x();")
    second = _result("second", "PASSED", [(1, True)], "y();")

    summary = merge_batch_results(suite, [first, second])

    assert summary.accepted == [first]
    assert summary.rejected_redundant == [second]
    assert suite.tests == [first]
    assert suite.coverage._covered_keys == {ConditionKey(1, True)}


def test_summary_records_rejected_redundant_streak_per_batch():
    suite = TestSuite(function_path="/f.cpp::f()")
    suite.coverage.consecutive_redundant = 2
    summary = merge_batch_results(
        suite,
        [
            _result("dupe1", "PASSED", []),
            _result("dupe2", "PASSED", []),
            _result("dupe3", "PASSED", []),
        ],
        min_suite_size=0,
    )

    summary.record_redundancy(suite)

    assert suite.coverage.consecutive_redundant == 3
    assert suite.tests == []


def test_summary_keeps_progress_reset_when_batch_has_progress():
    suite = TestSuite(function_path="/f.cpp::f()")
    suite.coverage.consecutive_redundant = 2
    progress = _result("progress", "PASSED", [(1, True)], "x();")
    dupe = _result("dupe", "PASSED", [(1, True)], "y();")
    summary = merge_batch_results(suite, [progress, dupe], min_suite_size=0)

    summary.record_redundancy(suite)

    assert summary.accepted == [progress]
    assert summary.rejected_redundant == [dupe]
    assert suite.coverage.consecutive_redundant == 0


def test_summary_does_not_count_failed_batch_as_redundant_streak():
    suite = TestSuite(function_path="/f.cpp::f()")
    suite.coverage.consecutive_redundant = 2
    failed = _result("failed", "FAILED", [], "bad();")
    dupe = _result("dupe", "PASSED", [], "dupe();")
    summary = merge_batch_results(suite, [failed, dupe], min_suite_size=0)

    summary.record_redundancy(suite)

    assert summary.accepted == [failed]
    assert summary.rejected_redundant == [dupe]
    assert suite.coverage.consecutive_redundant == 2


def test_merge_accepts_non_passing_results_for_visibility():
    suite = TestSuite(function_path="/f.cpp::f()")
    failure = _result("compile", "COMPILE_ERROR", [], "bad();")

    summary = merge_batch_results(suite, [failure])

    assert summary.accepted == [failure]
    assert summary.rejected_redundant == []
    assert suite.tests == [failure]


def test_merge_orders_by_marginal_gain_before_body_length():
    suite = TestSuite(function_path="/f.cpp::f()")
    small_gain_short = _result("one", "PASSED", [(1, True)], "x();")
    big_gain_long = _result("two", "PASSED", [(2, True), (3, False)], "longer_call();")

    summary = merge_batch_results(suite, [small_gain_short, big_gain_long])

    assert summary.accepted == [big_gain_long, small_gain_short]
    assert [test.test_name for test in suite.tests] == ["two", "one"]


def test_async_execute_candidates_uses_sync_executor_in_threads():
    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
            self.calls.append((absolute_path, test_body, test_name))
            return ExecuteResult(raw={"testName": test_name, "status": "PASSED"})

    executor = Executor()
    results = asyncio.run(
        async_execute_candidates(
            "/f.cpp::f()",
            [TestcaseCandidate("a();", "A"), TestcaseCandidate("b();", "B")],
            executor,
        )
    )

    assert [result.test_name for result in results] == ["A", "B"]
    assert executor.calls == [("/f.cpp::f()", "a();", "A"), ("/f.cpp::f()", "b();", "B")]
