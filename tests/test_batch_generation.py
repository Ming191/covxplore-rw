from covxplore.generation.batch import merge_batch_results
from covxplore.tools.execute_testcase import _format_batch_summary
from covxplore.types import CoverageDetail, TestResult, TestSuite, UnvisitedBranch


def _result(name: str, *, stmt: int = 0, branch: int = 0, body: str | None = None) -> TestResult:
    return TestResult(
        test_name=name,
        test_body=body or name,
        status="PASSED",
        statement_coverage=CoverageDetail(visited=stmt, total=3),
        branch_coverage=CoverageDetail(visited=branch, total=2),
    )


def test_merge_accepts_first_passing_zero_gain_seed_result():
    suite = TestSuite(function_path="/f.cpp::f()")
    result = _result("seed")
    summary = merge_batch_results(suite, [result])
    assert summary.accepted == [result]
    assert suite.tests == [result]


def test_merge_rejects_later_zero_gain_result_and_counts_redundancy():
    suite = TestSuite(function_path="/f.cpp::f()")
    merge_batch_results(suite, [_result("seed")])
    duplicate = _result("duplicate")
    summary = merge_batch_results(suite, [duplicate])
    assert summary.rejected_redundant == [duplicate]
    assert suite.rejected_tests == [duplicate]
    assert suite.redundancy_rate == 0.5


def test_merge_orders_by_structural_gain():
    suite = TestSuite(function_path="/f.cpp::f()")
    small = _result("small", stmt=1, body="x();")
    large = _result("large", stmt=2, branch=1, body="long();")
    summary = merge_batch_results(suite, [small, large])
    assert summary.accepted == [large]
    assert summary.rejected_redundant == [small]


def test_execution_feedback_does_not_repeat_coverage_guidance():
    suite = TestSuite(function_path="/f.cpp::f()")
    result = _result("seed", stmt=1, branch=1)
    result.unvisited_branches = [
        UnvisitedBranch(
            node_id=2,
            condition="x > 0",
            true_visited=True,
            false_visited=False,
        )
    ]
    summary = merge_batch_results(suite, [result])

    feedback = _format_batch_summary(suite, summary)

    assert "Accepted: seed(PASSED" in feedback
    assert "Branch coverage:" not in feedback
    assert "missing:" not in feedback
    assert "Suite best:" not in feedback
