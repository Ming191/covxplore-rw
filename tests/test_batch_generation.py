from covxplore.generation.batch import merge_batch_results
from covxplore.types import CoverageDetail, TestResult, TestSuite


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
