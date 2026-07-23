from covxplore.types import CoverageDetail, TestResult, TestSuite, UnvisitedBranch, UnvisitedStatement


def test_test_result_defaults_to_structural_fields():
    result = TestResult(test_name="t", test_body="f();", status="PASSED")
    assert result.new_structural_coverage == 0
    assert result.statement_coverage.total == 0
    assert result.branch_coverage.total == 0


def test_suite_assigns_accepted_order():
    suite = TestSuite(function_path="/f.cpp::f()")
    suite.add_result(TestResult(test_name="t", test_body="f();", status="PASSED"))
    assert suite.tests[0].accepted_order == 1


def test_unvisited_structural_dtos_allow_missing_ids():
    assert UnvisitedStatement(statement="x;").node_id is None
    assert UnvisitedBranch(condition="x", true_visited=False, false_visited=False).node_id is None


def test_coverage_detail_defaults():
    assert CoverageDetail().pct == 0.0
