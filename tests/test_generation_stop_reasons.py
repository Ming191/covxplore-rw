from types import SimpleNamespace

from covxplore.generation.stop_reasons import coverage_target_reached, infer_stop
from covxplore.types import ConditionKey, CoverageDetail, TestResult, TestSuite


def _config(max_iterations=10, redundant_streak_limit=3, mcdc_target=1.0):
    return SimpleNamespace(
        max_iterations=max_iterations,
        redundant_streak_limit=redundant_streak_limit,
        mcdc_target=mcdc_target,
    )


def test_coverage_target_zero_iteration_guard():
    suite = TestSuite(function_path="f")
    assert coverage_target_reached(suite, _config()) is False


def test_stop_reason_precedence_coverage_over_max_iter():
    suite = TestSuite(function_path="f")
    suite.iteration_count = 10
    suite.tests = [TestResult(test_name="t", test_body="", status="PASSED")]

    assert infer_stop(suite, _config(max_iterations=10)) == "coverage_target"


def test_stop_reason_precedence_redundant_before_max_iter():
    suite = TestSuite(function_path="f")
    suite.iteration_count = 10
    suite.coverage.consecutive_redundant = 3
    suite.coverage.total_mcdc_pairs = 2
    suite.coverage._covered_keys = {ConditionKey(1, True)}
    suite.coverage._condition_id_to_text = {1: "x"}

    assert infer_stop(suite, _config(max_iterations=10, redundant_streak_limit=3)) == "redundant_streak"


def test_stop_reason_coverage_target_when_reached():
    suite = TestSuite(function_path="f")
    suite.iteration_count = 1
    suite.tests = [
        TestResult(
            test_name="t",
            test_body="",
            status="PASSED",
            statement_coverage=CoverageDetail(visited=1, total=1, progress=1.0),
            branch_coverage=CoverageDetail(visited=1, total=1, progress=1.0),
        )
    ]

    assert infer_stop(suite, _config()) == "coverage_target"
