from types import SimpleNamespace

from covxplore.generation.stop_reasons import StopPolicy
from covxplore.types import ConditionKey, CoverageDetail, TestResult, TestSuite


def _policy(max_batches=10, redundant_streak_limit=3, fail_streak_limit=3, mcdc_target=1.0):
    return StopPolicy(
        max_batches=max_batches,
        redundant_streak_limit=redundant_streak_limit,
        fail_streak_limit=fail_streak_limit,
        mcdc_target=mcdc_target,
    )


def test_coverage_target_zero_iteration_guard():
    suite = TestSuite(function_path="f")
    assert _policy().coverage_target_reached(suite) is False


def test_stop_reason_precedence_coverage_over_max_batches():
    suite = TestSuite(function_path="f")
    suite.batch_count = 10
    suite.tests = [TestResult(test_name="t", test_body="", status="PASSED")]

    assert _policy(max_batches=10).terminal_reason(suite) == "coverage_target"


def test_stop_reason_precedence_redundant_before_max_batches():
    suite = TestSuite(function_path="f")
    suite.batch_count = 10
    suite.coverage.consecutive_redundant = 3
    suite.coverage.total_mcdc_pairs = 2
    suite.coverage._covered_keys = {ConditionKey(1, True)}
    suite.coverage._condition_id_to_text = {1: "x"}

    assert _policy(max_batches=10, redundant_streak_limit=3).terminal_reason(suite) == "redundant_streak"


def test_stop_reason_coverage_target_when_reached():
    suite = TestSuite(function_path="f")
    suite.batch_count = 1
    suite.tests = [
        TestResult(
            test_name="t",
            test_body="",
            status="PASSED",
            statement_coverage=CoverageDetail(visited=1, total=1, progress=1.0),
            branch_coverage=CoverageDetail(visited=1, total=1, progress=1.0),
        )
    ]

    assert _policy().terminal_reason(suite) == "coverage_target"


def test_hard_stop_requires_structural_totals_for_coverage():
    suite = TestSuite(function_path="f")
    suite.batch_count = 1
    suite.tests = [TestResult(test_name="t", test_body="", status="PASSED")]

    assert _policy().terminal_reason(suite) == "coverage_target"
    assert _policy().hard_stop_reason(suite) is None


def test_from_config_uses_defaults_for_partial_config():
    policy = StopPolicy.from_config(SimpleNamespace(max_batches=5, mcdc_target=0.8))

    assert policy.max_batches == 5
    assert policy.mcdc_target == 0.8
    assert policy.redundant_streak_limit == 3
    assert policy.fail_streak_limit == 3
