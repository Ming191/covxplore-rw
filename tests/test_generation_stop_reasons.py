from types import SimpleNamespace

from covxplore.generation.stop_reasons import StopPolicy
from covxplore.types import CoverageDetail, TestResult, TestSuite


def _policy(**kwargs):
    return StopPolicy(max_batches=kwargs.get("max_batches", 10), redundant_streak_limit=3, fail_streak_limit=3)


def test_coverage_target_zero_iteration_guard():
    assert _policy().coverage_target_reached(TestSuite(function_path="f")) is False


def test_stop_when_statement_and_branch_complete():
    suite = TestSuite(function_path="f")
    suite.batch_count = 1
    result = TestResult(test_name="t", test_body="", status="PASSED", statement_coverage=CoverageDetail(visited=1, total=1), branch_coverage=CoverageDetail(visited=2, total=2))
    suite.add_result(result)
    assert _policy().terminal_reason(suite) == "coverage_target"


def test_hard_stop_requires_structural_totals():
    suite = TestSuite(function_path="f")
    suite.batch_count = 1
    suite.tests = [TestResult(test_name="t", test_body="", status="PASSED")]
    assert _policy().hard_stop_reason(suite) is None
    assert _policy().terminal_reason(suite) == "agent_done"


def test_seeded_zero_coverage_is_not_complete():
    suite = TestSuite(function_path="f")
    suite.coverage.seed_totals(7, 4)
    suite.batch_count = 1
    suite.tests = [TestResult(test_name="t", test_body="", status="FAILED")]
    assert _policy().terminal_reason(suite) == "agent_done"


def test_from_config_uses_structural_defaults():
    policy = StopPolicy.from_config(SimpleNamespace(max_batches=5))
    assert policy.max_batches == 5
    assert policy.redundant_streak_limit == 3
    assert policy.fail_streak_limit == 3
