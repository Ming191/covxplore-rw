from types import SimpleNamespace

from covxplore.generation.stop_reasons import (
    coverage_target_reached,
    deduce_agent_done_stop_reason,
)
from covxplore.types import TestResult, TestSuite


def _config(max_iterations=10, redundant_streak_limit=3, mcdc_target=1.0):
    return SimpleNamespace(
        max_iterations=max_iterations,
        redundant_streak_limit=redundant_streak_limit,
        mcdc_target=mcdc_target,
    )


def test_coverage_target_zero_iteration_guard():
    suite = TestSuite(function_path="f")
    assert coverage_target_reached(suite, _config()) is False


def test_stop_reason_precedence_max_iter_over_redundant_and_coverage():
    suite = TestSuite(function_path="f")
    suite.iteration_count = 10
    suite.coverage.consecutive_redundant = 10

    assert deduce_agent_done_stop_reason(suite, _config(max_iterations=10)) == "max_iter"


def test_stop_reason_precedence_redundant_over_coverage():
    suite = TestSuite(function_path="f")
    suite.iteration_count = 1
    suite.coverage.consecutive_redundant = 3


    assert (
        deduce_agent_done_stop_reason(suite, _config(redundant_streak_limit=3))
        == "redundant_streak"
    )


def test_stop_reason_coverage_target_when_reached():
    suite = TestSuite(function_path="f")
    suite.iteration_count = 1
    suite.tests = [TestResult(test_name="t", test_body="", status="PASSED")]

    assert deduce_agent_done_stop_reason(suite, _config()) == "coverage_target"
