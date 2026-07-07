from covxplore.crew import _guard
from covxplore.types import TestResult, TestSuite
from covxplore.tools.execute_testcase import RunContext


def _ctx(suite: TestSuite) -> RunContext:
    ctx = RunContext()
    ctx._suites["run"] = suite
    ctx._active_run_id = "run"
    return ctx


def test_guard_retries_first_final_answer_once():
    suite = TestSuite(function_path="f")

    ok, message = _guard(_ctx(suite), max_iterations=10)("done")

    assert ok is False
    assert "call execute_testcase_batch" in message


def test_guard_retries_three_final_answers_then_accepts():
    suite = TestSuite(function_path="f")
    guard = _guard(_ctx(suite), max_iterations=10)

    assert guard("done")[0] is False
    assert guard("done")[0] is False
    assert guard("done")[0] is False
    assert guard("done") == (True, "done")


def test_guard_accepts_final_answer_at_max_iterations():
    suite = TestSuite(function_path="f")
    suite.iteration_count = 10
    suite.tests.append(TestResult(test_name="t", test_body="", status="PASSED"))

    assert _guard(_ctx(suite), max_iterations=10)("done") == (True, "done")
