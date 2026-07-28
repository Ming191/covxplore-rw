from covxplore.crews.test_generation.crew import _guard
from covxplore.types import TestResult, TestSuite
from covxplore.tools.execute_testcase import RunContext


def _ctx(suite: TestSuite) -> RunContext:
    ctx = RunContext()
    ctx._suites["run"] = suite
    ctx._active_run_id = "run"
    return ctx


def test_guard_retries_final_answer_before_session_batch():
    suite = TestSuite(function_path="f")

    ctx = _ctx(suite)
    ok, message = _guard(ctx, start_batch=0)("done")

    assert ok is False
    assert "Discovery is now disabled" in message
    assert ctx.discovery_locked is True


def test_guard_accepts_final_answer_after_session_batch():
    suite = TestSuite(function_path="f")
    suite.record_batch(False)

    assert _guard(_ctx(suite), start_batch=0)("done") == (True, "done")


def test_guard_retries_three_final_answers_then_accepts():
    suite = TestSuite(function_path="f")
    guard = _guard(_ctx(suite), start_batch=0)

    assert guard("done")[0] is False
    assert guard("done")[0] is False
    assert guard("done")[0] is False
    assert guard("done") == (True, "done")
