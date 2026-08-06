import pytest

from covxplore.api_client import AkaUTError, ExecuteResult
from covxplore.status import TestStatus
from covxplore.tools.execute_testcase import ExecuteTestcaseBatchTool, HardStop, RunContext
from covxplore.types import TestSuite


class _FakeExecutor:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def execute(self, absolute_path, test_body, test_name=None):
        self.calls.append((absolute_path, test_body, test_name))
        if self.error is not None:
            raise self.error
        return self.response or ExecuteResult(raw={"testName": test_name, "status": "PASSED"})


def test_run_context_isolates_suites_by_run_id():
    context = RunContext()
    first = context.reset_suite("/a.cpp::f()", "one")
    second = context.reset_suite("/b.cpp::g()", "two")
    assert isinstance(first, TestSuite)
    assert context.get_suite("one") is first
    assert context.active_suite() is second
    context.cleanup_suite("one")
    assert context.get_suite("one") is None
    assert context.active_suite() is second


def test_batch_requires_active_suite():
    output = ExecuteTestcaseBatchTool(run_context=RunContext())._run(
        [{"test_body": "f();", "test_name": "t1"}]
    )
    assert "could not resolve target function path" in output


def test_batch_executes_and_keeps_contract_errors():
    context = RunContext()
    suite = context.reset_suite("/x.cpp::f()", "run")
    executor = _FakeExecutor()
    output = ExecuteTestcaseBatchTool(run_context=context, executor=executor)._run(
        [
            {"test_body": "f(1);", "test_name": "good"},
            {"test_body": "```cpp\nf();\n```", "test_name": "bad"},
        ]
    )
    assert executor.calls == [("/x.cpp::f()", "f(1);", "good")]
    assert "Accepted:" in output
    assert [test.status for test in suite.tests] == [
        TestStatus.PASSED.value,
        TestStatus.COMPILE_ERROR.value,
    ]


def test_batch_reports_failed_logs():
    context = RunContext()
    context.reset_suite("/x.cpp::f()", "run")
    executor = _FakeExecutor(
        response=ExecuteResult(
            raw={"testName": "bad", "status": "FAILED", "executeLog": "stderr"}
        )
    )
    output = ExecuteTestcaseBatchTool(run_context=context, executor=executor)._run(
        [{"test_body": "bad();", "test_name": "bad"}]
    )
    assert "--- bad | FAILED ---" in output
    assert "stderr" in output


def test_batch_classifies_compile_error():
    context = RunContext()
    suite = context.reset_suite("/x.cpp::f()", "run")
    executor = _FakeExecutor(error=AkaUTError("ld: cannot find ./test.cpp.out"))
    output = ExecuteTestcaseBatchTool(run_context=context, executor=executor)._run(
        [{"test_body": "bad();", "test_name": "bad"}]
    )
    assert suite.tests[0].status == TestStatus.COMPILE_ERROR.value
    assert "--- bad | COMPILE_ERROR ---" in output


def test_batch_hard_stops_at_coverage_target():
    context = RunContext()
    suite = context.reset_suite("/x.cpp::f()", "run")
    suite.coverage.seed_totals(1, 1)
    executor = _FakeExecutor(
        response=ExecuteResult(
            raw={
                "testName": "done",
                "status": "PASSED",
                "statementCoverage": {"visited": 1, "total": 1, "progress": 1.0},
                "branchCoverage": {"visited": 1, "total": 1, "progress": 1.0},
            }
        )
    )
    with pytest.raises(HardStop) as exc:
        ExecuteTestcaseBatchTool(run_context=context, executor=executor)._run(
            [{"test_body": "f();", "test_name": "done"}]
        )
    assert exc.value.reason == "coverage_target"
