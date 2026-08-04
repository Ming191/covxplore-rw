"""Tests for RunContext and ExecuteTestcaseTool custom context — pure, no network."""

import pytest

_EP = [{"node_id": 1, "polarity": "TRUE"}]

from covxplore.api_client import AkaUTError, ExecuteResult
from covxplore.types import TestResult, TestSuite
from covxplore.status import TestStatus
from covxplore.tools.execute_testcase import (
    ExecuteTestcaseBatchTool,
    HardStop,
    RunContext,
    ExecuteTestcaseTool,
)


def test_parse_trace_summary_accepts_target_function_condition_steps_contract():
    from covxplore.tools.execute_testcase import _parse_trace_summary

    summary = _parse_trace_summary(
        {
            "rawStepCount": 3,
            "targetFunctionStepCount": 2,
            "conditionStepCount": 1,
            "uniqueConditionOffsets": 1,
            "visitedFunctions": ["foo"],
            "targetFunctionConditionSteps": [
                {
                    "line": 4,
                    "start": 8,
                    "end": 15,
                    "full": "if (a == b)",
                    "sub": "a == b",
                    "nodeId": 9,
                    "branch": "FALSE",
                    "decisionRole": "condition",
                    "decisionId": "foo:8:15",
                    "conditionIndex": 0,
                    "conditionValue": False,
                    "runtimeValues": [
                        {"expression": "a", "value": "1", "type": "int"},
                        {"expression": "b", "value": "2", "type": "int"},
                    ],
                }
            ],
        }
    )

    assert summary is not None
    assert summary.raw_step_count == 3
    step = summary.target_function_condition_steps[0]
    assert step.node_id == 9
    assert step.runtime_values[1].expression == "b"
    assert summary.model_dump(by_alias=True)["targetFunctionConditionSteps"][0]["nodeId"] == 9


class TestRunContext:
    def test_reset_suite_creates_suite_by_run_id(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/f.cpp::foo()", "run-A")

        assert isinstance(suite, TestSuite)
        assert suite.function_path == "/f.cpp::foo()"
        assert ctx._suites["run-A"] is suite
        assert ctx.get_suite("run-A") is suite
        assert ctx.active_suite() is suite

    def test_cleanup_suite_removes_only_requested_run(self):
        ctx = RunContext()
        ctx.reset_suite("/a.cpp::f()", "run-1")
        ctx.reset_suite("/b.cpp::g()", "run-2")

        ctx.cleanup_suite("run-1")

        assert "run-1" not in ctx._suites
        assert "run-2" in ctx._suites
        assert ctx.active_suite() is ctx.get_suite("run-2")

    def test_cleanup_nonexistent_run_no_error(self):
        ctx = RunContext()
        ctx.cleanup_suite("nope")  # should not raise

    def test_get_suite_returns_suite_by_explicit_run_id(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/f.cpp::foo()", "run-X")

        assert ctx.get_suite("run-X") is suite
        assert ctx.get_suite("missing") is None

    def test_isolation_between_contexts(self):
        ctx1 = RunContext()
        ctx2 = RunContext()
        suite1 = ctx1.reset_suite("/f1.cpp::f()", "r1")
        suite2 = ctx2.reset_suite("/f2.cpp::g()", "r2")

        assert ctx1.get_suite("r1") is suite1
        assert ctx2.get_suite("r2") is suite2
        assert ctx1.get_suite("r2") is None
        assert ctx2.get_suite("r1") is None
        assert suite1.function_path == "/f1.cpp::f()"
        assert suite2.function_path == "/f2.cpp::g()"


class TestExecuteTestcaseToolCustomContext:
    def test_default_run_context_factory(self):
        tool = ExecuteTestcaseTool()
        assert isinstance(tool._run_context, RunContext)
        assert tool._run_context.get_suite("missing") is None

    def test_custom_run_context_passed(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/test.cpp::f()", "custom-r")
        tool = ExecuteTestcaseTool(run_context=ctx)
        assert tool._run_context is ctx
        assert tool._run_context.get_suite("custom-r") is suite
        assert suite.function_path == "/test.cpp::f()"

    def test_run_context_isolation(self):
        ctx1 = RunContext()
        suite1 = ctx1.reset_suite("/a.cpp::a()", "r1")
        ctx2 = RunContext()
        suite2 = ctx2.reset_suite("/b.cpp::b()", "r2")

        tool1 = ExecuteTestcaseTool(run_context=ctx1)
        tool2 = ExecuteTestcaseTool(run_context=ctx2)

        assert tool1._run_context.get_suite("r1") is suite1
        assert tool2._run_context.get_suite("r2") is suite2
        assert suite1.function_path == "/a.cpp::a()"
        assert suite2.function_path == "/b.cpp::b()"


class _FakeExecutor:
    response: ExecuteResult | None = None
    error: AkaUTError | None = None
    calls: list[tuple[str, str, str | None]] = []

    def execute(self, absolute_path, test_body, test_name=None):
        self.calls.append((absolute_path, test_body, test_name))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class TestExecuteTestcaseToolRobustResponses:
    def test_coverage_none_does_not_crash(self):
        raw = {
            "testName": "t1",
            "status": "PASSED",
            "statementCoverage": None,
            "branchCoverage": None,
            "mcdcCoverage": None,
        }
        _FakeExecutor.response = ExecuteResult(raw=raw)
        _FakeExecutor.error = None

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="f();", test_name="t1", expected_path=_EP
        )

        assert "Stmt: 0/0 (0%)" in output
        assert "Branch: 0/0 (0%)" in output

    def test_none_trace_and_unvisited_lists_do_not_crash(self):
        raw = {
            "testName": "t1",
            "status": "PASSED",
            "unvisitedMcdcConditions": None,
            "unvisitedStatements": None,
            "unvisitedBranches": None,
            "conditionTrace": None,
        }
        _FakeExecutor.response = ExecuteResult(raw=raw)
        _FakeExecutor.error = None

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="f();", test_name="t1", expected_path=_EP
        )

        assert "===  t1 | PASSED" in output

    def test_invalid_trace_summary_does_not_crash(self):
        raw = {
            "testName": "t1",
            "status": "PASSED",
            "traceSummary": {"visited_functions": [object()]},
        }
        _FakeExecutor.response = ExecuteResult(raw=raw)
        _FakeExecutor.error = None

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="f();", test_name="t1", expected_path=_EP
        )

        assert "===  t1 | PASSED" in output

    def test_unknown_akaut_error_is_execute_error_and_not_suite_result(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        _FakeExecutor.response = None
        _FakeExecutor.error = AkaUTError("POST http://localhost/api/testcase/execute failed: boom")

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="f();", test_name="t1", expected_path=_EP
        )

        assert "[EXECUTE_ERROR]" in output
        assert "COMPILE_ERROR" not in output
        assert suite.tests == []

    def test_tool_uses_active_suite(self):
        ctx = RunContext()
        suite_a = ctx.reset_suite("/a.cpp::f()", "run-A")
        suite_b = ctx.reset_suite("/b.cpp::g()", "run-B")
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="f();", test_name="t1", expected_path=_EP
        )

        assert "Suite best" in output
        assert suite_a.tests == []
        assert len(suite_b.tests) == 1

    def test_tool_injects_absolute_path_from_run_context(self):
        ctx = RunContext()
        ctx.reset_suite("/auto.cpp::f()", "run-A")
        executor = _FakeExecutor()
        executor.calls = []
        executor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        executor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=executor)._run(
            test_body="f();",
            test_name="t1",
            expected_path=_EP,
        )

        assert "===  t1 | PASSED" in output
        assert executor.calls == [("/auto.cpp::f()", "f();", "t1")]

    def test_tool_rejects_missing_path_for_unknown_run(self):
        executor = _FakeExecutor()
        executor.calls = []

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=executor)._run(
            test_body="f();",
            test_name="t1",
            expected_path=_EP,
        )

        assert "could not resolve target function path" in output
        assert executor.calls == []

    def test_cleaned_up_active_run_rejects_without_suite_mutation(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/known.cpp::f()", "known-run")
        ctx.cleanup_suite("known-run")
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="f();", test_name="t1", expected_path=_EP
        )

        assert "could not resolve target function path" in output
        assert suite.tests == []

    def test_contract_error_rejects_without_calling_executor(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        executor = _FakeExecutor()
        executor.calls = []

        output = ExecuteTestcaseTool(run_context=ctx, executor=executor)._run(
            test_body="```cpp\nf();\n```",
            test_name="bad",
            expected_path=_EP,
        )

        assert "[CONTRACT_ERROR]" in output
        assert "MARKDOWN_FENCE" in output
        assert executor.calls == []
        assert len(suite.tests) == 1
        assert suite.tests[0].status == TestStatus.COMPILE_ERROR.value

    def test_tool_accepts_phase4_target_metadata(self):
        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
            test_name="t1",
            target_node_id=2,
            target_polarity="TRUE",
            target_reason="cover node 2 true",
            expected_path=[{"node_id": 2, "polarity": "TRUE"}],
        )

        assert "===  t1 | PASSED" in output
        assert "Target → node=2 polarity=TRUE reason=cover node 2 true" in output

    def test_tool_rejects_invalid_phase4_action_without_calling_executor(self):
        executor = _FakeExecutor()
        executor.calls = []

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=executor)._run(
            test_body="int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
            test_name="t1",
            target_node_id=2,
            expected_path=[{"node_id": 2, "polarity": "TRUE"}],
        )

        assert "[ACTION_ERROR]" in output
        assert "target_node_id requires target_polarity" in output
        assert executor.calls == []

    def test_tool_surfaces_target_reason_warning(self):
        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            test_body="int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
            test_name="t1",
            target_node_id=2,
            target_polarity="FALSE",
            expected_path=[{"node_id": 2, "polarity": "FALSE"}],
        )

        assert "Action warnings:" in output
        assert "target_polarity provided without target_reason" in output

    def test_tool_hard_stops_when_coverage_target_reached(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        _FakeExecutor.response = ExecuteResult(
            raw={
                "testName": "t1",
                "status": "PASSED",
                "statementCoverage": {"visited": 2, "total": 2, "progress": 1.0},
                "branchCoverage": {"visited": 2, "total": 2, "progress": 1.0},
                "mcdcCoverage": {"visited": 0, "total": 0, "progress": 0.0},
            }
        )
        _FakeExecutor.error = None

        with pytest.raises(HardStop) as exc:
            ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
                test_body="f();", test_name="t1", expected_path=_EP
            )

        assert exc.value.reason == "coverage_target"
        assert len(suite.tests) == 1

    def test_tool_hard_stops_on_redundant_streak(self, monkeypatch):
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        suite.coverage.consecutive_redundant = 2

        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 0,
                "redundant_streak_limit": 3,
                "fail_streak_limit": 3,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        # Omit statement/branch coverage so _total_statements/_total_branches stay 0
        # and _is_coverage_done returns False (guard triggers).
        _FakeExecutor.response = ExecuteResult(
            raw={
                "testName": "t1",
                "status": "PASSED",
            }
        )
        _FakeExecutor.error = None

        with pytest.raises(HardStop) as exc:
            ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
                test_body="f();", test_name="t1", expected_path=_EP
            )

        assert exc.value.reason == "redundant_streak"
        assert suite.coverage.consecutive_redundant == 3


class TestExecuteTestcaseBatchTool:
    def test_batch_arun_executes_merges_and_keeps_contract_errors(self):
        class Executor:
            def __init__(self):
                self.calls = []

            def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
                self.calls.append((absolute_path, test_body, test_name))
                trace = [
                    {
                        "nodeId": 1,
                        "condition": "c1",
                        "trueBranchVisited": True,
                        "falseBranchVisited": False,
                    }
                ] if test_name in {"first", "dupe"} else [
                    {
                        "nodeId": 2,
                        "condition": "c2",
                        "trueBranchVisited": True,
                        "falseBranchVisited": False,
                    },
                    {
                        "nodeId": 3,
                        "condition": "c3",
                        "trueBranchVisited": False,
                        "falseBranchVisited": True,
                    },
                ]
                return ExecuteResult(
                    raw={
                        "testName": test_name,
                        "status": "PASSED",
                        "conditionTrace": trace,
                    }
                )

        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        executor = Executor()

        output = ExecuteTestcaseBatchTool(run_context=ctx, executor=executor)._run(
            [
                {"test_body": "f(1);", "test_name": "first", "expected_path": _EP},
                {"test_body": "f(1);", "test_name": "dupe", "expected_path": [{"node_id": 2, "polarity": "TRUE"}]},
                {"test_body": "```cpp\nf();\n```", "test_name": "bad", "expected_path": [{"node_id": 3, "polarity": "TRUE"}]},
            ]
        )

        assert len(executor.calls) == 2
        assert ("/x.cpp::f()", "f(1);", "first") in executor.calls
        assert ("/x.cpp::f()", "f(1);", "dupe") in executor.calls
        assert "Accepted:" in output
        assert "Rejected redundant: dupe" in output
        assert [test.test_name for test in suite.tests] == ["first", "bad"]
        assert suite.tests[-1].status == TestStatus.COMPILE_ERROR.value

    def test_batch_executes_candidates_sequentially(self):
        class Executor:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.calls = []

            def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(test_name)
                self.active -= 1
                return ExecuteResult(raw={"testName": test_name, "status": "PASSED"})

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        executor = Executor()

        ExecuteTestcaseBatchTool(run_context=ctx, executor=executor)._run(
            [
                {"test_body": "a();", "test_name": "a", "expected_path": _EP},
                {"test_body": "b();", "test_name": "b", "expected_path": [{"node_id": 2, "polarity": "TRUE"}]},
            ]
        )

        assert executor.calls == ["a", "b"]
        assert executor.max_active == 1

    def test_batch_reports_failed_logs_per_test_name(self):
        class Executor:
            def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
                return ExecuteResult(
                    raw={
                        "testName": test_name,
                        "status": "FAILED",
                        "executeLog": f"stderr for {test_name}",
                    }
                )

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")

        output = ExecuteTestcaseBatchTool(run_context=ctx, executor=Executor())._run(
            [
                {"test_body": "bad1();", "test_name": "bad1", "expected_path": _EP},
                {"test_body": "bad2();", "test_name": "bad2", "expected_path": [{"node_id": 2, "polarity": "TRUE"}]},
            ]
        )

        assert "Failed execution logs:" in output
        assert "--- bad1 | FAILED ---" in output
        assert "stderr for bad1" in output
        assert "--- bad2 | FAILED ---" in output
        assert "stderr for bad2" in output

    def test_batch_classifies_linker_missing_out_as_compile_error(self):
        class Executor:
            def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
                raise AkaUTError("ld.exe: cannot find ./test.cpp.out: No such file or directory")

        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")

        output = ExecuteTestcaseBatchTool(run_context=ctx, executor=Executor())._run(
            [{"test_body": "bad();", "test_name": "bad", "expected_path": _EP}]
        )

        assert suite.tests[0].status == TestStatus.COMPILE_ERROR.value
        assert "--- bad | COMPILE_ERROR ---" in output
        assert "cannot find ./test.cpp.out" in output

    def test_batch_reports_unknown_logs(self):
        class Executor:
            def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
                raise AkaUTError("backend exploded before returning status")

        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")

        output = ExecuteTestcaseBatchTool(run_context=ctx, executor=Executor())._run(
            [{"test_body": "maybe();", "test_name": "mystery", "expected_path": _EP}]
        )

        assert suite.tests[0].status == TestStatus.UNKNOWN.value
        assert "--- mystery | UNKNOWN ---" in output
        assert "backend exploded before returning status" in output

    def test_batch_fail_streak_counts_batches_not_candidates(self, monkeypatch):
        class Executor:
            def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
                raise AkaUTError("compile error")

        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 0,
                "redundant_streak_limit": 3,
                "fail_streak_limit": 3,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        tool = ExecuteTestcaseBatchTool(run_context=ctx, executor=Executor())
        batch = [
            {"test_body": "bad1();", "test_name": "bad1", "expected_path": _EP},
            {"test_body": "bad2();", "test_name": "bad2", "expected_path": [{"node_id": 2, "polarity": "TRUE"}]},
            {"test_body": "bad3();", "test_name": "bad3", "expected_path": [{"node_id": 3, "polarity": "TRUE"}]},
        ]

        assert "Accepted:" in tool._run(batch)
        assert suite.consecutive_failures() == 1
        assert len(suite.tests) == 3
        assert "Accepted:" in tool._run(batch)
        assert suite.consecutive_failures() == 2
        with pytest.raises(HardStop) as exc:
            tool._run(batch)

        assert exc.value.reason == "fail_streak"
        assert suite.consecutive_failures() == 3
        assert len(suite.tests) == 9

    def test_batch_arun_hard_stops_on_rejected_redundant(self, monkeypatch):
        class Executor:
            def execute(self, absolute_path: str, test_body: str, test_name: str | None = None):
                return ExecuteResult(raw={"testName": test_name, "status": "PASSED"})

        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 0,
                "redundant_streak_limit": 3,
                "fail_streak_limit": 5,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        suite.coverage.consecutive_redundant = 2

        with pytest.raises(HardStop) as exc:
            ExecuteTestcaseBatchTool(run_context=ctx, executor=Executor())._run(
                [{"test_body": "f();", "test_name": "dupe", "expected_path": _EP}]
            )

        assert exc.value.reason == "redundant_streak"
        assert [test.test_name for test in suite.tests] == ["dupe"]
        assert suite.coverage.consecutive_redundant == 3

    def test_batch_arun_requires_active_suite(self):
        output = ExecuteTestcaseBatchTool(run_context=RunContext())._run(
            [{"test_body": "f();", "test_name": "t1", "expected_path": _EP}]
        )

        assert "could not resolve target function path" in output



def test_fatal_tool_error_is_normal_exception():
    from covxplore.tools.execute_testcase import FatalToolError

    assert issubclass(FatalToolError, Exception)
    assert FatalToolError.__bases__ == (Exception,)
