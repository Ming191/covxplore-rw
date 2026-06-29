"""Tests for RunContext and ExecuteTestcaseTool custom context — pure, no network."""

import json
from types import SimpleNamespace

import pytest

from covxplore.api_client import AkaUTError, ExecuteResult
from covxplore.types import ConditionTraceEntry, TestResult, TestSuite, UnvisitedBranch
from covxplore.status import TestStatus
from covxplore.tools.execute_testcase import (
    HardStop,
    RunContext,
    ExecuteTestcaseTool,
    _format_condition_trace,
    _next_targets,
    _target_score,
)


class TestRunContext:
    def test_reset_suite_creates_suite_by_run_id(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/f.cpp::foo()", "run-A")

        assert isinstance(suite, TestSuite)
        assert suite.function_path == "/f.cpp::foo()"
        assert ctx._suites["run-A"] is suite
        assert ctx.get_suite("run-A") is suite

    def test_cleanup_suite_removes_only_requested_run(self):
        ctx = RunContext()
        ctx.reset_suite("/a.cpp::f()", "run-1")
        ctx.reset_suite("/b.cpp::g()", "run-2")

        ctx.cleanup_suite("run-1")

        assert "run-1" not in ctx._suites
        assert "run-2" in ctx._suites

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

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=_FakeExecutor())._run("run-1", "/x.cpp::f()", "f();", "t1")

        assert "Stmt: 0/0 (0%)" in output
        assert "Branch: 0/0 (0%)" in output
        assert "MC/DC: 0/0 (0%)" in output

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

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=_FakeExecutor())._run("run-1", "/x.cpp::f()", "f();", "t1")

        assert "===  t1 | PASSED" in output

    def test_invalid_trace_summary_does_not_crash(self):
        raw = {
            "testName": "t1",
            "status": "PASSED",
            "traceSummary": {"visited_functions": [object()]},
        }
        _FakeExecutor.response = ExecuteResult(raw=raw)
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=_FakeExecutor())._run("run-1", "/x.cpp::f()", "f();", "t1")

        assert "===  t1 | PASSED" in output

    def test_unknown_akaut_error_is_execute_error_and_not_suite_result(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        _FakeExecutor.response = None
        _FakeExecutor.error = AkaUTError("POST http://localhost/api/testcase/execute failed: boom")

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run("run-1", "/x.cpp::f()", "f();", "t1")

        assert "[EXECUTE_ERROR]" in output
        assert "COMPILE_ERROR" not in output
        assert suite.tests == []

    def test_tool_uses_explicit_run_id_for_suite_lookup(self):
        ctx = RunContext()
        suite_a = ctx.reset_suite("/a.cpp::f()", "run-A")
        suite_b = ctx.reset_suite("/b.cpp::g()", "run-B")
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run("run-A", "/a.cpp::f()", "f();", "t1")

        assert "Suite best" in output
        assert len(suite_a.tests) == 1
        assert suite_b.tests == []

    def test_unknown_run_id_executes_without_suite_mutation(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/known.cpp::f()", "known-run")
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run("unknown-run", "/x.cpp::f()", "f();", "t1")

        assert "===  t1 | PASSED" in output
        assert "Suite best" not in output
        assert suite.tests == []

    def test_contract_error_rejects_without_calling_executor(self):
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        executor = _FakeExecutor()
        executor.calls = []

        output = ExecuteTestcaseTool(run_context=ctx, executor=executor)._run(
            "run-1",
            "/x.cpp::f()",
            "```cpp\nf();\n```",
            "bad",
        )

        assert "[CONTRACT_ERROR]" in output
        assert "MARKDOWN_FENCE" in output
        assert executor.calls == []
        assert len(suite.tests) == 1
        assert suite.tests[0].status == TestStatus.COMPILE_ERROR.value

    def test_tool_accepts_phase4_target_metadata(self):
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=_FakeExecutor())._run(
            "run-1",
            "/x.cpp::f()",
            "int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
            "t1",
            target_node_id=2,
            target_polarity="TRUE",
            target_reason="cover node 2 true",
        )

        assert "===  t1 | PASSED" in output
        assert "Target → node=2 polarity=TRUE reason=cover node 2 true" in output

    def test_compact_feedback_returns_json_target_hit_and_next_targets(self, monkeypatch):
        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 1,
                "redundant_streak_limit": 99,
                "fail_streak_limit": 99,
                "mcdc_target": 1.0,
                "compact_feedback": True,
                "compact_feedback_top_targets": 2,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        _FakeExecutor.response = ExecuteResult(
            raw={
                "testName": "t1",
                "status": "PASSED",
                "mcdcCoverage": {"visited": 1, "total": 4, "progress": 0.25},
                "unvisitedMcdcConditions": [
                    {
                        "nodeId": 2,
                        "condition": "x > 0",
                        "trueBranchVisited": True,
                        "falseBranchVisited": False,
                    },
                    {
                        "nodeId": 3,
                        "condition": "y == 1",
                        "trueBranchVisited": False,
                        "falseBranchVisited": False,
                    },
                ],
                "conditionTrace": [
                    {
                        "nodeId": 2,
                        "condition": "x > 0",
                        "trueBranchVisited": True,
                        "falseBranchVisited": False,
                    }
                ],
            }
        )
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            "run-1",
            "/x.cpp::f()",
            "int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
            "t1",
            target_node_id=2,
            target_polarity="TRUE",
            target_reason="cover node 2 true",
        )

        payload = json.loads(output)
        assert payload["status"] == "PASSED"
        assert payload["target"]["hit"] is True
        assert payload["suite"]["mcdc"] == "1/4"
        assert payload["next_targets"] == [
            {
                "node_id": 2,
                "polarity": "FALSE",
                "kind": "mcdc",
                "condition": "x > 0",
                "attempts": 0,
                "misses": 0,
            },
            {
                "node_id": 3,
                "polarity": "TRUE",
                "kind": "mcdc",
                "condition": "y == 1",
                "attempts": 0,
                "misses": 0,
            },
        ]

    def test_compact_feedback_includes_miss_context_after_two_misses(self, monkeypatch):
        class QueueExecutor:
            def __init__(self):
                self.responses = [
                    ExecuteResult(
                        raw={
                            "testName": "t1",
                            "status": "PASSED",
                            "mcdcCoverage": {"visited": 0, "total": 2, "progress": 0.0},
                            "unvisitedMcdcConditions": [
                                {
                                    "nodeId": 1,
                                    "condition": "a",
                                    "trueBranchVisited": False,
                                    "falseBranchVisited": False,
                                }
                            ],
                            "conditionTrace": [
                                {
                                    "nodeId": 9,
                                    "condition": "blocker",
                                    "trueBranchVisited": False,
                                    "falseBranchVisited": True,
                                }
                            ],
                        }
                    ),
                    ExecuteResult(
                        raw={
                            "testName": "t2",
                            "status": "PASSED",
                            "mcdcCoverage": {"visited": 0, "total": 2, "progress": 0.0},
                            "unvisitedMcdcConditions": [
                                {
                                    "nodeId": 1,
                                    "condition": "a",
                                    "trueBranchVisited": False,
                                    "falseBranchVisited": False,
                                }
                            ],
                            "conditionTrace": [
                                {
                                    "nodeId": 9,
                                    "condition": "blocker",
                                    "trueBranchVisited": False,
                                    "falseBranchVisited": True,
                                }
                            ],
                        }
                    ),
                ]

            def execute(self, absolute_path, test_body, test_name=None):
                return self.responses.pop(0)

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 99,
                "redundant_streak_limit": 99,
                "fail_streak_limit": 99,
                "mcdc_target": 1.0,
                "compact_feedback": True,
                "compact_feedback_top_targets": 3,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        tool = ExecuteTestcaseTool(run_context=ctx, executor=QueueExecutor())
        tool._run(
            "run-1",
            "/x.cpp::f()",
            "f();",
            "t1",
            target_node_id=1,
            target_polarity="TRUE",
            target_reason="miss once",
        )

        output = tool._run(
            "run-1",
            "/x.cpp::f()",
            "f();",
            "t2",
            target_node_id=1,
            target_polarity="TRUE",
            target_reason="miss twice",
        )

        miss_context = json.loads(output)["target"]["miss_context"]
        assert miss_context["attempts"] == 2
        assert miss_context["misses"] == 2
        assert miss_context["trace_chain"] == [
            {"node_id": 9, "condition": "blocker", "true": False, "false": True}
        ]

    def test_compact_feedback_blocks_target_after_three_misses(self, monkeypatch):
        class QueueExecutor:
            def __init__(self):
                self.responses = [
                    ExecuteResult(
                        raw={
                            "testName": f"t{i}",
                            "status": "PASSED",
                            "mcdcCoverage": {"visited": 0, "total": 2, "progress": 0.0},
                            "unvisitedMcdcConditions": [
                                {
                                    "nodeId": 1,
                                    "condition": "a",
                                    "trueBranchVisited": False,
                                    "falseBranchVisited": False,
                                }
                            ],
                        }
                    )
                    for i in range(1, 4)
                ]

            def execute(self, absolute_path, test_body, test_name=None):
                return self.responses.pop(0)

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 99,
                "redundant_streak_limit": 99,
                "fail_streak_limit": 99,
                "mcdc_target": 1.0,
                "compact_feedback": True,
                "compact_feedback_top_targets": 2,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        tool = ExecuteTestcaseTool(run_context=ctx, executor=QueueExecutor())
        for i in range(1, 3):
            tool._run(
                "run-1",
                "/x.cpp::f()",
                "f();",
                f"t{i}",
                target_node_id=1,
                target_polarity="TRUE",
                target_reason="miss",
            )

        output = tool._run(
            "run-1",
            "/x.cpp::f()",
            "f();",
            "t3",
            target_node_id=1,
            target_polarity="TRUE",
            target_reason="miss",
        )

        payload = json.loads(output)
        assert payload["blocked_targets"] == [
            {
                "node_id": 1,
                "polarity": "TRUE",
                "reason": "blocked after 3 target misses across 3 attempts",
            }
        ]
        assert all(
            target["polarity"] != "TRUE" for target in payload["next_targets"]
        )

    def test_compact_feedback_rejects_stale_target_without_execution(self, monkeypatch):
        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 1,
                "redundant_streak_limit": 99,
                "fail_streak_limit": 99,
                "mcdc_target": 1.0,
                "compact_feedback": True,
                "compact_feedback_top_targets": 1,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        executor = _FakeExecutor()
        executor.calls = []
        executor.response = ExecuteResult(
            raw={
                "testName": "t1",
                "status": "PASSED",
                "mcdcCoverage": {"visited": 0, "total": 2, "progress": 0.0},
                "unvisitedMcdcConditions": [
                    {
                        "nodeId": 1,
                        "condition": "a",
                        "trueBranchVisited": False,
                        "falseBranchVisited": False,
                    }
                ],
            }
        )
        tool = ExecuteTestcaseTool(run_context=ctx, executor=executor)
        tool._run("run-1", "/x.cpp::f()", "f();", "t1")

        output = tool._run(
            "run-1",
            "/x.cpp::f()",
            "f();",
            "t2",
            target_node_id=99,
            target_polarity="TRUE",
            target_reason="stale target",
        )

        payload = json.loads(output)
        assert payload["status"] == "TARGET_NOT_ALLOWED"
        assert payload["next_targets"] == [
            {
                "node_id": 1,
                "polarity": "TRUE",
                "kind": "mcdc",
                "condition": "a",
                "attempts": 0,
                "misses": 0,
            }
        ]
        assert len(executor.calls) == 1

    def test_compact_feedback_uses_cumulative_mcdc_gaps(self, monkeypatch):
        class QueueExecutor:
            def __init__(self):
                self.responses = [
                    ExecuteResult(
                        raw={
                            "testName": "t1",
                            "status": "PASSED",
                            "mcdcCoverage": {"visited": 1, "total": 2, "progress": 0.5},
                            "unvisitedMcdcConditions": [
                                {
                                    "nodeId": 23,
                                    "condition": "testLeading",
                                    "trueBranchVisited": True,
                                    "falseBranchVisited": False,
                                }
                            ],
                            "conditionTrace": [
                                {
                                    "nodeId": 23,
                                    "condition": "testLeading",
                                    "trueBranchVisited": True,
                                    "falseBranchVisited": False,
                                }
                            ],
                        }
                    ),
                    ExecuteResult(
                        raw={
                            "testName": "t2",
                            "status": "PASSED",
                            "mcdcCoverage": {"visited": 1, "total": 2, "progress": 0.5},
                            "unvisitedMcdcConditions": [
                                {
                                    "nodeId": 23,
                                    "condition": "testLeading",
                                    "trueBranchVisited": False,
                                    "falseBranchVisited": False,
                                }
                            ],
                        }
                    ),
                ]

            def execute(self, absolute_path, test_body, test_name=None):
                return self.responses.pop(0)

        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 1,
                "redundant_streak_limit": 99,
                "fail_streak_limit": 99,
                "mcdc_target": 1.0,
                "compact_feedback": True,
                "compact_feedback_top_targets": 3,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        tool = ExecuteTestcaseTool(run_context=ctx, executor=QueueExecutor())

        tool._run("run-1", "/x.cpp::f()", "f();", "t1")
        output = tool._run("run-1", "/x.cpp::f()", "f();", "t2")

        payload = json.loads(output)
        assert {
            "node_id": 23,
            "polarity": "TRUE",
            "kind": "mcdc",
            "condition": "testLeading",
            "attempts": 0,
            "misses": 0,
        } not in payload["next_targets"]
        assert payload["next_targets"][0]["node_id"] == 23
        assert payload["next_targets"][0]["polarity"] == "FALSE"

    def test_next_targets_omit_blocked_targets(self):
        suite = TestSuite("/x.cpp::f()")
        suite.coverage.seed_conditions(
            [
                SimpleNamespace(node_id=1, condition="a", line_in_function=1),
                SimpleNamespace(node_id=2, condition="b", line_in_function=2),
            ],
            total_mcdc_pairs=4,
        )
        result = TestResult(
            test_name="t1",
            test_body="f();",
            status=TestStatus.PASSED.value,
        )

        targets = _next_targets(result, suite, 4, {(1, "TRUE"): "blocked"})

        assert {"node_id": 1, "polarity": "TRUE"} not in [
            {"node_id": target["node_id"], "polarity": target["polarity"]}
            for target in targets
        ]
        assert targets[0]["node_id"] == 1
        assert targets[0]["polarity"] == "FALSE"

    def test_next_targets_are_mcdc_first_when_mcdc_remains(self):
        suite = TestSuite("/x.cpp::f()")
        suite.coverage.seed_conditions(
            [SimpleNamespace(node_id=1, condition="a", line_in_function=1)],
            total_mcdc_pairs=2,
        )
        result = TestResult(
            test_name="t1",
            test_body="f();",
            status=TestStatus.PASSED.value,
            unvisited_branches=[
                UnvisitedBranch(
                    node_id=99,
                    condition="branch",
                    true_visited=False,
                    false_visited=False,
                )
            ],
        )

        targets = _next_targets(result, suite, 3)

        assert targets == [
            {
                "node_id": 1,
                "polarity": "TRUE",
                "kind": "mcdc",
                "condition": "a",
                "attempts": 0,
                "misses": 0,
            },
            {
                "node_id": 1,
                "polarity": "FALSE",
                "kind": "mcdc",
                "condition": "a",
                "attempts": 0,
                "misses": 0,
            },
        ]

    def test_target_score_penalizes_attempts_misses_and_redundancy(self):
        clean = {
            "node_id": 1,
            "polarity": "TRUE",
            "kind": "mcdc",
            "attempts": 0,
            "misses": 0,
        }
        missed = {**clean, "attempts": 2, "misses": 2}
        suite = SimpleNamespace(
            tests=[
                TestResult(
                    test_name="t1",
                    test_body="f();",
                    status=TestStatus.PASSED.value,
                    target_node_id=1,
                    target_polarity="TRUE",
                    is_redundant=True,
                )
            ]
        )

        assert _target_score(clean, None) == 100
        assert _target_score(missed, None) == -70
        assert _target_score(clean, suite) == 0

    def test_compact_feedback_truncates_failure_log(self, monkeypatch):
        ctx = RunContext()
        ctx.reset_suite("/x.cpp::f()", "run-1")
        settings = type(
            "Settings",
            (),
            {
                "min_suite_size": 1,
                "redundant_streak_limit": 99,
                "fail_streak_limit": 99,
                "mcdc_target": 1.0,
                "compact_feedback": True,
                "compact_feedback_top_targets": 3,
            },
        )()
        monkeypatch.setattr("covxplore.tools.execute_testcase.get_settings", lambda: settings)
        _FakeExecutor.response = ExecuteResult(
            raw={
                "testName": "t1",
                "status": "COMPILE_ERROR",
                "executeLog": "error: " + "x" * 1000,
            }
        )
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=ctx, executor=_FakeExecutor())._run(
            "run-1", "/x.cpp::f()", "f();", "t1"
        )

        payload = json.loads(output)
        assert payload["log"]["category"] == "compile_error"
        assert len(payload["log"]["excerpt"]) == 500

    def test_tool_rejects_invalid_phase4_action_without_calling_executor(self):
        executor = _FakeExecutor()
        executor.calls = []

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=executor)._run(
            "run-1",
            "/x.cpp::f()",
            "int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
            "t1",
            target_node_id=2,
        )

        assert "[ACTION_ERROR]" in output
        assert "target_node_id requires target_polarity" in output
        assert executor.calls == []

    def test_tool_surfaces_target_reason_warning(self):
        _FakeExecutor.response = ExecuteResult(raw={"testName": "t1", "status": "PASSED"})
        _FakeExecutor.error = None

        output = ExecuteTestcaseTool(run_context=RunContext(), executor=_FakeExecutor())._run(
            "run-1",
            "/x.cpp::f()",
            "int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
            "t1",
            target_node_id=2,
            target_polarity="FALSE",
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
                "run-1", "/x.cpp::f()", "f();", "t1"
            )

        assert exc.value.reason == "coverage_target"
        assert len(suite.tests) == 1

    def test_tool_hard_stops_on_redundant_streak(self, monkeypatch):
        ctx = RunContext()
        suite = ctx.reset_suite("/x.cpp::f()", "run-1")
        suite.coverage.total_mcdc_pairs = 2
        suite.coverage.consecutive_redundant = 2

        settings = type(
            "Settings",
            (),
            {"min_suite_size": 0, "redundant_streak_limit": 3, "mcdc_target": 1.0},
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
                "run-1", "/x.cpp::f()", "f();", "t1"
            )

        assert exc.value.reason == "redundant_streak"
        assert suite.coverage.consecutive_redundant == 3


def test_format_condition_trace_sorts_mixed_node_ids_without_crashing():
    result = TestResult(
        test_name="t1",
        test_body="f();",
        status=TestStatus.PASSED.value,
        condition_trace=[
            ConditionTraceEntry.model_construct(node_id="b", condition="b", true_branch_visited=True, false_branch_visited=False, line_in_function=None),
            ConditionTraceEntry.model_construct(node_id=1, condition="a", true_branch_visited=False, false_branch_visited=True, line_in_function=None),
            ConditionTraceEntry.model_construct(node_id=None, condition="c", true_branch_visited=True, false_branch_visited=True, line_in_function=None),
        ],
    )

    output = _format_condition_trace(result)

    assert output is not None
    assert "[node:1 line+?] 'a' TRUE=NO FALSE=YES" in output
    assert "[node:b line+?] 'b' TRUE=YES FALSE=NO" in output
    assert "[node:? line+?] 'c' TRUE=YES FALSE=YES" in output


def test_fatal_tool_error_is_normal_exception():
    from covxplore.tools.execute_testcase import FatalToolError

    assert issubclass(FatalToolError, Exception)
    assert FatalToolError.__bases__ == (Exception,)
