from pathlib import Path

from covxplore.api_client import normalize_trace_summary
from covxplore_ui.app import create_app
from covxplore_ui.manager import ManagedRun, RunManager
from covxplore_ui.report import build_report_from_summary
from covxplore_ui.results import summary_to_run_state


def test_cors_allows_vite_preview_origin():
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://127.0.0.1:4173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:4173"


def test_normalize_trace_summary_camel_case():
    raw = {
        "rawStepCount": 3,
        "targetFunctionStepCount": 2,
        "conditionStepCount": 1,
        "uniqueConditionOffsets": 1,
        "visitedFunctions": ["fn"],
        "targetFunctionConditionSteps": [{"lineInFunction": 4}],
    }

    assert normalize_trace_summary(raw) == {
        "raw_step_count": 3,
        "target_function_step_count": 2,
        "condition_step_count": 1,
        "unique_condition_offsets": 1,
        "visited_functions": ["fn"],
    }


def test_summary_to_run_state_uses_camel_case():
    state = summary_to_run_state(
        {
            "run_id": "run_1",
            "function_path": "file.cpp\\f()",
            "prompt_variant": "full",
            "stop_reason": "coverage_target",
            "error": None,
            "metrics": {
                "statement_coverage_pct": 1.0,
                "branch_coverage_pct": 0.5,
                "mcdc_coverage_pct": 0.75,
                "covered_mcdc_pairs": 3,
                "total_mcdc_pairs": 4,
            },
            "test_suite": [
                {
                    "iteration": 1,
                    "test_name": "tc_1",
                    "status": "PASSED",
                    "new_mcdc_pairs_covered": 2,
                    "is_redundant": False,
                }
            ],
        },
        Path("results/gen_run_1.json"),
    )

    assert state["runId"] == "run_1"
    assert state["metrics"]["mcdcCoveragePct"] == 0.75
    assert state["tests"][0]["testName"] == "tc_1"
    assert state["tests"][0]["isPassed"] is True


def test_run_manager_cancel_marks_cancelling_until_worker_exits():
    manager = RunManager(default_out_dir=Path("missing-results"))
    run = ManagedRun(
        run_id="run_1",
        function_path="file.cpp\\f()",
        variant="full",
        out_dir=Path("results"),
        status="running",
    )
    manager._runs[run.run_id] = run

    state = manager.cancel("run_1")

    assert state is not None
    assert state["status"] == "cancelling"
    assert run.cancel_event.is_set()
    assert run.events[-1].type == "log"


def test_run_manager_cancelled_worker_emits_final_cancelled_event():
    manager = RunManager(default_out_dir=Path("missing-results"))
    run = ManagedRun(
        run_id="run_1",
        function_path="file.cpp\f()",
        variant="full",
        out_dir=Path("results"),
        status="cancelling",
    )
    run.cancel_event.set()
    manager._runs[run.run_id] = run

    manager._finalize_cancelled_locked(run)

    assert run.status == "cancelled"
    assert run.events[-1].type == "run_cancelled"


def test_run_manager_normalizes_completed_event_metrics():
    manager = RunManager(default_out_dir=Path("missing-results"))
    run = ManagedRun(
        run_id="run_1",
        function_path="file.cpp\\f()",
        variant="full",
        out_dir=Path("results"),
        status="running",
    )

    manager._append_event_locked(
        run,
        "run_completed",
        {
            "stopReason": "coverage_target",
            "metrics": {
                "statement_coverage_pct": 1.0,
                "branch_coverage_pct": 0.5,
                "mcdc_coverage_pct": 0.75,
            },
        },
    )

    assert run.status == "completed"
    assert run.metrics["statementCoveragePct"] == 1.0
    assert run.metrics["branchCoveragePct"] == 0.5
    assert run.metrics["mcdcCoveragePct"] == 0.75


def test_build_report_from_enriched_result():
    summary = {
        "run_id": "run_1",
        "function_path": "file.cpp\\f()",
        "prompt_variant": "full",
        "stop_reason": "coverage_target",
        "error": None,
        "function_source": "int f(int x) {\n  if (x > 0) return 1;\n  return 0;\n}",
        "static_conditions": [
            {
                "node_id": 7,
                "condition": "x > 0",
                "line_in_function": 1,
                "start_offset": 18,
                "end_offset": 23,
            }
        ],
        "metrics": {
            "statement_coverage_pct": 1.0,
            "branch_coverage_pct": 0.5,
            "mcdc_coverage_pct": 0.5,
            "covered_statements": 2,
            "total_statements": 2,
            "covered_branches": 1,
            "total_branches": 2,
            "covered_mcdc_pairs": 1,
            "total_mcdc_pairs": 2,
            "total_input_tokens": 11,
            "total_output_tokens": 5,
            "total_tokens": 16,
            "elapsed_sec": 1.25,
        },
        "test_suite": [
            {
                "iteration": 1,
                "test_name": "tc_positive",
                "status": "PASSED",
                "test_body": "ASSERT_EQ(f(1), 1);",
                "execute_log": "ok",
                "elapsed_ms": 120.0,
                "token_input": 11,
                "token_output": 5,
                "new_mcdc_pairs_covered": 1,
                "is_redundant": False,
                "statement_coverage": {"visited": 2, "total": 2, "progress": 1.0},
                "branch_coverage": {"visited": 1, "total": 2, "progress": 0.5},
                "mcdc_coverage": {"visited": 1, "total": 2, "progress": 0.5},
                "condition_trace": [
                    {
                        "node_id": 7,
                        "condition": "x > 0",
                        "true_branch_visited": True,
                        "false_branch_visited": False,
                        "line_in_function": 1,
                    }
                ],
                "trace_summary": {"raw_step_count": 2},
                "unvisited_mcdc": [
                    {
                        "node_id": 7,
                        "condition": "x > 0",
                        "true_branch_visited": True,
                        "false_branch_visited": False,
                    }
                ],
                "unvisited_statements": [
                    {
                        "node_id": 3,
                        "statement": "return 0;",
                        "line_in_function": 2,
                    }
                ],
                "unvisited_branches": [
                    {
                        "node_id": 7,
                        "condition": "x > 0",
                        "true_visited": True,
                        "false_visited": False,
                        "line_in_function": 1,
                    }
                ],
            },
            {
                "iteration": 2,
                "test_name": "tc_negative",
                "status": "PASSED",
                "new_mcdc_pairs_covered": 1,
                "is_redundant": False,
                "condition_trace": [
                    {
                        "node_id": 7,
                        "condition": "x > 0",
                        "true_branch_visited": False,
                        "false_branch_visited": True,
                        "line_in_function": 1,
                    }
                ],
                "unvisited_mcdc": [],
                "unvisited_statements": [],
                "unvisited_branches": [],
            },
        ],
    }

    report = build_report_from_summary(summary, Path("results/gen_run_1.json"))

    assert report["summary"]["legacy"] is False
    assert report["summary"]["totalTokens"] == 16
    assert report["summary"]["elapsedSec"] == 1.25
    assert report["source"]["lines"][1]["conditionIds"] == [7]
    assert report["conditions"][0]["trueCovered"] is True
    assert report["conditions"][0]["falseCovered"] is True
    assert report["statements"][0]["covered"] is True
    assert report["branches"][0]["missing"] == []
    assert report["tests"][0]["testBody"] == "ASSERT_EQ(f(1), 1);"
    assert report["tests"][0]["executeLog"] == "ok"


def test_build_report_from_legacy_result_does_not_crash():
    summary = {
        "run_id": "legacy_1",
        "function_path": "file.cpp\\f()",
        "prompt_variant": "full",
        "stop_reason": "max_iter",
        "error": None,
        "metrics": {"mcdc_coverage_pct": 0.0},
        "test_suite": [
            {
                "iteration": 1,
                "test_name": "tc_old",
                "status": "COMPILE_ERROR",
                "new_mcdc_pairs_covered": 0,
                "is_redundant": False,
                "test_body": "bad();",
                "execute_log": "compile failed",
            }
        ],
    }

    report = build_report_from_summary(summary)

    assert report["summary"]["legacy"] is True
    assert report["tests"][0]["testBody"] == "bad();"
    assert report["tests"][0]["executeLog"] == "compile failed"
    assert report["conditions"] == []
