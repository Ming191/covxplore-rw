from __future__ import annotations

import time
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError, ExecuteResult
from covxplore.config import get_settings
from covxplore.events import CancelChecker, EventSink, emit_event, keys_to_camel
from covxplore.models import (
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TestSuite,
    TraceSummary,
    UnvisitedBranch,
    UnvisitedMcdc,
    UnvisitedStatement,
)
from covxplore.status import TestStatus, is_failure_status


class FatalToolError(BaseException):
    pass


_suites: dict[str, TestSuite] = {}
_current_run_id: str | None = None
_event_sink: EventSink | None = None
_cancel_checker: CancelChecker | None = None
_max_tests: int | None = None


def get_shared_suite() -> TestSuite | None:
    if _current_run_id is None:
        return None
    return _suites.get(_current_run_id)


def set_current_run(run_id: str) -> None:
    global _current_run_id
    _current_run_id = run_id


def reset_shared_suite(function_path: str, run_id: str) -> TestSuite:
    global _current_run_id
    _current_run_id = run_id
    suite = TestSuite(function_path=function_path)
    _suites[run_id] = suite
    return suite


def cleanup_suite(run_id: str) -> None:
    _suites.pop(run_id, None)


def configure_run_hooks(
    *,
    event_sink: EventSink | None = None,
    cancel_checker: CancelChecker | None = None,
    max_tests: int | None = None,
) -> None:
    global _event_sink, _cancel_checker, _max_tests
    _event_sink = event_sink
    _cancel_checker = cancel_checker
    _max_tests = max_tests


class _Input(BaseModel):
    absolute_path: str = Field(
        ...,
        description=(
            "Absolute path of the function node being tested "
            "(same value used throughout a generation run)."
        ),
    )
    test_body: str = Field(
        ...,
        description=(
            "Complete C++ test driver body to be placed inside AkaUT's test "
            "harness. Must be valid C++ that calls the function under test and "
            "sets up all required input variables. Do NOT include main() or "
            "include guards — AkaUT wraps the body automatically."
        ),
    )
    test_name: str | None = Field(
        default=None,
        description="Optional test case name (auto-generated if omitted).",
    )


class ExecuteTestcaseTool(BaseTool):
    name: str = "execute_testcase"
    description: str = (
        "Compile and execute a C++ test driver body for the target function. "
        "Returns execution status (PASSED/FAILED/RUNTIME_ERROR/COMPILE_ERROR), MC/DC coverage "
        "delta, and a list of still-unvisited condition polarities. Use the "
        "unvisited list to guide your next test."
    )
    args_schema: type[BaseModel] = _Input

    def _run(
        self,
        absolute_path: str,
        test_body: str,
        test_name: str | None = None,
    ) -> str:
        cfg = get_settings()
        suite = get_shared_suite()

        if _cancel_checker is not None and _cancel_checker():
            emit_event(
                _event_sink,
                "log",
                {"level": "warning", "message": "Run cancellation requested."},
            )
            return "[CANCELLED] Run cancellation requested. Do not call more tools."

        if (
            _max_tests is not None
            and suite is not None
            and suite.iteration_count >= _max_tests
        ):
            emit_event(
                _event_sink,
                "log",
                {
                    "level": "warning",
                    "message": f"Max tests reached ({_max_tests}).",
                },
            )
            return (
                f"[MAX_TESTS_REACHED] {_max_tests} tests have already been executed. "
                "Output the final DONE summary without calling more tools."
            )

        emit_event(
            _event_sink,
            "test_started",
            {
                "testName": test_name,
                "absolutePath": absolute_path,
                "iteration": (suite.iteration_count + 1) if suite else None,
            },
        )

        t0 = time.monotonic()
        try:
            with AkaUTClient() as client:
                raw: ExecuteResult = client.execute_testcase(
                    absolute_path, test_body, test_name
                )
        except AkaUTError as exc:
            elapsed = (time.monotonic() - t0) * 1000
            failed = TestResult(
                test_name=test_name or "unknown",
                test_body=test_body,
                status=TestStatus.INFRA_ERROR.value,
                execute_log=str(exc),
                elapsed_ms=elapsed,
            )
            if suite:
                suite.add_result(failed, cfg.min_suite_size)
            emit_event(_event_sink, "test_completed", _test_payload(failed, suite))
            emit_event(_event_sink, "coverage_updated", _suite_payload(suite))
            return (
                f"[INFRA_ERROR] execute_testcase request failed: {exc}\n"
                "This is an AkaUT REST/API/backend issue, not necessarily a C++ "
                "compile error. Check AkaUT server, loaded environment, and path."
            )

        elapsed = (time.monotonic() - t0) * 1000

        result = TestResult(
            test_name=raw.test_name,
            test_body=test_body,
            status=raw.status,
            execute_log=raw.execute_log,
            elapsed_ms=elapsed,
            statement_coverage=CoverageDetail(
                visited=raw.statement_coverage.get("visited", 0),
                total=raw.statement_coverage.get("total", 0),
                progress=raw.statement_coverage.get("progress", 0.0),
            ),
            branch_coverage=CoverageDetail(
                visited=raw.branch_coverage.get("visited", 0),
                total=raw.branch_coverage.get("total", 0),
                progress=raw.branch_coverage.get("progress", 0.0),
            ),
            mcdc_coverage=CoverageDetail(
                visited=raw.mcdc_coverage.get("visited", 0),
                total=raw.mcdc_coverage.get("total", 0),
                progress=raw.mcdc_coverage.get("progress", 0.0),
            ),
            unvisited_mcdc=[
                UnvisitedMcdc(
                    node_id=u.get("nodeId"),
                    condition=u.get("condition", ""),
                    true_branch_visited=u.get("trueBranchVisited", False),
                    false_branch_visited=u.get("falseBranchVisited", False),
                    line_in_function=u.get("lineInFunction"),
                    start_offset=u.get("startOffsetInFunction") or u.get("startOffset"),
                    end_offset=u.get("endOffsetInFunction") or u.get("endOffset"),
                )
                for u in raw.unvisited_mcdc_conditions
            ],
            unvisited_statements=[
                UnvisitedStatement(
                    node_id=s.get("nodeId"),
                    statement=s.get("statement", ""),
                    line_in_function=s.get("lineInFunction"),
                    start_offset=s.get("startOffsetInFunction"),
                    end_offset=s.get("endOffsetInFunction"),
                )
                for s in raw.unvisited_statements
            ],
            unvisited_branches=[
                UnvisitedBranch(
                    node_id=b.get("nodeId"),
                    condition=b.get("condition", ""),
                    true_visited=b.get("trueVisited", False),
                    false_visited=b.get("falseVisited", False),
                    line_in_function=b.get("lineInFunction"),
                    start_offset=b.get("startOffsetInFunction"),
                    end_offset=b.get("endOffsetInFunction"),
                )
                for b in raw.unvisited_branches
            ],
            condition_trace=[
                ConditionTraceEntry(
                    node_id=e.get("nodeId"),
                    condition=e.get("condition", ""),
                    true_branch_visited=e.get("trueBranchVisited", False),
                    false_branch_visited=e.get("falseBranchVisited", False),
                    line_in_function=e.get("lineInFunction"),
                    start_offset_in_function=e.get("startOffsetInFunction"),
                    end_offset_in_function=e.get("endOffsetInFunction"),
                )
                for e in raw.condition_trace
            ],
            trace_summary=TraceSummary(**raw.trace_summary)
            if raw.trace_summary
            else None,
        )

        if suite:
            suite.add_result(result, cfg.min_suite_size)

        emit_event(_event_sink, "test_completed", _test_payload(result, suite))
        emit_event(_event_sink, "coverage_updated", _suite_payload(suite))
        return _format_summary(result, suite)


def _format_summary(result: TestResult, suite: TestSuite | None) -> str:
    lines = []
    redundant_tag = " [REDUNDANT — 0 new MC/DC pairs]" if result.is_redundant else ""
    lines.append(f"===  {result.test_name} | {result.status}{redundant_tag} ===")
    s = result.statement_coverage
    b = result.branch_coverage
    m = result.mcdc_coverage
    lines.append(
        f"This test  → Stmt: {s.visited}/{s.total} ({s.progress * 100:.0f}%) | "
        f"Branch: {b.visited}/{b.total} ({b.progress * 100:.0f}%) | "
        f"MC/DC: {m.visited}/{m.total} ({m.progress * 100:.0f}%) +{result.new_mcdc_pairs_covered} new pairs"
    )
    if suite:
        lines.append(
            f"Suite best → Stmt: {suite.statement_coverage_pct * 100:.0f}% | "
            f"Branch: {suite.branch_coverage_pct * 100:.0f}% | "
            f"MC/DC: {len(suite.covered_keys)}/{suite.total_mcdc_conditions} "
            f"({suite.mcdc_coverage_pct * 100:.0f}%) | "
            f"iter={suite.iteration_count} | "
            f"redundancy={suite.redundancy_rate * 100:.0f}%"
        )
        try:
            gap = suite.coverage_gap_prompt_fragment()
        except RuntimeError as exc:
            raise FatalToolError(str(exc)) from exc
        lines.append("")
        lines.append(gap)

    trace_block = _format_condition_trace(result)
    if trace_block:
        lines.append("")
        lines.append(trace_block)

    if is_failure_status(result.status) and result.execute_log:
        lines.append(f"\nExecution log:\n{result.execute_log.strip()}")

    return "\n".join(lines)


def _coverage_payload(cov: CoverageDetail) -> dict:
    return {"visited": cov.visited, "total": cov.total, "progress": cov.progress}


def _test_payload(result: TestResult, suite: TestSuite | None) -> dict:
    return {
        "iteration": result.iteration,
        "testName": result.test_name,
        "status": result.status,
        "testBody": result.test_body,
        "executeLog": result.execute_log,
        "elapsedMs": result.elapsed_ms,
        "statementCoverage": _coverage_payload(result.statement_coverage),
        "branchCoverage": _coverage_payload(result.branch_coverage),
        "mcdcCoverage": _coverage_payload(result.mcdc_coverage),
        "newMcdcPairsCovered": result.new_mcdc_pairs_covered,
        "isRedundant": result.is_redundant,
        "unvisitedMcdc": keys_to_camel(
            [item.model_dump() for item in result.unvisited_mcdc]
        ),
        "unvisitedStatements": keys_to_camel(
            [item.model_dump() for item in result.unvisited_statements]
        ),
        "unvisitedBranches": keys_to_camel(
            [item.model_dump() for item in result.unvisited_branches]
        ),
        "conditionTrace": keys_to_camel(
            [item.model_dump() for item in result.condition_trace]
        ),
        "traceSummary": keys_to_camel(
            result.trace_summary.model_dump() if result.trace_summary else {}
        ),
        "suite": _suite_payload(suite),
    }


def _suite_payload(suite: TestSuite | None) -> dict:
    if suite is None:
        return {}
    return {
        "statementCoveragePct": suite.statement_coverage_pct,
        "branchCoveragePct": suite.branch_coverage_pct,
        "mcdcCoveragePct": suite.mcdc_coverage_pct,
        "coveredStatements": suite.covered_statements,
        "totalStatements": suite.total_statements,
        "coveredBranches": suite.covered_branches,
        "totalBranches": suite.total_branches,
        "coveredMcdcPairs": len(suite.covered_keys),
        "totalMcdcPairs": suite.total_mcdc_conditions,
        "iterationsUsed": suite.iteration_count,
        "redundancyRate": suite.redundancy_rate,
        "unvisitedMcdc": keys_to_camel(suite.unvisited_summary()),
    }


def _format_condition_trace(result: TestResult) -> str | None:
    if not result.condition_trace:
        return None
    if is_failure_status(result.status):
        return None

    lines = ["Condition evaluation this test:"]
    sorted_trace = sorted(
        result.condition_trace,
        key=lambda e: (
            e.node_id is None,
            e.node_id if e.node_id is not None else float("inf"),
            e.line_in_function is None,
            e.line_in_function if e.line_in_function is not None else float("inf"),
            e.condition,
        ),
    )
    for entry in sorted_trace:
        t_mark = "YES" if entry.true_branch_visited else "NO"
        f_mark = "YES" if entry.false_branch_visited else "NO"
        node_tag = entry.node_id if entry.node_id is not None else "?"
        line_tag = (
            f"line+{entry.line_in_function}"
            if entry.line_in_function is not None
            else "line+?"
        )
        lines.append(
            f"  [node:{node_tag} {line_tag}] {entry.condition!r} TRUE={t_mark} FALSE={f_mark}"
        )

    return "\n".join(lines)
