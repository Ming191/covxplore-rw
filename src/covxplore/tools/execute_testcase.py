from __future__ import annotations

import time
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError, ExecuteResult
from covxplore.config import get_settings
from covxplore.coverage_models import (
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TraceSummary,
    UnvisitedBranch,
    UnvisitedMcdc,
    UnvisitedStatement,
)
from covxplore.status import TestStatus, is_failure_status
from covxplore.test_suite import TestSuite


class FatalToolError(RuntimeError):
    """Fatal tool execution error that should stop the current generation run."""


_suites: dict[str, TestSuite] = {}
_current_run_id: str | None = None


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


def test_result_from_execute_result(
    raw: ExecuteResult,
    test_body: str,
    elapsed_ms: float,
) -> TestResult:
    return TestResult(
        test_name=raw.test_name,
        test_body=test_body,
        status=raw.status,
        execute_log=raw.execute_log,
        elapsed_ms=elapsed_ms,
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
                status=TestStatus.COMPILE_ERROR.value,
                execute_log=str(exc),
                elapsed_ms=elapsed,
            )
            if suite:
                suite.add_result(failed, cfg.min_suite_size)
            return (
                f"[COMPILE_ERROR] execute_testcase failed: {exc}\n"
                "Review the test body for syntax errors, missing includes, or "
                "incorrect variable types and try again."
            )

        elapsed = (time.monotonic() - t0) * 1000

        result = test_result_from_execute_result(raw, test_body, elapsed)

        if suite:
            suite.add_result(result, cfg.min_suite_size)

        return _format_summary(result, suite)


def _format_summary(result: TestResult, suite: TestSuite | None) -> str:
    lines = []
    redundant_tag = " [REDUNDANT — 0 new MC/DC pairs]" if result.is_redundant else ""
    lines.append(f"===  {result.test_name} | {result.status}{redundant_tag} ===")
    s = result.statement_coverage
    b = result.branch_coverage
    m = result.mcdc_coverage
    lines.append(
        f"This test  → Stmt: {s.visited}/{s.total} ({s.progress * 100:.0f}%) "
        f"+{result.new_statements_covered} new | "
        f"Branch: {b.visited}/{b.total} ({b.progress * 100:.0f}%) "
        f"+{result.new_branches_covered} new | "
        f"MC/DC: {m.visited}/{m.total} ({m.progress * 100:.0f}%) "
        f"+{result.new_mcdc_pairs_covered} new pairs"
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
        lines.append("")
        lines.append(_format_next_step_hint(result, suite))

    trace_block = _format_condition_trace(result)
    if trace_block:
        lines.append("")
        lines.append(trace_block)

    if is_failure_status(result.status) and result.execute_log:
        lines.append(f"\nExecution log:\n{result.execute_log.strip()}")

    return "\n".join(lines)


def _format_next_step_hint(result: TestResult, suite: TestSuite) -> str:
    gained = (
        result.new_statements_covered
        + result.new_branches_covered
        + result.new_mcdc_pairs_covered
    )
    remaining_parts = []
    if suite.cumulative_unvisited_statements:
        remaining_parts.append(f"{len(suite.cumulative_unvisited_statements)} stmt")
    if suite.cumulative_unvisited_branches:
        remaining_parts.append(f"{len(suite.cumulative_unvisited_branches)} branch-node")
    if suite.unvisited_summary():
        remaining_parts.append(f"{len(suite.unvisited_summary())} MC/DC node")
    remaining = ", ".join(remaining_parts) if remaining_parts else "no known uncovered targets"

    if gained > 0:
        return (
            f"Next-step hint: this candidate improved coverage; remaining targets: {remaining}. "
            "Continue with a different uncovered nodeId/polarity that can add more coverage."
        )
    if result.is_redundant:
        return (
            f"Next-step hint: redundant candidate; remaining targets: {remaining}. "
            "Do not retry a near-duplicate body. Switch input category, nodeId family, or branch polarity."
        )
    return (
        f"Next-step hint: no new cumulative coverage from this candidate; remaining targets: {remaining}. "
        "Prefer a structurally different path over small value tweaks."
    )


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
