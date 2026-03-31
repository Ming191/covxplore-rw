from __future__ import annotations

import time
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError, ExecuteResult
from covxplore.config import get_settings
from covxplore.models import (
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TestSuite,
    TraceSummary,
    UnvisitedMcdc,
)
from covxplore.status import TestStatus, is_failure_status

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
                )
                for u in raw.unvisited_mcdc_conditions
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

        return _format_summary(result, suite)


def _format_summary(result: TestResult, suite: TestSuite | None) -> str:
    lines = []
    redundant_tag = " [REDUNDANT — 0 new MC/DC pairs]" if result.is_redundant else ""
    lines.append(f"===  {result.test_name} | {result.status}{redundant_tag} ===")
    m = result.mcdc_coverage
    lines.append(
        f"This test  → MC/DC: {m.visited}/{m.total} "
        f"({m.progress * 100:.0f}%) | +{result.new_mcdc_pairs_covered} new pairs"
    )
    if suite:
        lines.append(
            f"Suite total → MC/DC: {len(suite.covered_keys)}/{suite.total_mcdc_conditions} "
            f"({suite.mcdc_coverage_pct * 100:.0f}%) | "
            f"iter={suite.iteration_count} | "
            f"redundancy={suite.redundancy_rate * 100:.0f}%"
        )
        gap = suite.coverage_gap_prompt_fragment()
        lines.append("")
        lines.append(gap)

    trace_block = _format_condition_trace(result)
    if trace_block:
        lines.append("")
        lines.append(trace_block)

    if is_failure_status(result.status) and result.execute_log:
        lines.append(f"\nExecution log:\n{result.execute_log.strip()}")

    return "\n".join(lines)


def _format_condition_trace(result: TestResult) -> str | None:
    if not result.condition_trace:
        return None
    if is_failure_status(result.status):
        return None

    trace_map: dict[str, tuple[bool, bool, int | None]] = {}
    for entry in result.condition_trace:
        prev = trace_map.get(entry.condition, (False, False, entry.line_in_function))
        trace_map[entry.condition] = (
            prev[0] or entry.true_branch_visited,
            prev[1] or entry.false_branch_visited,
            prev[2] if prev[2] is not None else entry.line_in_function,
        )

    if not trace_map:
        return None

    lines = ["Condition evaluation this test:"]
    for cond, (tv, fv, line) in trace_map.items():
        t_mark = "YES" if tv else "NO "
        f_mark = "YES" if fv else "NO "
        line_tag = f"line+{line}" if line is not None else "line+?"
        lines.append(f"  [{line_tag}] {cond!r:50s}  TRUE={t_mark}  FALSE={f_mark}")

    return "\n".join(lines)
