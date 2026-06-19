from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from covxplore.api_client import AkaUTClient, AkaUTError, ExecuteResult
from covxplore.config import get_settings
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


logger = logging.getLogger(__name__)


class FatalToolError(Exception):
    pass


@dataclass
class RunContext:
    """Owns suites dict + current_run_id for one generation run.

    Assumes one active run per context; cleanup resets stale active-run state.
    """

    _suites: dict[str, TestSuite] = field(default_factory=dict)
    _current_run_id: str | None = field(default=None, init=False)

    def get_shared_suite(self) -> TestSuite | None:
        if self._current_run_id is None:
            return None
        return self._suites.get(self._current_run_id)

    def set_current_run(self, run_id: str) -> None:
        self._current_run_id = run_id

    def reset_shared_suite(self, function_path: str, run_id: str) -> TestSuite:
        self._current_run_id = run_id
        suite = TestSuite(function_path=function_path)
        self._suites[run_id] = suite
        return suite

    def cleanup_suite(self, run_id: str) -> None:
        self._suites.pop(run_id, None)
        if self._current_run_id == run_id:
            self._current_run_id = None


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

    _run_context: RunContext = PrivateAttr(default_factory=RunContext)

    def __init__(self, run_context: RunContext | None = None, **data):
        super().__init__(**data)
        if run_context is not None:
            object.__setattr__(self, "_run_context", run_context)

    def _run(
        self,
        absolute_path: str,
        test_body: str,
        test_name: str | None = None,
    ) -> str:
        cfg = get_settings()
        suite = self._run_context.get_shared_suite()

        t0 = time.monotonic()
        try:
            with AkaUTClient() as client:
                raw: ExecuteResult = client.execute_testcase(
                    absolute_path, test_body, test_name
                )
        except AkaUTError as exc:
            elapsed = (time.monotonic() - t0) * 1000
            if _is_compile_or_test_body_error(exc):
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
            return (
                f"[EXECUTE_ERROR] execute_testcase API/server error: {exc}\n"
                "AkaUT did not return a confirmed compile/test-body failure. "
                "Check server health, network/API response, and backend logs before changing the test body."
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
            trace_summary=_parse_trace_summary(raw.trace_summary),
        )

        if suite:
            suite.add_result(result, cfg.min_suite_size)

        return _format_summary(result, suite)


def _is_compile_or_test_body_error(exc: AkaUTError) -> bool:
    message = str(exc).lower()
    compile_markers = (
        "compile",
        "compilation",
        "syntax error",
        "test body",
        "testbody",
        "missing include",
        "incorrect variable type",
    )
    return any(marker in message for marker in compile_markers)


def _parse_trace_summary(raw: dict | None) -> TraceSummary | None:
    if not raw:
        return None
    try:
        return TraceSummary(**raw)
    except (TypeError, ValidationError) as exc:
        logger.debug("Ignoring invalid traceSummary payload: %s", exc)
        return None


def _sort_value(value: object) -> tuple[bool, str]:
    return value is None, str(value)


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


def _format_condition_trace(result: TestResult) -> str | None:
    if not result.condition_trace:
        return None
    if is_failure_status(result.status):
        return None

    lines = ["Condition evaluation this test:"]
    sorted_trace = sorted(
        result.condition_trace,
        key=lambda e: (
            *_sort_value(e.node_id),
            *_sort_value(e.line_in_function),
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
