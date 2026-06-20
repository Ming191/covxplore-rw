from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, ValidationError

from covxplore.agents.guardrails import validate_tool_input
from covxplore.agents.schemas import GenerateTestAction
from covxplore.api_client import AkaUTError, ExecuteResult
from covxplore.config import get_settings
from covxplore.driver.contract import ContractViolation
from covxplore.driver import AkaUTExecutor, DriverContractValidator, TestCaseExecutor
from covxplore.types import (
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TestSuite,
    TraceSummary,
    UnvisitedBranch,
    UnvisitedMcdc,
    UnvisitedStatement,
)
from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.status import TestStatus, is_failure_status


logger = logging.getLogger(__name__)


class FatalToolError(Exception):
    pass


@dataclass
class RunContext:
    """Owns suites by explicit generation run id."""

    _suites: dict[str, TestSuite] = field(default_factory=dict)

    def get_suite(self, run_id: str) -> TestSuite | None:
        return self._suites.get(run_id)

    def reset_suite(self, function_path: str, run_id: str) -> TestSuite:
        suite = TestSuite(function_path=function_path)
        self._suites[run_id] = suite
        return suite

    def cleanup_suite(self, run_id: str) -> None:
        self._suites.pop(run_id, None)


class ExecuteTestcaseTool(BaseTool):
    name: str = "execute_testcase"
    description: str = (
        "Compile and execute a C++ test driver body for the target function. "
        "Returns execution status (PASSED/FAILED/RUNTIME_ERROR/COMPILE_ERROR), MC/DC coverage "
        "delta, and a list of still-unvisited condition polarities. Use the "
        "unvisited list to guide your next test."
    )
    args_schema: type[BaseModel] = GenerateTestAction

    _run_context: RunContext = PrivateAttr(default_factory=RunContext)
    _executor: TestCaseExecutor = PrivateAttr(default_factory=AkaUTExecutor)
    _validator: DriverContractValidator = PrivateAttr(default_factory=DriverContractValidator)

    def __init__(
        self,
        run_context: RunContext | None = None,
        executor: TestCaseExecutor | None = None,
        validator: DriverContractValidator | None = None,
        **data,
    ):
        super().__init__(**data)
        if run_context is not None:
            object.__setattr__(self, "_run_context", run_context)
        if executor is not None:
            object.__setattr__(self, "_executor", executor)
        if validator is not None:
            object.__setattr__(self, "_validator", validator)

    def _run(
        self,
        run_id: str,
        absolute_path: str,
        test_body: str,
        test_name: str | None = None,
        target_node_id: int | None = None,
        target_polarity: str | None = None,
        target_reason: str | None = None,
    ) -> str:
        cfg = get_settings()
        suite = self._run_context.get_suite(run_id)

        action_validation = validate_tool_input(
            {
                "run_id": run_id,
                "absolute_path": absolute_path,
                "test_body": test_body,
                "test_name": test_name,
                "target_node_id": target_node_id,
                "target_polarity": target_polarity,
                "target_reason": target_reason,
            }
        )
        if not action_validation.ok:
            return "[ACTION_ERROR] Invalid execute_testcase action:\n" + "\n".join(
                f"- {error}" for error in action_validation.errors
            )

        violations = self._validator.validate(test_body)
        errors = [violation for violation in violations if violation.severity == "ERROR"]
        if errors:
            elapsed = 0.0
            failed = TestResult(
                test_name=test_name or "contract_error",
                test_body=test_body,
                status=TestStatus.COMPILE_ERROR.value,
                execute_log=_format_contract_violations(violations),
                elapsed_ms=elapsed,
                target_node_id=target_node_id,
                target_polarity=target_polarity,
                target_reason=target_reason,
            )
            if suite:
                suite.add_result(failed, cfg.min_suite_size)
            return (
                "[CONTRACT_ERROR] execute_testcase rejected test body before execution:\n"
                f"{_format_contract_violations(violations)}\n"
                "Send only valid driver body code. Do not include markdown fences, main(), "
                "or whole translation units."
            )

        t0 = time.monotonic()
        try:
            raw: ExecuteResult = self._executor.execute(absolute_path, test_body, test_name)
        except AkaUTError as exc:
            elapsed = (time.monotonic() - t0) * 1000
            if _is_compile_or_test_body_error(exc):
                failed = TestResult(
                    test_name=test_name or "unknown",
                    test_body=test_body,
                    status=TestStatus.COMPILE_ERROR.value,
                    execute_log=str(exc),
                    elapsed_ms=elapsed,
                    target_node_id=target_node_id,
                    target_polarity=target_polarity,
                    target_reason=target_reason,
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
            target_node_id=target_node_id,
            target_polarity=target_polarity,
            target_reason=target_reason,
        )

        if suite:
            suite.add_result(result, cfg.min_suite_size)

        return _format_summary(result, suite, action_validation.warnings)


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


def _format_contract_violations(violations: list[ContractViolation]) -> str:
    return "\n".join(
        f"- {violation.severity} {violation.code}: {violation.message}"
        for violation in violations
    )


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


def _format_summary(
    result: TestResult,
    suite: TestSuite | None,
    action_warnings: list[str] | None = None,
) -> str:
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
    if result.target_node_id is not None:
        lines.append(
            f"Target → node={result.target_node_id} polarity={result.target_polarity}"
            + (f" reason={result.target_reason}" if result.target_reason else "")
        )
    if action_warnings:
        lines.append("Action warnings:")
        lines.extend(f"- {warning}" for warning in action_warnings)
    if suite:
        metrics = suite.coverage.metrics(suite.tests)
        lines.append(
            f"Suite best → Stmt: {metrics.statement_pct * 100:.0f}% | "
            f"Branch: {metrics.branch_pct * 100:.0f}% | "
            f"MC/DC: {metrics.covered_mcdc_pairs}/{metrics.total_mcdc_pairs} "
            f"({metrics.mcdc_pct * 100:.0f}%) | "
            f"iter={suite.iteration_count} | "
            f"redundancy={suite.redundancy_rate * 100:.0f}%"
        )
        try:
            gap = GapAnalyzer().analyze(
                suite.coverage.gap_input(suite.tests, suite.iteration_count)
            ).text
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
