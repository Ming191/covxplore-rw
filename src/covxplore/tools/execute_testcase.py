from __future__ import annotations

import logging
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, ValidationError

from covxplore.agents.guardrails import validate_tool_input
from covxplore.agents.schemas import GenerateTestAction, GenerateTestBatchAction
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
from covxplore.generation.batch import merge_batch_results
from covxplore.status import TestStatus, is_failure_status


logger = logging.getLogger(__name__)


class FatalToolError(Exception):
    pass


class HardStop(BaseException):
    """Raised to terminate CrewAI loop when generation goal is reached."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


def _is_coverage_done(suite: TestSuite, mcdc_target: float) -> bool:
    """Return True when all applicable coverage targets are satisfied."""
    if suite.iteration_count == 0:
        return False
    metrics = suite.coverage.metrics(suite.tests)
    if metrics.total_statements == 0 and metrics.total_branches == 0:
        return False
    mcdc_done = (
        metrics.total_mcdc_pairs == 0
        or metrics.mcdc_pct >= mcdc_target
        or not suite.coverage.unvisited_summary()
    )
    stmt_done = (
        metrics.total_statements == 0
        or metrics.covered_statements >= metrics.total_statements
    )
    branch_done = (
        metrics.total_branches == 0 or metrics.covered_branches >= metrics.total_branches
    )
    return mcdc_done and stmt_done and branch_done


def _raise_if_hard_stop(suite: TestSuite, cfg: object) -> None:
    """Raise HardStop immediately after redundant streak, fail streak, or coverage target."""
    redundant_limit: int = getattr(cfg, "redundant_streak_limit", 3)
    fail_limit: int = getattr(cfg, "fail_streak_limit", 3)
    mcdc_target: float = getattr(cfg, "mcdc_target", 1.0)

    if suite.coverage.consecutive_redundant >= redundant_limit:
        raise HardStop(
            "redundant_streak",
            f"Hard stop: {redundant_limit} consecutive redundant tests — agent loop terminated.",
        )
    if suite.consecutive_failures() >= fail_limit:
        raise HardStop(
            "fail_streak",
            f"Hard stop: {fail_limit} consecutive failing tests — agent loop terminated.",
        )
    if _is_coverage_done(suite, mcdc_target):
        raise HardStop(
            "coverage_target",
            "Hard stop: all coverage targets met — agent loop terminated.",
        )


@dataclass
class RunContext:
    """Owns suites by explicit generation run id."""

    _suites: dict[str, TestSuite] = field(default_factory=dict)
    _active_run_id: str | None = None

    def get_suite(self, run_id: str) -> TestSuite | None:
        return self._suites.get(run_id)

    def active_suite(self) -> TestSuite | None:
        if self._active_run_id is None:
            return None
        return self._suites.get(self._active_run_id)

    def reset_suite(self, function_path: str, run_id: str) -> TestSuite:
        suite = TestSuite(function_path=function_path)
        self._suites[run_id] = suite
        self._active_run_id = run_id
        return suite

    def cleanup_suite(self, run_id: str) -> None:
        self._suites.pop(run_id, None)
        if self._active_run_id == run_id:
            self._active_run_id = None


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
        test_body: str | None = None,
        test_name: str | None = None,
        target_node_id: int | None = None,
        target_polarity: str | None = None,
        target_reason: str | None = None,
    ) -> str:
        cfg = get_settings()
        suite = self._run_context.active_suite()
        function_path = suite.function_path if suite else None

        action_validation = validate_tool_input(
            {
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
        if function_path is None:
            return (
                "[ACTION_ERROR] execute_testcase could not resolve target function path. "
                "Start a generation run before calling the tool."
            )
        assert test_body is not None

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
                suite.mark_iter(True)
                _raise_if_hard_stop(suite, cfg)
            return (
                "[CONTRACT_ERROR] execute_testcase rejected test body before execution:\n"
                f"{_format_contract_violations(violations)}\n"
                "Send only valid driver body code. Do not include markdown fences, main(), "
                "or whole translation units."
            )

        t0 = time.monotonic()
        try:
            raw: ExecuteResult = self._executor.execute(function_path, test_body, test_name)
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
                    suite.mark_iter(True)
                    _raise_if_hard_stop(suite, cfg)
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

        result = _result_from_execute_result(
            raw,
            test_body=test_body,
            elapsed_ms=elapsed,
            target_node_id=target_node_id,
            target_polarity=target_polarity,
            target_reason=target_reason,
        )

        if suite:
            suite.add_result(result, cfg.min_suite_size)
            suite.mark_iter(is_failure_status(result.status))
            _raise_if_hard_stop(suite, cfg)

        return _format_summary(result, suite, action_validation.warnings)


class ExecuteTestcaseBatchTool(BaseTool):
    name: str = "execute_testcase_batch"
    description: str = (
        "Compile and execute a batch of 3-5 C++ test driver bodies for the active target function. "
        "Prefer distinct nodeId/polarity targets. Returns accepted tests, redundant rejections, "
        "suite coverage, and remaining gap guidance."
    )
    args_schema: type[BaseModel] = GenerateTestBatchAction

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

    async def _arun(self, candidates: list[dict] | list[GenerateTestAction]) -> str:
        cfg = get_settings()
        suite = self._run_context.active_suite()
        if suite is None:
            return (
                "[ACTION_ERROR] execute_testcase_batch could not resolve target function path. "
                "Start a generation run before calling the tool."
            )

        try:
            batch = GenerateTestBatchAction.model_validate({"candidates": candidates})
        except ValidationError as exc:
            return "[ACTION_ERROR] Invalid execute_testcase_batch action:\n" + "\n".join(
                f"- {_format_validation_error(error)}" for error in exc.errors()
            )

        results: list[TestResult] = []
        executable: list[GenerateTestAction] = []
        for candidate in batch.candidates:
            violations = self._validator.validate(candidate.test_body)
            errors = [violation for violation in violations if violation.severity == "ERROR"]
            if errors:
                results.append(_contract_error_result(candidate, violations))
            else:
                executable.append(candidate)

        async_results = await asyncio.gather(
            *(self._execute_one(suite.function_path, candidate) for candidate in executable)
        )
        results.extend(async_results)

        summary = merge_batch_results(suite, results, cfg.min_suite_size)
        summary.record_redundancy(suite)
        suite.mark_iter(
            bool(results) and all(is_failure_status(result.status) for result in results)
        )
        _raise_if_hard_stop(suite, cfg)
        return _format_batch_summary(suite, summary)

    def _run(self, candidates: list[dict] | list[GenerateTestAction]) -> str:
        return asyncio.run(self._arun(candidates))

    async def _execute_one(self, function_path: str, candidate: GenerateTestAction) -> TestResult:
        t0 = time.monotonic()
        try:
            raw = await asyncio.to_thread(
                self._executor.execute,
                function_path,
                candidate.test_body,
                candidate.test_name,
            )
        except AkaUTError as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return TestResult(
                test_name=candidate.test_name or "unknown",
                test_body=candidate.test_body,
                status=(
                    TestStatus.COMPILE_ERROR.value
                    if _is_compile_or_test_body_error(exc)
                    else TestStatus.UNKNOWN.value
                ),
                execute_log=str(exc),
                elapsed_ms=elapsed,
                target_node_id=candidate.target_node_id,
                target_polarity=candidate.target_polarity,
                target_reason=candidate.target_reason,
            )
        elapsed = (time.monotonic() - t0) * 1000
        return _result_from_execute_result(
            raw,
            test_body=candidate.test_body,
            elapsed_ms=elapsed,
            target_node_id=candidate.target_node_id,
            target_polarity=candidate.target_polarity,
            target_reason=candidate.target_reason,
        )


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


def _format_validation_error(error: Mapping[str, Any]) -> str:
    loc = ".".join(str(part) for part in error.get("loc", ())) or "payload"
    return f"{loc}: {error.get('msg', 'invalid value')}"


def _contract_error_result(
    action: GenerateTestAction,
    violations: list[ContractViolation],
) -> TestResult:
    return TestResult(
        test_name=action.test_name or "contract_error",
        test_body=action.test_body,
        status=TestStatus.COMPILE_ERROR.value,
        execute_log=_format_contract_violations(violations),
        target_node_id=action.target_node_id,
        target_polarity=action.target_polarity,
        target_reason=action.target_reason,
    )


def _format_batch_summary(suite: TestSuite, summary) -> str:
    lines = ["=== execute_testcase_batch summary ==="]
    lines.append(
        "Accepted: "
        + (
            ", ".join(f"{r.test_name}({r.status}, +{r.new_mcdc_pairs_covered})" for r in summary.accepted)
            if summary.accepted
            else "none"
        )
    )
    lines.append(
        "Rejected redundant: "
        + (
            ", ".join(r.test_name for r in summary.rejected_redundant)
            if summary.rejected_redundant
            else "none"
        )
    )
    rejected_details = _format_rejected_redundant_details(summary.rejected_redundant)
    if rejected_details:
        lines.append(rejected_details)
    metrics = suite.coverage.metrics(suite.tests)
    lines.append(
        f"Suite best → Stmt: {metrics.statement_pct * 100:.0f}% | "
        f"Branch: {metrics.branch_pct * 100:.0f}% | "
        f"MC/DC: {metrics.covered_mcdc_pairs}/{metrics.total_mcdc_pairs} "
        f"({metrics.mcdc_pct * 100:.0f}%) | "
        f"iter={suite.iteration_count} | redundancy={suite.redundancy_rate * 100:.0f}%"
    )

    failed_logs = [
        _format_execution_log(result)
        for result in summary.accepted
        if is_failure_status(result.status)
    ]
    failed_logs = [log for log in failed_logs if log]
    if failed_logs:
        lines.append("")
        lines.append("Failed execution logs:")
        lines.extend(failed_logs)

    lines.append("")
    lines.append(
        GapAnalyzer().analyze(
            suite.coverage.gap_input(suite.tests, suite.iteration_count)
        ).text
    )
    return "\n".join(lines)


def _format_rejected_redundant_details(results: list[TestResult]) -> str:
    if not results:
        return ""
    lines = ["Redundant diagnostics (0 new MC/DC pairs):"]
    for result in results[:3]:
        target = (
            f" target=node:{result.target_node_id} {result.target_polarity}"
            if result.target_node_id is not None
            else ""
        )
        lines.append(f"- {result.test_name}:{target}")
        target_status = _target_observation_status(result)
        if target_status:
            lines.append(f"  Target observation: {target_status}")
        trace_lines = _condition_trace_lines(result, max_lines=8, preserve_order=True)
        if trace_lines:
            lines.append("  Actual condition order:")
            lines.extend(f"    {line}" for line in trace_lines)
    if len(results) > 3:
        lines.append(f"- ... and {len(results) - 3} more redundant candidate(s)")
    lines.append(
        "If the target node was not evaluated, change upstream input/state so control reaches that node; "
        "for parser/iterator/stream-like arguments, initialize cursor/state at the point immediately before the target path, not necessarily object start."
    )
    return "\n".join(lines)


def _target_observation_status(result: TestResult) -> str:
    if result.target_node_id is None:
        return ""
    target_entries = [e for e in result.condition_trace if e.node_id == result.target_node_id]
    if not target_entries:
        return f"node:{result.target_node_id} was not present in trace"
    entry = target_entries[-1]
    wanted = (result.target_polarity or "").upper()
    if wanted == "TRUE":
        return "target TRUE observed" if entry.true_branch_visited else "target TRUE not observed"
    if wanted == "FALSE":
        return "target FALSE observed" if entry.false_branch_visited else "target FALSE not observed"
    return (
        f"node:{result.target_node_id} observed TRUE={_yes(entry.true_branch_visited)} "
        f"FALSE={_yes(entry.false_branch_visited)}"
    )


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _format_execution_log(result: TestResult) -> str | None:
    if not result.execute_log:
        return None
    log = result.execute_log.strip()
    if not log:
        return None
    if len(log) > 2000:
        log = "...<truncated>...\n" + log[-2000:]
    return f"--- {result.test_name} | {result.status} ---\n{log}"


def _result_from_execute_result(
    raw: ExecuteResult,
    *,
    test_body: str,
    elapsed_ms: float,
    target_node_id: int | None = None,
    target_polarity: str | None = None,
    target_reason: str | None = None,
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
        trace_summary=_parse_trace_summary(raw.trace_summary),
        target_node_id=target_node_id,
        target_polarity=target_polarity,
        target_reason=target_reason,
    )


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

    if is_failure_status(result.status):
        log = _format_execution_log(result)
        if log:
            lines.append(f"\nExecution log:\n{log}")

    return "\n".join(lines)


def _format_condition_trace(result: TestResult) -> str | None:
    trace_lines = _condition_trace_lines(result)
    if not trace_lines or is_failure_status(result.status):
        return None
    return "\n".join(["Condition evaluation this test:", *trace_lines])


def _condition_trace_lines(
    result: TestResult,
    max_lines: int | None = None,
    *,
    preserve_order: bool = False,
) -> list[str]:
    if not result.condition_trace:
        return []
    entries = list(result.condition_trace)
    if not preserve_order:
        entries = sorted(
            entries,
            key=lambda e: (
                *_sort_value(e.node_id),
                *_sort_value(e.line_in_function),
                e.condition,
            ),
        )
    if max_lines is not None:
        entries = entries[:max_lines]
    lines = []
    for index, entry in enumerate(entries, start=1):
        t_mark = _yes(entry.true_branch_visited)
        f_mark = _yes(entry.false_branch_visited)
        node_tag = entry.node_id if entry.node_id is not None else "?"
        line_tag = (
            f"line+{entry.line_in_function}"
            if entry.line_in_function is not None
            else "line+?"
        )
        prefix = f"{index}. " if preserve_order else ""
        lines.append(
            f"{prefix}[node:{node_tag} {line_tag}] {entry.condition!r} TRUE={t_mark} FALSE={f_mark}"
        )
    if max_lines is not None and len(result.condition_trace) > max_lines:
        lines.append(f"... and {len(result.condition_trace) - max_lines} more")
    return lines
