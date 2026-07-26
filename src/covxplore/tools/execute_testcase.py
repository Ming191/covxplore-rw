from __future__ import annotations

import logging
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, ValidationError

from covxplore.agents.guardrails import validate_tool_input
from covxplore.agents.schemas import (
    GenerateTestAction,
    GenerateTestBatchAction,
    GenerateTestBatchAnyAction,
    GenerateTestBatchOptionalPathAction,
    GenerateTestBatchSingleAction,
)
from covxplore.api_client import AkaUTError, ExecuteResult
from covxplore.config import get_settings
from covxplore.driver.contract import ContractViolation
from covxplore.driver import AkaUTExecutor, DriverContractValidator, TestCaseExecutor
from covxplore.types import (
    CoverageDetail,
    ExpectedPathStep,
    TestResult,
    TestSuite,
    TraceSummary,
    UnvisitedBranch,
    UnvisitedStatement,
)
from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.coverage.path_comparison import (
    compare_expected_path,
    effective_expected_path,
    format_divergence,
)
from covxplore.generation.batch import filter_preflight_candidates, merge_batch_results
from covxplore.generation.stop_reasons import StopPolicy
from covxplore.status import TestStatus, is_hard_fail


logger = logging.getLogger(__name__)


class FatalToolError(Exception):
    pass


class HardStop(BaseException):
    """Raised to terminate CrewAI loop when generation goal is reached."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


def _raise_if_hard_stop(suite: TestSuite, cfg: object) -> None:
    """Raise HardStop immediately after redundant streak, fail streak, or coverage target."""
    policy = StopPolicy.from_config(cfg)
    reason = policy.hard_stop_reason(suite)
    if reason is not None:
        raise HardStop(reason, policy.hard_stop_message(reason))


@dataclass
class RunContext:
    """Owns suites by explicit generation run id."""

    _suites: dict[str, TestSuite] = field(default_factory=dict)
    _active_run_id: str | None = None
    branch_catalog_ids: set[int] = field(default_factory=set)
    path_feedback: bool = True
    include_exec_detail: bool = True
    max_batch_candidates: int = 5

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

    def set_branch_catalog_ids(self, node_ids: set[int] | list[int] | None) -> None:
        self.branch_catalog_ids = {int(node_id) for node_id in (node_ids or [])}

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
        expected_path: list[dict | ExpectedPathStep] | None = None,
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
                "expected_path": expected_path or [],
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
        action = GenerateTestAction.model_validate(
            {
                "test_body": test_body,
                "test_name": test_name,
                "target_node_id": target_node_id,
                "target_polarity": target_polarity,
                "target_reason": target_reason,
                "expected_path": expected_path or [],
            }
        )

        violations = self._validator.validate(test_body)
        errors = [violation for violation in violations if violation.severity == "ERROR"]
        if errors:
            elapsed = 0.0
            failed = _contract_error_result(action, violations)
            failed.elapsed_ms = elapsed
            if suite:
                suite.add_result(failed, cfg.min_suite_size)
                suite.record_batch(True)
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
                    target_node_id=action.target_node_id,
                    target_polarity=action.target_polarity,
                    target_reason=action.target_reason,
                    expected_path=list(action.expected_path),
                )
                if suite:
                    suite.add_result(failed, cfg.min_suite_size)
                    suite.record_batch(True)
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
            target_node_id=action.target_node_id,
            target_polarity=action.target_polarity,
            target_reason=action.target_reason,
            expected_path=list(action.expected_path),
        )

        if suite:
            suite.add_result(result, cfg.min_suite_size)
            suite.record_batch(is_hard_fail(result.status))
            _raise_if_hard_stop(suite, cfg)

        return _format_summary(result, suite, action_validation.warnings)


class ExecuteTestcaseBatchTool(BaseTool):
    name: str = "execute_testcase_batch"
    description: str = (
        "Compile and execute a batch of 3-5 C++ test driver bodies for the active target function. "
        "Every candidate MUST include a non-empty expected_path (ordered TRUE/FALSE branch outcomes "
        "from function entry to the target). Prefer distinct structural targets. "
        "Returns accepted tests, redundant rejections, path-divergence diagnostics, suite coverage, "
        "and remaining gap guidance."
    )
    args_schema: type[BaseModel] = GenerateTestBatchAction
    batch_schema: ClassVar[type[BaseModel]] = GenerateTestBatchAction

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
            batch = self.batch_schema.model_validate({"candidates": candidates})
        except ValidationError as exc:
            return "[ACTION_ERROR] Invalid execute_testcase_batch action:\n" + "\n".join(
                f"- {_format_validation_error(error)}" for error in exc.errors()
            )

        results: list[TestResult] = []
        contract_ok: list[GenerateTestAction] = []
        for candidate in batch.candidates:
            violations = self._validator.validate(candidate.test_body)
            errors = [violation for violation in violations if violation.severity == "ERROR"]
            if errors:
                results.append(_contract_error_result(candidate, violations))
            else:
                contract_ok.append(candidate)

        preflight = filter_preflight_candidates(suite, contract_ok)
        async_results = await asyncio.gather(
            *(self._execute_one(suite.function_path, candidate) for candidate in preflight.executable)
        )
        results.extend(async_results)

        summary = merge_batch_results(
            suite, results, cfg.min_suite_size, preflight_notes=preflight.notes
        )
        summary.record_redundancy(suite)
        suite.record_batch(
            bool(results) and all(is_hard_fail(result.status) for result in results)
        )
        _raise_if_hard_stop(suite, cfg)
        known = self._run_context.branch_catalog_ids or None
        return _format_batch_summary(
            suite,
            summary,
            known_node_ids=known,
            path_feedback=self._run_context.path_feedback,
            include_exec_detail=self._run_context.include_exec_detail,
        )

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
                expected_path=list(candidate.expected_path),
            )
        elapsed = (time.monotonic() - t0) * 1000
        return _result_from_execute_result(
            raw,
            test_body=candidate.test_body,
            elapsed_ms=elapsed,
            target_node_id=candidate.target_node_id,
            target_polarity=candidate.target_polarity,
            target_reason=candidate.target_reason,
            expected_path=list(candidate.expected_path),
        )


class ExecuteTestcaseBatchAnyTool(ExecuteTestcaseBatchTool):
    description: str = (
        "Compile and execute one broad batch of C++ test driver bodies for the active target function. "
        "Use one distinct candidate per useful uncovered structural gap; avoid duplicate path shapes. "
        "Large batches are allowed for one-shot coverage planning but produce long feedback."
    )
    args_schema: type[BaseModel] = GenerateTestBatchAnyAction
    batch_schema: ClassVar[type[BaseModel]] = GenerateTestBatchAnyAction


class ExecuteTestcaseBatchSingleTool(ExecuteTestcaseBatchTool):
    """Ablation: force one candidate per batch."""

    description: str = (
        "Compile and execute exactly one C++ test driver body for the active target function. "
        "Submit a single candidate with a non-empty expected_path. "
        "Returns accepted/redundant status, suite coverage, and remaining gap guidance."
    )
    args_schema: type[BaseModel] = GenerateTestBatchSingleAction
    batch_schema: ClassVar[type[BaseModel]] = GenerateTestBatchSingleAction


class ExecuteTestcaseBatchOptionalPathTool(ExecuteTestcaseBatchTool):
    """No-path ablation: expected_path is optional on each candidate."""

    description: str = (
        "Compile and execute a batch of 3-5 C++ test driver bodies for the active target function. "
        "Prefer distinct structural targets. expected_path is optional in this mode. "
        "Returns accepted tests, redundant rejections, suite coverage, and remaining gap guidance."
    )
    args_schema: type[BaseModel] = GenerateTestBatchOptionalPathAction
    batch_schema: ClassVar[type[BaseModel]] = GenerateTestBatchOptionalPathAction


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
        "cannot find",
        ".cpp.out",
        "ld returned",
        "collect2",
        "file not recognized",
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
        expected_path=list(action.expected_path),
    )


def _format_batch_summary(
    suite: TestSuite,
    summary,
    *,
    known_node_ids: set[int] | None = None,
    path_feedback: bool = True,
    include_exec_detail: bool = True,
) -> str:
    lines = ["=== execute_testcase_batch summary ==="]
    lines.append(
        "Accepted: "
        + (
            ", ".join(f"{r.test_name}({r.status}, +{r.new_structural_coverage})" for r in summary.accepted)
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
    if summary.preflight_notes:
        lines.append("Preflight skipped (not executed):")
        lines.extend(f"- {note}" for note in summary.preflight_notes)
    rejected_details = _format_rejected_redundant_details(
        summary.rejected_redundant, known_node_ids=known_node_ids
    )
    if rejected_details:
        lines.append(rejected_details)
    if path_feedback:
        path_details = _format_path_diagnostics(
            summary.accepted + summary.rejected_redundant,
            known_node_ids=known_node_ids,
        )
        if path_details:
            lines.append(path_details)
    metrics = suite.coverage.metrics(suite.tests)
    lines.append(
        f"Suite best → Stmt: {metrics.statement_pct * 100:.0f}% | "
        f"Branch: {metrics.branch_pct * 100:.0f}% | "
        f"batch={suite.batch_count} | redundancy={suite.redundancy_rate * 100:.0f}%"
    )

    if include_exec_detail:
        failed_logs = [
            _format_execution_log(result)
            for result in summary.accepted
            if result.status != TestStatus.PASSED.value
        ]
        failed_logs = [log for log in failed_logs if log]
        if failed_logs:
            lines.append("")
            lines.append("Failed execution logs:")
            lines.extend(failed_logs)

    lines.append("")
    lines.append(
        GapAnalyzer().analyze(
            suite.coverage.gap_input(suite.tests, suite.batch_count)
        ).text
    )
    return "\n".join(lines)


def _format_rejected_redundant_details(
    results: list[TestResult],
    *,
    known_node_ids: set[int] | None = None,
) -> str:
    if not results:
        return ""
    lines = ["Redundant diagnostics (0 new structural coverage):"]
    for result in results[:3]:
        target = (
            f" target=node:{result.target_node_id} {result.target_polarity}"
            if result.target_node_id is not None
            else ""
        )
        lines.append(f"- {result.test_name}:{target}")
        target_status = _target_observation_status(result, known_node_ids=known_node_ids)
        if target_status:
            lines.append(f"  Target observation: {target_status}")
    if len(results) > 3:
        lines.append(f"- ... and {len(results) - 3} more redundant candidate(s)")
    lines.append(
        "If the target node was not evaluated, change upstream input/state so control reaches that node; "
        "for parser/iterator/stream-like arguments, initialize cursor/state at the point immediately before the target path, not necessarily object start."
    )
    return "\n".join(lines)


def _format_path_diagnostics(
    results: list[TestResult],
    *,
    known_node_ids: set[int] | None = None,
) -> str:
    """Report expected_path vs runtime mismatches for executed candidates."""
    lines: list[str] = []
    for result in results:
        if not effective_expected_path(result):
            continue
        comparison = compare_expected_path(result, known_node_ids=known_node_ids)
        path = effective_expected_path(result)
        path_text = " → ".join(f"node{s.node_id}={s.polarity}" for s in path)
        if comparison.matched is True:
            lines.append(f"- {result.test_name}: expected_path matched ({path_text})")
            continue
        if comparison.matched is False:
            divergence = format_divergence(result, known_node_ids=known_node_ids)
            lines.append(
                f"- {result.test_name}: PATH DIVERGENCE — predicted [{path_text}]; "
                f"{divergence or 'predicted polarity was not observed'}"
            )
    if not lines:
        return ""
    return "Path diagnostics:\n" + "\n".join(lines)


def _target_observation_status(
    result: TestResult,
    *,
    known_node_ids: set[int] | None = None,
) -> str:
    comparison = compare_expected_path(result, known_node_ids=known_node_ids)
    if comparison.matched is None:
        return "no expected_path/target provided"
    if comparison.matched is True:
        path = effective_expected_path(result)
        return "matched expected_path (" + " → ".join(
            f"node{s.node_id}={s.polarity}" for s in path
        ) + ")"
    return format_divergence(result, known_node_ids=known_node_ids) or "predicted path diverged from runtime"


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
    expected_path: list[ExpectedPathStep] | None = None,
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
        trace_summary=_parse_trace_summary(raw.trace_summary),
        target_node_id=target_node_id,
        target_polarity=target_polarity,
        target_reason=target_reason,
        expected_path=list(expected_path or []),
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
    redundant_tag = " [REDUNDANT — 0 new structural coverage]" if result.is_redundant else ""
    lines.append(f"===  {result.test_name} | {result.status}{redundant_tag} ===")
    s = result.statement_coverage
    b = result.branch_coverage
    lines.append(
        f"This test → Stmt: {s.visited}/{s.total} ({s.progress * 100:.0f}%) | "
        f"Branch: {b.visited}/{b.total} ({b.progress * 100:.0f}%) | "
        f"+{result.new_structural_coverage} new structural coverage"
    )
    if result.target_node_id is not None:
        lines.append(
            f"Target → node={result.target_node_id} polarity={result.target_polarity}"
            + (f" reason={result.target_reason}" if result.target_reason else "")
        )
    path = effective_expected_path(result)
    if path:
        lines.append(
            "Expected path → " + " → ".join(f"node{s.node_id}={s.polarity}" for s in path)
        )
        path_status = _target_observation_status(result)
        if path_status:
            lines.append(f"Path observation: {path_status}")
    if action_warnings:
        lines.append("Action warnings:")
        lines.extend(f"- {warning}" for warning in action_warnings)
    if suite:
        metrics = suite.coverage.metrics(suite.tests)
        lines.append(
            f"Suite best → Stmt: {metrics.statement_pct * 100:.0f}% | "
            f"Branch: {metrics.branch_pct * 100:.0f}% | "
            f"batch={suite.batch_count} | "
            f"redundancy={suite.redundancy_rate * 100:.0f}%"
        )
        try:
            gap = GapAnalyzer().analyze(
                suite.coverage.gap_input(suite.tests, suite.batch_count)
            ).text
        except RuntimeError as exc:
            raise FatalToolError(str(exc)) from exc
        lines.append("")
        lines.append(gap)

    if result.status != TestStatus.PASSED.value:
        log = _format_execution_log(result)
        if log:
            lines.append(f"\nExecution log:\n{log}")

    return "\n".join(lines)