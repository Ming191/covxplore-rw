from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, ValidationError

from covxplore.agents.schemas import GenerateTestAction, GenerateTestBatchAction
from covxplore.api_client import AkaUTError, ExecuteResult
from covxplore.config import get_settings
from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.driver import AkaUTExecutor, DriverContractValidator, TestCaseExecutor
from covxplore.driver.contract import ContractViolation
from covxplore.generation.batch import merge_batch_results
from covxplore.generation.stop_reasons import StopPolicy
from covxplore.status import TestStatus, is_hard_fail
from covxplore.types import (
    CoverageDetail,
    TestResult,
    TestSuite,
    TraceSummary,
    UnvisitedBranch,
    UnvisitedStatement,
)

logger = logging.getLogger(__name__)


class HardStop(BaseException):
    """Terminate the generation loop after a deterministic stop condition."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


@dataclass
class RunContext:
    """Own test suites by explicit generation run ID."""

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


class ExecuteTestcaseBatchTool(BaseTool):
    """Validate, execute, and merge one no-path candidate batch."""

    name: str = "execute_testcase_batch"
    description: str = (
        "Compile and execute one to eight C++ test driver bodies for the active target function. "
        "Returns accepted tests, redundant rejections, execution failures, suite coverage, "
        "and remaining gap guidance."
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
                "Start a generation run before executing tests."
            )

        try:
            batch = GenerateTestBatchAction.model_validate({"candidates": candidates})
        except ValidationError as exc:
            return "[ACTION_ERROR] Invalid execute_testcase_batch action:\n" + "\n".join(
                f"- {_format_validation_error(error)}" for error in exc.errors()
            )

        results: list[TestResult] = []
        for candidate in batch.candidates:
            violations = self._validator.validate(candidate.test_body)
            if any(violation.severity == "ERROR" for violation in violations):
                results.append(_contract_error_result(candidate, violations))
            else:
                results.append(await self._execute_one(suite.function_path, candidate))

        summary = merge_batch_results(suite, results, cfg.min_suite_size)
        summary.record_redundancy(suite)
        suite.record_batch(bool(results) and all(is_hard_fail(result.status) for result in results))
        _raise_if_hard_stop(suite, cfg)
        return _format_batch_summary(suite, summary)

    def _run(self, candidates: list[dict] | list[GenerateTestAction]) -> str:
        return asyncio.run(self._arun(candidates))

    async def _execute_one(self, function_path: str, candidate: GenerateTestAction) -> TestResult:
        started = time.monotonic()
        try:
            raw = await asyncio.to_thread(
                self._executor.execute,
                function_path,
                candidate.test_body,
                candidate.test_name,
            )
        except AkaUTError as exc:
            return TestResult(
                test_name=candidate.test_name or "unknown",
                test_body=candidate.test_body,
                status=(
                    TestStatus.COMPILE_ERROR.value
                    if _is_compile_or_test_body_error(exc)
                    else TestStatus.UNKNOWN.value
                ),
                execute_log=str(exc),
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        return _result_from_execute_result(
            raw,
            test_body=candidate.test_body,
            elapsed_ms=(time.monotonic() - started) * 1000,
        )


def _raise_if_hard_stop(suite: TestSuite, cfg: object) -> None:
    policy = StopPolicy.from_config(cfg)
    reason = policy.hard_stop_reason(suite)
    if reason is not None:
        raise HardStop(reason, policy.hard_stop_message(reason))


def _is_compile_or_test_body_error(exc: AkaUTError) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
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
    )


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
    )


def _format_batch_summary(suite: TestSuite, summary) -> str:
    lines = ["=== execute_testcase_batch summary ==="]
    lines.append(
        "Accepted: "
        + (
            ", ".join(
                f"{result.test_name}({result.status}, +{result.new_structural_coverage})"
                for result in summary.accepted
            )
            if summary.accepted
            else "none"
        )
    )
    lines.append(
        "Rejected redundant: "
        + (
            ", ".join(result.test_name for result in summary.rejected_redundant)
            if summary.rejected_redundant
            else "none"
        )
    )

    failed_logs = [
        _format_execution_log(result)
        for result in summary.accepted
        if result.status != TestStatus.PASSED.value
    ]
    failed_logs = [log for log in failed_logs if log]
    if failed_logs:
        lines.extend(["", "Failed execution logs:", *failed_logs])

    metrics = suite.coverage.metrics(suite.tests)
    lines.append(
        f"Suite best: Stmt {metrics.statement_pct * 100:.0f}% | "
        f"Branch {metrics.branch_pct * 100:.0f}% | "
        f"batch={suite.batch_count} | redundancy={suite.redundancy_rate * 100:.0f}%"
    )
    lines.extend(
        [
            "",
            GapAnalyzer().analyze(
                suite.coverage.gap_input(suite.tests, suite.batch_count)
            ).text,
        ]
    )
    return "\n".join(lines)


def _format_execution_log(result: TestResult) -> str | None:
    if not result.execute_log or not result.execute_log.strip():
        return None
    log = result.execute_log.strip()
    if len(log) > 2000:
        log = "...<truncated>...\n" + log[-2000:]
    return f"--- {result.test_name} | {result.status} ---\n{log}"


def _result_from_execute_result(
    raw: ExecuteResult,
    *,
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
        unvisited_statements=[
            UnvisitedStatement(
                node_id=item.get("nodeId"),
                statement=item.get("statement", ""),
                line_in_function=item.get("lineInFunction"),
                start_offset=item.get("startOffsetInFunction"),
                end_offset=item.get("endOffsetInFunction"),
            )
            for item in raw.unvisited_statements
        ],
        unvisited_branches=[
            UnvisitedBranch(
                node_id=item.get("nodeId"),
                condition=item.get("condition", ""),
                true_visited=item.get("trueVisited", False),
                false_visited=item.get("falseVisited", False),
                line_in_function=item.get("lineInFunction"),
                start_offset=item.get("startOffsetInFunction"),
                end_offset=item.get("endOffsetInFunction"),
            )
            for item in raw.unvisited_branches
        ],
        trace_summary=_parse_trace_summary(raw.trace_summary),
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
