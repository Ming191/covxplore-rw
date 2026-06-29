from __future__ import annotations

import json
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
    fail_limit: int = getattr(cfg, "fail_streak_limit", 5)
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
    _allowed_targets: dict[str, set[tuple[int, str]]] = field(default_factory=dict)

    def get_suite(self, run_id: str) -> TestSuite | None:
        return self._suites.get(run_id)

    def allowed_targets(self, run_id: str) -> set[tuple[int, str]] | None:
        return self._allowed_targets.get(run_id)

    def set_allowed_targets(self, run_id: str, targets: list[dict]) -> None:
        self._allowed_targets[run_id] = {
            (target["node_id"], target["polarity"])
            for target in targets
            if target["node_id"] is not None
        }

    def reset_suite(self, function_path: str, run_id: str) -> TestSuite:
        suite = TestSuite(function_path=function_path)
        self._suites[run_id] = suite
        self._allowed_targets.pop(run_id, None)
        return suite

    def cleanup_suite(self, run_id: str) -> None:
        self._suites.pop(run_id, None)
        self._allowed_targets.pop(run_id, None)


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

        top_targets = getattr(cfg, "compact_feedback_top_targets", 3)
        if getattr(cfg, "compact_feedback", False):
            allowed_targets = self._run_context.allowed_targets(run_id)
            requested_target = (
                (target_node_id, target_polarity.upper())
                if target_node_id is not None and target_polarity is not None
                else None
            )
            if allowed_targets is not None and requested_target not in allowed_targets:
                return _format_target_not_allowed(
                    target_node_id,
                    target_polarity,
                    _next_targets(
                        TestResult(test_name="", test_body="", status="UNKNOWN"),
                        suite,
                        top_targets,
                    ),
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
            if getattr(cfg, "compact_feedback", False):
                next_targets = _next_targets(failed, suite, top_targets)
                self._run_context.set_allowed_targets(run_id, next_targets)
                return _format_compact_summary(failed, suite, [], top_targets, next_targets)
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
                if getattr(cfg, "compact_feedback", False):
                    next_targets = _next_targets(failed, suite, top_targets)
                    self._run_context.set_allowed_targets(run_id, next_targets)
                    return _format_compact_summary(failed, suite, [], top_targets, next_targets)
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

        next_targets = None
        if suite:
            suite.add_result(result, cfg.min_suite_size)
            if getattr(cfg, "compact_feedback", False):
                next_targets = _next_targets(result, suite, top_targets)
            _raise_if_hard_stop(suite, cfg)

        if getattr(cfg, "compact_feedback", False):
            next_targets = next_targets or _next_targets(result, suite, top_targets)
            self._run_context.set_allowed_targets(run_id, next_targets)
            return _format_compact_summary(
                result,
                suite,
                action_validation.warnings,
                top_targets,
                next_targets,
            )
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


def _target_hit(result: TestResult) -> bool | None:
    if result.target_node_id is None or result.target_polarity is None:
        return None
    want_true = result.target_polarity.upper() == "TRUE"
    for entry in result.condition_trace:
        if entry.node_id != result.target_node_id:
            continue
        return entry.true_branch_visited if want_true else entry.false_branch_visited
    return False


def _target_attempts(suite: TestSuite | None, node_id: int | None, polarity: str) -> tuple[int, int]:
    if suite is None or node_id is None:
        return 0, 0
    attempts = 0
    misses = 0
    wanted = polarity.upper()
    for test in suite.tests:
        if test.target_node_id == node_id and (test.target_polarity or "").upper() == wanted:
            attempts += 1
            if _target_hit(test) is False:
                misses += 1
    return attempts, misses


def _last_target_redundant(suite: TestSuite | None, node_id: int | None, polarity: str) -> bool:
    if suite is None or node_id is None:
        return False
    wanted = polarity.upper()
    for test in reversed(suite.tests):
        if test.target_node_id == node_id and (test.target_polarity or "").upper() == wanted:
            return test.is_redundant
    return False


def _target_score(target: dict, suite: TestSuite | None) -> int:
    base = 100 if target["kind"] == "mcdc" else 40
    attempts = int(target["attempts"])
    misses = int(target["misses"])
    redundant_penalty = 100 if _last_target_redundant(
        suite, target["node_id"], target["polarity"]
    ) else 0
    return base - 25 * attempts - 60 * misses - redundant_penalty


def _next_targets(
    result: TestResult,
    suite: TestSuite | None,
    limit: int,
) -> list[dict]:
    targets = []
    if suite is not None:
        for item in suite.coverage.unvisited_summary():
            node_id = item["condition_id"]
            condition = item["condition"]
            if item["needs_true"]:
                attempts, misses = _target_attempts(suite, node_id, "TRUE")
                targets.append(
                    {
                        "node_id": node_id,
                        "polarity": "TRUE",
                        "kind": "mcdc",
                        "condition": condition,
                        "attempts": attempts,
                        "misses": misses,
                    }
                )
            if item["needs_false"]:
                attempts, misses = _target_attempts(suite, node_id, "FALSE")
                targets.append(
                    {
                        "node_id": node_id,
                        "polarity": "FALSE",
                        "kind": "mcdc",
                        "condition": condition,
                        "attempts": attempts,
                        "misses": misses,
                    }
                )
    else:
        for item in result.unvisited_mcdc:
            if not item.true_branch_visited:
                attempts, misses = _target_attempts(suite, item.node_id, "TRUE")
                targets.append(
                    {
                        "node_id": item.node_id,
                        "polarity": "TRUE",
                        "kind": "mcdc",
                        "condition": item.condition,
                        "attempts": attempts,
                        "misses": misses,
                    }
                )
            if not item.false_branch_visited:
                attempts, misses = _target_attempts(suite, item.node_id, "FALSE")
                targets.append(
                    {
                        "node_id": item.node_id,
                        "polarity": "FALSE",
                        "kind": "mcdc",
                        "condition": item.condition,
                        "attempts": attempts,
                        "misses": misses,
                    }
                )
    for item in result.unvisited_branches:
        if not item.true_visited:
            attempts, misses = _target_attempts(suite, item.node_id, "TRUE")
            targets.append(
                {
                    "node_id": item.node_id,
                    "polarity": "TRUE",
                    "kind": "branch",
                    "condition": item.condition,
                    "attempts": attempts,
                    "misses": misses,
                }
            )
        if not item.false_visited:
            attempts, misses = _target_attempts(suite, item.node_id, "FALSE")
            targets.append(
                {
                    "node_id": item.node_id,
                    "polarity": "FALSE",
                    "kind": "branch",
                    "condition": item.condition,
                    "attempts": attempts,
                    "misses": misses,
                }
            )
    mcdc_targets = [target for target in targets if target["kind"] == "mcdc"]
    if mcdc_targets:
        targets = mcdc_targets
    targets.sort(
        key=lambda target: (
            -_target_score(target, suite),
            target["misses"],
            target["attempts"],
            target["node_id"] is None,
            target["node_id"] or 0,
            target["polarity"] == "FALSE",
        )
    )
    return targets[: max(0, limit)]


def _first_blocker(result: TestResult) -> dict | None:
    if _target_hit(result) is not False:
        return None
    for entry in result.condition_trace:
        if entry.node_id == result.target_node_id:
            break
        if not entry.true_branch_visited:
            return {
                "node_id": entry.node_id,
                "condition": entry.condition,
                "needed": "TRUE",
            }
        if not entry.false_branch_visited:
            return {
                "node_id": entry.node_id,
                "condition": entry.condition,
                "needed": "FALSE",
            }
    return None


def _trace_chain(result: TestResult, limit: int = 12) -> list[dict]:
    chain = []
    for entry in result.condition_trace[:limit]:
        chain.append(
            {
                "node_id": entry.node_id,
                "condition": entry.condition,
                "true": entry.true_branch_visited,
                "false": entry.false_branch_visited,
            }
        )
    return chain


def _target_miss_context(result: TestResult, suite: TestSuite | None) -> dict | None:
    if _target_hit(result) is not False:
        return None
    attempts, misses = _target_attempts(
        suite, result.target_node_id, result.target_polarity or ""
    )
    if misses < 2:
        return None
    return {
        "attempts": attempts,
        "misses": misses,
        "trace_chain": _trace_chain(result),
        "repair_hint": "This target missed repeatedly. Do not retry the same input shape. First satisfy first_blocker, then drive the target polarity.",
    }


def _format_target_not_allowed(
    node_id: int | None,
    polarity: str | None,
    next_targets: list[dict],
) -> str:
    return json.dumps(
        {
            "status": "TARGET_NOT_ALLOWED",
            "target": {"node_id": node_id, "polarity": polarity},
            "message": "Use one of next_targets exactly; stale or missing targets are not executed.",
            "next_targets": next_targets,
        },
        separators=(",", ":"),
    )


def _compact_log(result: TestResult, limit: int = 500) -> dict | None:
    if not is_failure_status(result.status) or not result.execute_log:
        return None
    log = result.execute_log.strip()
    lower = log.lower()
    category = "failure"
    if "compile" in lower or "error:" in lower:
        category = "compile_error"
    elif "runtime" in lower or "segmentation" in lower or "exception" in lower:
        category = "runtime_error"
    return {"category": category, "excerpt": log[:limit]}


def _format_compact_summary(
    result: TestResult,
    suite: TestSuite | None,
    action_warnings: list[str] | None = None,
    top_targets: int = 3,
    next_targets: list[dict] | None = None,
) -> str:
    metrics = suite.coverage.metrics(suite.tests) if suite is not None else None
    payload = {
        "status": result.status,
        "test_name": result.test_name,
        "new_mcdc_pairs": result.new_mcdc_pairs_covered,
        "redundant": result.is_redundant,
        "target": None,
        "suite": {
            "iteration": suite.iteration_count if suite else result.iteration,
            "statement": (
                f"{metrics.covered_statements}/{metrics.total_statements}"
                if metrics is not None
                else f"{result.statement_coverage.visited}/{result.statement_coverage.total}"
            ),
            "branch": (
                f"{metrics.covered_branches}/{metrics.total_branches}"
                if metrics is not None
                else f"{result.branch_coverage.visited}/{result.branch_coverage.total}"
            ),
            "mcdc": (
                f"{metrics.covered_mcdc_pairs}/{metrics.total_mcdc_pairs}"
                if metrics is not None
                else f"{result.mcdc_coverage.visited}/{result.mcdc_coverage.total}"
            ),
        },
        "next_targets": next_targets if next_targets is not None else _next_targets(result, suite, top_targets),
    }
    if result.target_node_id is not None:
        payload["target"] = {
            "node_id": result.target_node_id,
            "polarity": result.target_polarity,
            "reason": result.target_reason,
            "hit": _target_hit(result),
            "first_blocker": _first_blocker(result),
            "miss_context": _target_miss_context(result, suite),
        }
    if action_warnings:
        payload["warnings"] = action_warnings
    log = _compact_log(result)
    if log:
        payload["log"] = log
    return json.dumps(payload, separators=(",", ":"))


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
