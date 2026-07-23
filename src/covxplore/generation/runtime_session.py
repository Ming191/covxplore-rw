from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel, Field

from covxplore.coverage.state import CoverageState
from covxplore.generation.stop_reasons import StopPolicy, StopReason
from covxplore.generation.tokens import TokenLedger
from covxplore.observability import extract_trace_id, flush_observability, get_trace_url
from covxplore.tools.execute_testcase import RunContext
from covxplore.types import (
    TestResult,
    TestSuite,
    UnvisitedBranch,
    UnvisitedStatement,
)


class GenerationFlowState(BaseModel):
    config: dict = Field(default_factory=dict)
    suite: dict = Field(default_factory=dict)
    static_prompt: dict = Field(default_factory=dict)
    stop_reason: StopReason | None = None
    error_message: str | None = None
    token_ledger: dict = Field(default_factory=dict)
    trace_urls: list[str] = Field(default_factory=list)
    llm_interactions: list[dict] = Field(default_factory=list)


class TestSuiteCodec:
    VERSION = 3

    def dump_suite(self, suite: TestSuite) -> dict:
        return {
            "version": self.VERSION,
            "function_path": suite.function_path,
            "batch_count": suite.batch_count,
            "fail_streak": suite.fail_streak,
            "elapsed_sec": suite.elapsed_sec,
            "tests": [self._dump_result(test) for test in suite.tests],
            "rejected_tests": [self._dump_result(test) for test in suite.rejected_tests],
            "coverage": self._dump_coverage(suite.coverage),
        }

    def load_suite(self, data: Mapping[str, Any]) -> TestSuite:
        if int(data.get("version", 0)) != self.VERSION:
            raise ValueError(f"Unsupported TestSuite snapshot version: {data.get('version')}")
        function_path = str(data.get("function_path") or "")
        if not function_path:
            raise ValueError("TestSuite snapshot missing function_path")

        suite = TestSuite(function_path=function_path)
        suite.batch_count = self._non_negative_int(data.get("batch_count"), "batch_count")
        suite.fail_streak = self._non_negative_int(data.get("fail_streak"), "fail_streak")
        suite.started_at = time.monotonic() - max(0.0, float(data.get("elapsed_sec") or 0.0))
        suite.tests = [self._load_result(item) for item in data.get("tests") or []]
        suite.rejected_tests = [self._load_result(item) for item in data.get("rejected_tests") or []]
        suite.coverage = self._load_coverage(data.get("coverage") or {})

        max_order = max((test.accepted_order for test in suite.tests), default=0)
        if max_order > len(suite.tests):
            raise ValueError("TestSuite snapshot has accepted_order beyond accepted tests")
        return suite

    @staticmethod
    def _dump_result(result: TestResult) -> dict:
        return result.model_dump(mode="json")

    @staticmethod
    def _load_result(data: Mapping[str, Any]) -> TestResult:
        return TestResult.model_validate(data)

    def _dump_coverage(self, coverage: CoverageState) -> dict:
        return {
            "consecutive_redundant": coverage.consecutive_redundant,
            "_cumulative_uncovered_stmt_ids": (
                None
                if coverage._cumulative_uncovered_stmt_ids is None
                else sorted(coverage._cumulative_uncovered_stmt_ids)
            ),
            "_cumulative_uncovered_branch_keys": (
                None
                if coverage._cumulative_uncovered_branch_keys is None
                else [
                    {"node_id": node_id, "polarity": polarity}
                    for node_id, polarity in sorted(coverage._cumulative_uncovered_branch_keys)
                ]
            ),
            "_stmt_node_info": [
                statement.model_dump(mode="json")
                for _, statement in sorted(coverage._stmt_node_info.items())
            ],
            "_branch_node_info": [
                branch.model_dump(mode="json")
                for _, branch in sorted(coverage._branch_node_info.items())
            ],
            "_total_statements": coverage._total_statements,
            "_total_branches": coverage._total_branches,
        }

    def _load_coverage(self, data: Mapping[str, Any]) -> CoverageState:
        coverage = CoverageState()
        coverage.consecutive_redundant = self._non_negative_int(data.get("consecutive_redundant"), "consecutive_redundant")
        stmt_ids = data.get("_cumulative_uncovered_stmt_ids")
        coverage._cumulative_uncovered_stmt_ids = None if stmt_ids is None else {int(node_id) for node_id in stmt_ids}

        branch_keys = data.get("_cumulative_uncovered_branch_keys")
        coverage._cumulative_uncovered_branch_keys = None if branch_keys is None else {
            (int(item["node_id"]), bool(item["polarity"])) for item in branch_keys
        }

        coverage._stmt_node_info = {}
        for item in data.get("_stmt_node_info") or []:
            statement = UnvisitedStatement.model_validate(item)
            if statement.node_id is None:
                raise ValueError("_stmt_node_info entry missing node_id")
            coverage._stmt_node_info[statement.node_id] = statement

        coverage._branch_node_info = {}
        for item in data.get("_branch_node_info") or []:
            branch = UnvisitedBranch.model_validate(item)
            if branch.node_id is None:
                raise ValueError("_branch_node_info entry missing node_id")
            coverage._branch_node_info[branch.node_id] = branch

        coverage._total_statements = self._non_negative_int(data.get("_total_statements"), "_total_statements")
        coverage._total_branches = self._non_negative_int(data.get("_total_branches"), "_total_branches")
        return coverage

    @staticmethod
    def _non_negative_int(value: Any, field_name: str) -> int:
        result = int(value or 0)
        if result < 0:
            raise ValueError(f"{field_name} must be >= 0")
        return result


@dataclass
class GenerationRuntimeSession:
    config: Any
    run_context: RunContext = field(default_factory=RunContext)
    suite_codec: TestSuiteCodec = field(default_factory=TestSuiteCodec)
    token_ledger: TokenLedger = field(default_factory=TokenLedger)
    trace_urls: list[str] = field(default_factory=list)
    stop_policy: StopPolicy = field(init=False)

    def __post_init__(self) -> None:
        self.stop_policy = StopPolicy.from_config(self.config)

    def open(self, console=None) -> TestSuite:
        suite = self.run_context.reset_suite(self.config.function_path, self.config.run_id)
        return suite

    def active_suite(self) -> TestSuite:
        suite = self.run_context.get_suite(self.config.run_id)
        if suite is None:
            raise RuntimeError("Generation runtime session has no active suite")
        return suite

    def restore_suite(self, snapshot: Mapping[str, Any]) -> TestSuite:
        suite = self.suite_codec.load_suite(snapshot)
        self.run_context._suites[self.config.run_id] = suite
        self.run_context._active_run_id = self.config.run_id
        return suite

    def restore_ledger(self, snapshot: Mapping[str, Any]) -> None:
        self.token_ledger = TokenLedger.from_dict(dict(snapshot or {}))

    def record_crew_run(self, crew_inst, tracing_url: str | None) -> None:
        self.token_ledger.record_crew(crew_inst)
        resolved_url = tracing_url or get_trace_url()
        if resolved_url:
            self.trace_urls.append(resolved_url)
        flush_observability()
        self.token_ledger.record_trace(extract_trace_id(resolved_url))

    def finalize_tokens(self) -> None:
        self.token_ledger.finalize_suite(self.active_suite())

    def snapshot(
        self,
        *,
        stop_reason: StopReason | None = None,
        error_message: str | None = None,
        static_prompt: dict | None = None,
        llm_interactions: list[dict] | None = None,
    ) -> GenerationFlowState:
        return GenerationFlowState(
            config=self.config.to_dict(),
            suite=self.suite_codec.dump_suite(self.active_suite()),
            static_prompt=static_prompt or {},
            stop_reason=stop_reason,
            error_message=error_message,
            token_ledger=self.token_ledger.to_dict(),
            trace_urls=list(self.trace_urls),
            llm_interactions=llm_interactions or [],
        )

    def cleanup(self) -> None:
        self.run_context.cleanup_suite(self.config.run_id)
