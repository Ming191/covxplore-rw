from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel, Field

from covxplore.coverage.state import CoverageState, _non_negative_int
from covxplore.generation.stop_reasons import StopPolicy, StopReason
from covxplore.generation.tokens import TokenLedger
from covxplore.tools.execute_testcase import RunContext
from covxplore.types import TestResult, TestSuite


if TYPE_CHECKING:
    from covxplore.generator import GenerationConfig


class StaticPromptSnapshot(BaseModel):
    """Typed snapshot of the static prompt data carried through flow state."""

    context_text: str = ""
    source_text: str = ""
    total_statements: int | None = None
    total_branches: int | None = None


class GenerationFlowState(BaseModel):
    config: dict = Field(default_factory=dict)
    suite: dict = Field(default_factory=dict)
    static_prompt: StaticPromptSnapshot = Field(default_factory=StaticPromptSnapshot)
    stop_reason: StopReason | None = None
    error_message: str | None = None
    token_ledger: dict = Field(default_factory=dict)

    execution_feedback: str = ""


class TestSuiteCodec:
    VERSION = 3

    def dump_suite(self, suite: TestSuite) -> dict:
        return {
            "version": self.VERSION,
            "function_path": suite.function_path,
            "batch_count": suite.batch_count,
            "fail_streak": suite.fail_streak,
            "elapsed_sec": suite.elapsed_sec,
            "tests": [test.model_dump(mode="json") for test in suite.tests],
            "rejected_tests": [test.model_dump(mode="json") for test in suite.rejected_tests],
            "coverage": self._dump_coverage(suite.coverage),
        }

    def load_suite(self, data: Mapping[str, Any]) -> TestSuite:
        if int(data.get("version", 0)) != self.VERSION:
            raise ValueError(f"Unsupported TestSuite snapshot version: {data.get('version')}")
        function_path = str(data.get("function_path") or "")
        if not function_path:
            raise ValueError("TestSuite snapshot missing function_path")

        suite = TestSuite(function_path=function_path)
        suite.batch_count = _non_negative_int(data.get("batch_count"), "batch_count")
        suite.fail_streak = _non_negative_int(data.get("fail_streak"), "fail_streak")
        suite.started_at = time.monotonic() - max(0.0, float(data.get("elapsed_sec") or 0.0))
        suite.tests = [TestResult.model_validate(item) for item in data.get("tests") or []]
        suite.rejected_tests = [
            TestResult.model_validate(item) for item in data.get("rejected_tests") or []
        ]
        suite.coverage = self._load_coverage(data.get("coverage") or {})

        max_order = max((test.accepted_order for test in suite.tests), default=0)
        if max_order > len(suite.tests):
            raise ValueError("TestSuite snapshot has accepted_order beyond accepted tests")
        return suite

    @staticmethod
    def _dump_coverage(coverage: CoverageState) -> dict:
        return coverage.dump()

    def _load_coverage(self, data: Mapping[str, Any]) -> CoverageState:
        return CoverageState.load(data)


@dataclass
class GenerationRuntimeSession:
    config: GenerationConfig
    run_context: RunContext = field(default_factory=RunContext)
    suite_codec: TestSuiteCodec = field(default_factory=TestSuiteCodec)
    token_ledger: TokenLedger = field(default_factory=TokenLedger)
    stop_policy: StopPolicy = field(init=False)

    def __post_init__(self) -> None:
        self.stop_policy = StopPolicy.from_config(self.config)

    def open(self) -> TestSuite:
        return self.run_context.reset_suite(self.config.function_path, self.config.run_id)

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

    def record_crew_run(self, crew_inst) -> None:
        self.token_ledger.record_crew(crew_inst)

    def finalize_tokens(self) -> None:
        self.token_ledger.finalize_suite(self.active_suite())

    def snapshot(
        self,
        *,
        stop_reason: StopReason | None = None,
        error_message: str | None = None,
        static_prompt: StaticPromptSnapshot | None = None,
        execution_feedback: str = "",
    ) -> GenerationFlowState:
        return GenerationFlowState(
            config=self.config.to_dict(),
            suite=self.suite_codec.dump_suite(self.active_suite()),
            static_prompt=static_prompt or StaticPromptSnapshot(),
            stop_reason=stop_reason,
            error_message=error_message,
            token_ledger=self.token_ledger.to_dict(),
            execution_feedback=execution_feedback,
        )

    def cleanup(self) -> None:
        self.run_context.cleanup_suite(self.config.run_id)
