"""Core test-generation public API."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rich.console import Console

from covxplore.config import get_settings
from covxplore.generation.stop_reasons import StopReason
from covxplore.types import TestSuite

_console = Console()


@dataclass
class GenerationConfig:
    """Fully specifies one generation run."""

    function_path: str
    prompt_variant: str
    max_batches: int = field(default_factory=lambda: get_settings().max_batches)
    redundant_streak_limit: int = field(
        default_factory=lambda: get_settings().redundant_streak_limit
    )
    fail_streak_limit: int = field(
        default_factory=lambda: get_settings().fail_streak_limit
    )
    reasoning: bool = field(default_factory=lambda: get_settings().agent_reasoning)
    run_id: str | None = None

    def __post_init__(self):
        if not self.run_id:
            base = self.function_path.split("::")[-1].split("(")[0]
            if not base:
                base = "func"
            safe_func = re.sub(r"[^a-zA-Z0-9]", "_", base).strip("_")
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{ts}_{safe_func}"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "function_path": self.function_path,
            "prompt_variant": self.prompt_variant,
            "max_batches": self.max_batches,
            "redundant_streak_limit": self.redundant_streak_limit,
            "fail_streak_limit": self.fail_streak_limit,
            "reasoning": self.reasoning,
        }


@dataclass
class GenerationResult:
    """Outcome of one completed generation run."""

    config: GenerationConfig
    suite: TestSuite
    stop_reason: StopReason
    error_message: str | None = None

    crew_prompt_tokens: int | None = None
    crew_completion_tokens: int | None = None
    tracing_url: str | None = None
    llm_interactions: list[dict] = field(default_factory=list)

    @property
    def redundancy_rate(self) -> float:
        return round(self.suite.redundancy_rate, 4)

    @property
    def total_input_tokens(self) -> int:
        if self.crew_prompt_tokens is not None:
            return self.crew_prompt_tokens
        return self.suite.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        if self.crew_completion_tokens is not None:
            return self.crew_completion_tokens
        return self.suite.total_output_tokens

    @property
    def elapsed_sec(self) -> float:
        return round(self.suite.elapsed_sec, 2)

    @property
    def batches_used(self) -> int:
        return self.suite.batch_count

    @property
    def accepted_test_count(self) -> int:
        return len(self.suite.tests)

    @property
    def rejected_candidate_count(self) -> int:
        return len(self.suite.rejected_tests)

    @property
    def candidate_count(self) -> int:
        return self.accepted_test_count + self.rejected_candidate_count

    @property
    def covered_statements(self) -> int:
        return self.suite.coverage.metrics(self.suite.tests).covered_statements

    @property
    def covered_branches(self) -> int:
        return self.suite.coverage.metrics(self.suite.tests).covered_branches

    def to_summary_dict(self) -> dict:
        """Canonical summary JSON shape."""
        metrics = self.suite.coverage.metrics(self.suite.tests)
        return {
            "run_id": self.config.run_id,
            "function_path": self.config.function_path,
            "prompt_variant": self.config.prompt_variant,
            "stop_reason": self.stop_reason,
            "error": self.error_message,
            "metrics": {
                "statement_coverage_pct": round(metrics.statement_pct, 4),
                "branch_coverage_pct": round(metrics.branch_pct, 4),
                "covered_statements": self.covered_statements,
                "total_statements": metrics.total_statements,
                "covered_branches": self.covered_branches,
                "total_branches": metrics.total_branches,
                "redundancy_rate": self.redundancy_rate,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "elapsed_sec": self.elapsed_sec,
                "batches_used": self.batches_used,
                "accepted_test_count": self.accepted_test_count,
                "rejected_candidate_count": self.rejected_candidate_count,
                "candidate_count": self.candidate_count,
                "tokens_per_batch": round((self.total_input_tokens + self.total_output_tokens) / max(self.batches_used, 1), 2),
                "tokens_per_candidate": round((self.total_input_tokens + self.total_output_tokens) / max(self.candidate_count, 1), 2),
            },
            "tracing_url": self.tracing_url,
            "llm_interactions": self.llm_interactions,
            "test_suite": [self._test_summary(t) for t in self.suite.tests],
            "rejected_tests": [self._test_summary(t) for t in self.suite.rejected_tests],
        }

    @staticmethod
    def _test_summary(t) -> dict:
        return {
            "test_name": t.test_name,
            "status": t.status,
            "is_redundant": t.is_redundant,
            "accepted_order": t.accepted_order,
            "elapsed_ms": round(t.elapsed_ms, 1),
            "token_input": t.token_input,
            "token_output": t.token_output,
            "execute_log": t.execute_log,
            "target_node_id": t.target_node_id,
            "target_polarity": t.target_polarity,
            "target_reason": t.target_reason,
            "test_body": t.test_body,
            "new_structural_coverage": t.new_structural_coverage,
            "statement_coverage": t.statement_coverage.model_dump(),
            "branch_coverage": t.branch_coverage.model_dump(),
        }


def generate(config: GenerationConfig) -> GenerationResult:
    """Execute a single generation run end-to-end through CrewAI Flow."""
    from covxplore.flows.generation_flow import GenerationFlowRunner

    return GenerationFlowRunner(console=_console).run(config)


def _print_result_summary(r: GenerationResult) -> None:
    m = r.to_summary_dict()["metrics"]
    _console.print(
        f"  Statement: [bold]{m['statement_coverage_pct'] * 100:.0f}%[/]  |  "
        f"Branch: [bold]{m['branch_coverage_pct'] * 100:.0f}%[/]  |  "
        f"redundancy: {m['redundancy_rate'] * 100:.0f}%  |  "
        f"tokens: {m['total_tokens']:,}  |  "
        f"time: {m['elapsed_sec']:.1f}s  |  "
        f"stop: {r.stop_reason}"
    )
