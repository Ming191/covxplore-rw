"""Core test-generation public API."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from rich.console import Console

from covxplore.config import Settings, get_settings
from covxplore.generation.stop_reasons import StopReason
from covxplore.prompts.registry import VARIANTS
from covxplore.types import TestSuite

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "ExperimentMetadata",
    "generate",
]


@dataclass(frozen=True)
class ExperimentMetadata:
    reasoning_technique: str
    provider_thinking: bool | None
    llm_provider: str
    model_id: str
    temperature: float
    max_tokens: int
    seed: int | None
    prompt_version: str
    context_version: str
    max_batches: int

    @classmethod
    def capture(
        cls,
        config: "GenerationConfig",
        settings: "Settings | None" = None,
    ) -> "ExperimentMetadata":
        s = settings or get_settings()
        provider = s.llm_provider.strip().lower()
        model_id = s.local_model if provider == "local" else s.deepseek_model
        return cls(
            reasoning_technique=config.prompt_variant,
            provider_thinking=s.llm_thinking,
            llm_provider=provider,
            model_id=model_id,
            temperature=s.llm_temperature,
            max_tokens=s.max_tokens,
            seed=s.llm_seed,
            prompt_version=s.prompt_version,
            context_version=config.context_version,
            max_batches=config.max_batches,
        )


@dataclass
class GenerationConfig:
    """Fully specify one generation run."""

    function_path: str
    prompt_variant: str
    context_version: str = field(default_factory=lambda: get_settings().context_version)
    max_batches: int = field(default_factory=lambda: get_settings().max_batches)
    redundant_streak_limit: int = field(
        default_factory=lambda: get_settings().redundant_streak_limit
    )
    fail_streak_limit: int = field(
        default_factory=lambda: get_settings().fail_streak_limit
    )
    run_id: str | None = None

    def __post_init__(self):
        if self.context_version not in {"v1", "v2"}:
            raise ValueError("context_version must be 'v1' or 'v2'")
        if self.prompt_variant not in VARIANTS:
            raise ValueError(
                f"Unknown reasoning technique {self.prompt_variant!r}. "
                f"Available: {', '.join(VARIANTS)}"
            )
        for field_name in ("max_batches", "redundant_streak_limit", "fail_streak_limit"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be > 0")
        if not self.run_id:
            base = self.function_path.split("::")[-1].split("(")[0] or "func"
            safe_func = re.sub(r"[^a-zA-Z0-9]", "_", base).strip("_") or "func"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{timestamp}_{safe_func}_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "function_path": self.function_path,
            "prompt_variant": self.prompt_variant,
            "context_version": self.context_version,
            "max_batches": self.max_batches,
            "redundant_streak_limit": self.redundant_streak_limit,
            "fail_streak_limit": self.fail_streak_limit,
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
    experiment: ExperimentMetadata | None = None
    _elapsed_sec: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._elapsed_sec = self.suite.elapsed_sec
        if self.experiment is None:
            self.experiment = ExperimentMetadata.capture(self.config)

    @property
    def redundancy_rate(self) -> float:
        return round(self.suite.redundancy_rate, 4)

    @property
    def total_input_tokens(self) -> int:
        return (
            self.crew_prompt_tokens
            if self.crew_prompt_tokens is not None
            else self.suite.total_input_tokens
        )

    @property
    def total_output_tokens(self) -> int:
        return (
            self.crew_completion_tokens
            if self.crew_completion_tokens is not None
            else self.suite.total_output_tokens
        )

    @property
    def elapsed_sec(self) -> float:
        return round(self._elapsed_sec, 2)

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
        metrics = self.suite.coverage.metrics(self.suite.tests)
        total_tokens = self.total_input_tokens + self.total_output_tokens
        return {
            "run_id": self.config.run_id,
            "function_path": self.config.function_path,
            "prompt_variant": self.config.prompt_variant,
            "context_version": self.config.context_version,
            "experiment": asdict(self.experiment),
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
                "total_tokens": total_tokens,
                "elapsed_sec": self.elapsed_sec,
                "batches_used": self.batches_used,
                "accepted_test_count": self.accepted_test_count,
                "rejected_candidate_count": self.rejected_candidate_count,
                "candidate_count": self.candidate_count,
                "tokens_per_batch": round(total_tokens / max(self.batches_used, 1), 2),
                "tokens_per_candidate": round(total_tokens / max(self.candidate_count, 1), 2),
            },
            "test_suite": [self._test_summary(test) for test in self.suite.tests],
            "rejected_tests": [self._test_summary(test) for test in self.suite.rejected_tests],
        }

    @staticmethod
    def _test_summary(test) -> dict:
        summary = {
            "test_name": test.test_name,
            "status": test.status,
            "is_redundant": test.is_redundant,
            "accepted_order": test.accepted_order,
            "elapsed_ms": round(test.elapsed_ms, 1),
            "token_input": test.token_input,
            "token_output": test.token_output,
            "execute_log": test.execute_log,
            "test_body": test.test_body,
            "new_structural_coverage": test.new_structural_coverage,
            "statement_coverage": test.statement_coverage.model_dump(),
            "branch_coverage": test.branch_coverage.model_dump(),
        }
        if test.trace_summary is not None:
            summary["trace_summary"] = test.trace_summary.model_dump(
                by_alias=True, exclude_none=True
            )
        return summary


def generate(config: GenerationConfig, console: Console | None = None) -> GenerationResult:
    from covxplore.flows.generation_flow import GenerationFlowRunner

    return GenerationFlowRunner(console=console or Console()).run(config)


def _print_result_summary(result: GenerationResult, console: Console | None = None) -> None:
    out = console or Console()
    metrics = result.to_summary_dict()["metrics"]
    out.print(
        f"  Statement: [bold]{metrics['statement_coverage_pct'] * 100:.0f}%[/]  |  "
        f"Branch: [bold]{metrics['branch_coverage_pct'] * 100:.0f}%[/]  |  "
        f"redundancy: {metrics['redundancy_rate'] * 100:.0f}%  |  "
        f"tokens: {metrics['total_tokens']:,}  |  "
        f"time: {metrics['elapsed_sec']:.1f}s  |  stop: {result.stop_reason}"
    )
