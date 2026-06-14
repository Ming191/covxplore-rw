"""Experiment data structures for the ablation study."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from covxplore.config import get_settings
from covxplore.models import TestSuite
from covxplore.status import TestStatus

StopReason = Literal[
    "max_iter",
    "coverage_target",
    "redundant_streak",
    "agent_done",
    "error",
]


@dataclass
class ExperimentConfig:
    """Fully specifies one ablation experiment run."""

    function_path: str
    prompt_variant: str
    max_iterations: int = field(default_factory=lambda: get_settings().max_iterations)
    max_tests: int | None = None
    mcdc_target: float = field(default_factory=lambda: get_settings().mcdc_target)
    redundant_streak_limit: int = field(
        default_factory=lambda: get_settings().redundant_streak_limit
    )
    run_id: str | None = None

    def __post_init__(self):
        if not self.run_id:
            import re
            import time

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
            "max_iterations": self.max_iterations,
            "max_tests": self.max_tests,
            "mcdc_target": self.mcdc_target,
        }


@dataclass
class ExperimentResult:
    """Outcome of one completed experiment run."""

    config: ExperimentConfig
    suite: TestSuite
    stop_reason: StopReason
    error_message: str | None = None

    crew_prompt_tokens: int | None = None
    crew_completion_tokens: int | None = None
    tracing_url: str | None = None  # CrewAI trace URL if tracing was enabled
    llm_interactions: list[dict] = field(
        default_factory=list
    )  # per-call thinking+answer log
    function_source: str | None = None
    function_context: str | None = None
    static_conditions: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Derived metrics (computed from suite)                               #
    # ------------------------------------------------------------------ #

    @property
    def final_mcdc_pct(self) -> float:
        return round(self.suite.mcdc_coverage_pct, 4)

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
    def iterations_used(self) -> int:
        return self.suite.iteration_count

    @property
    def covered_statements(self) -> int:
        return self.suite.covered_statements

    @property
    def covered_branches(self) -> int:
        return self.suite.covered_branches

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_summary_dict(self) -> dict:
        """Canonical summary JSON shape."""
        return {
            "run_id": self.config.run_id,
            "function_path": self.config.function_path,
            "prompt_variant": self.config.prompt_variant,
            "stop_reason": self.stop_reason,
            "error": self.error_message,
            "metrics": {
                "statement_coverage_pct": round(self.suite.statement_coverage_pct, 4),
                "branch_coverage_pct": round(self.suite.branch_coverage_pct, 4),
                "mcdc_coverage_pct": self.final_mcdc_pct,
                "covered_statements": self.covered_statements,
                "total_statements": self.suite.total_statements,
                "covered_branches": self.covered_branches,
                "total_branches": self.suite.total_branches,
                "covered_mcdc_pairs": len(self.suite.covered_keys),
                "total_mcdc_pairs": self.suite.total_mcdc_conditions,
                "redundancy_rate": self.redundancy_rate,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "elapsed_sec": self.elapsed_sec,
                "iterations_used": self.iterations_used,
            },
            "tracing_url": self.tracing_url,
            "llm_interactions": self.llm_interactions,
            "function_source": self.function_source,
            "function_context": self.function_context,
            "static_conditions": self.static_conditions,
            "total_conditions": len(self.static_conditions),
            "total_mcdc_pairs": self.suite.total_mcdc_conditions,
            "test_suite": [
                {
                    "test_name": t.test_name,
                    "status": t.status,
                    "new_mcdc_pairs_covered": t.new_mcdc_pairs_covered,
                    "is_redundant": t.is_redundant,
                    "iteration": t.iteration,
                    "elapsed_ms": round(t.elapsed_ms, 1),
                    "token_input": t.token_input,
                    "token_output": t.token_output,
                    "execute_log": t.execute_log,
                    "test_body": t.test_body,
                    "mcdc_coverage": t.mcdc_coverage.model_dump(),
                    "statement_coverage": t.statement_coverage.model_dump(),
                    "branch_coverage": t.branch_coverage.model_dump(),
                    "condition_trace": [
                        entry.model_dump() for entry in t.condition_trace
                    ],
                    "trace_summary": (
                        t.trace_summary.model_dump() if t.trace_summary else None
                    ),
                    "unvisited_mcdc": [
                        item.model_dump() for item in t.unvisited_mcdc
                    ],
                    "unvisited_statements": [
                        item.model_dump() for item in t.unvisited_statements
                    ],
                    "unvisited_branches": [
                        item.model_dump() for item in t.unvisited_branches
                    ],
                }
                for t in self.suite.tests
            ],
        }

    def to_flat_row(self) -> dict:
        """One-row dict for CSV / DataFrame export."""
        return {
            "run_id": self.config.run_id,
            "function_path": self.config.function_path,
            "prompt_variant": self.config.prompt_variant,
            "stop_reason": self.stop_reason,
            "statement_coverage_pct": round(self.suite.statement_coverage_pct, 4),
            "branch_coverage_pct": round(self.suite.branch_coverage_pct, 4),
            "mcdc_coverage_pct": self.final_mcdc_pct,
            "covered_statements": self.covered_statements,
            "total_statements": self.suite.total_statements,
            "covered_branches": self.covered_branches,
            "total_branches": self.suite.total_branches,
            "covered_mcdc_pairs": len(self.suite.covered_keys),
            "total_mcdc_pairs": self.suite.total_mcdc_conditions,
            "redundancy_rate": self.redundancy_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "elapsed_sec": self.elapsed_sec,
            "iterations_used": self.iterations_used,
            "num_tests": len(self.suite.tests),
            "num_passing": sum(
                1 for t in self.suite.tests if t.status == TestStatus.PASSED.value
            ),
            "num_redundant": sum(1 for t in self.suite.tests if t.is_redundant),
            "error": self.error_message or "",
        }
