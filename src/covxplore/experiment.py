"""Experiment data structures for the ablation study."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from covxplore.models import TestSuite
from covxplore.prompts.config import PromptConfig


StopReason = Literal["max_iter", "coverage_target", "agent_done", "error"]


@dataclass
class ExperimentConfig:
    """Fully specifies one ablation experiment run."""

    function_path: str
    prompt_variant: str                      # key in PromptRegistry.VARIANTS
    max_iterations: int = 15
    mcdc_target: float = 1.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "function_path": self.function_path,
            "prompt_variant": self.prompt_variant,
            "max_iterations": self.max_iterations,
            "mcdc_target": self.mcdc_target,
        }


@dataclass
class ExperimentResult:
    """Outcome of one completed experiment run."""

    config: ExperimentConfig
    suite: TestSuite
    stop_reason: StopReason
    error_message: str | None = None

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
        return self.suite.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.suite.total_output_tokens

    @property
    def elapsed_sec(self) -> float:
        return round(self.suite.elapsed_sec, 2)

    @property
    def iterations_used(self) -> int:
        return self.suite.iteration_count

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
                "mcdc_coverage_pct": self.final_mcdc_pct,
                "covered_mcdc_pairs": len(self.suite.covered_keys),
                "total_mcdc_pairs": self.suite.total_mcdc_conditions,
                "redundancy_rate": self.redundancy_rate,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "elapsed_sec": self.elapsed_sec,
                "iterations_used": self.iterations_used,
            },
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
            "mcdc_coverage_pct": self.final_mcdc_pct,
            "covered_mcdc_pairs": len(self.suite.covered_keys),
            "total_mcdc_pairs": self.suite.total_mcdc_conditions,
            "redundancy_rate": self.redundancy_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "elapsed_sec": self.elapsed_sec,
            "iterations_used": self.iterations_used,
            "num_tests": len(self.suite.tests),
            "num_passing": sum(1 for t in self.suite.tests if t.status == "PASSED"),
            "num_redundant": sum(1 for t in self.suite.tests if t.is_redundant),
            "error": self.error_message or "",
        }
