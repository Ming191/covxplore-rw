from __future__ import annotations

import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path

from covxplore.api_client import AkaUTClient
from covxplore.hybrid.budget import RunBudget
from covxplore.hybrid.coverage import ExactCoverageState
from covxplore.hybrid.artifacts import SUMMARY_COLUMNS, result_row, write_artifacts
from covxplore.hybrid.models import RouteDecision, TestAttemptRecord
from covxplore.hybrid.runner import (
    HybridGenerationConfig,
    HybridGenerationResult,
    HybridGenerationRunner,
)
from covxplore.hybrid.selector import SuccessfulSeed


EXPERIMENT_VARIANTS = (
    "llm",
    "symbolic",
    "symbolic_then_llm",
    "symbolic_llm_union",
    "hybrid_rule",
    "no_nearest_seed",
    "no_first_divergence_repair",
    "no_symbolic_partial",
    "hybrid_frozen",
)


@dataclass(frozen=True)
class ExperimentRecord:
    variant: str
    repeat: int
    row: dict[str, object]


class HybridExperimentRunner:
    def __init__(self, client: AkaUTClient) -> None:
        self.client = client

    def run_matrix(
        self,
        base: HybridGenerationConfig,
        *,
        variants: list[str],
        repeat: int,
        out_dir: Path,
    ) -> Path:
        unknown = sorted(set(variants) - set(EXPERIMENT_VARIANTS))
        if unknown:
            raise ValueError(f"unknown experiment variants: {unknown}")
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        records: list[ExperimentRecord] = []
        for repetition in range(1, repeat + 1):
            for variant in variants:
                config = replace(
                    base,
                    run_id=time.strftime("%Y%m%d_%H%M%S")
                    + f"_{variant}_{repetition:02d}",
                )
                result = self._run_variant(config, variant)
                write_artifacts(result, out_dir / variant)
                records.append(ExperimentRecord(variant, repetition, result_row(result)))
        path = out_dir / "experiment_summary.csv"
        self._write_summary(path, records)
        return path

    def _variant(
        self, base: HybridGenerationConfig, variant: str
    ) -> HybridGenerationConfig:
        if variant == "llm":
            return replace(base, strategy="llm", scheduler_mode="rule")
        if variant == "symbolic":
            return replace(base, strategy="symbolic", scheduler_mode="rule")
        if variant == "hybrid_rule":
            return replace(base, strategy="hybrid", scheduler_mode="rule")
        if variant == "hybrid_frozen":
            if base.policy_path is None:
                raise ValueError("hybrid_frozen requires --policy")
            return replace(base, strategy="hybrid", scheduler_mode="frozen")
        if variant == "no_nearest_seed":
            return replace(
                base,
                strategy="hybrid",
                scheduler_mode="rule",
                use_nearest_seed=False,
            )
        if variant == "no_first_divergence_repair":
            return replace(
                base,
                strategy="hybrid",
                scheduler_mode="rule",
                use_divergence_repair=False,
            )
        if variant == "no_symbolic_partial":
            return replace(
                base,
                strategy="hybrid",
                scheduler_mode="rule",
                use_symbolic_partial=False,
            )
        raise ValueError(f"unsupported experiment variant: {variant}")

    def _run_variant(
        self, base: HybridGenerationConfig, variant: str
    ) -> HybridGenerationResult:
        if variant == "symbolic_then_llm":
            return self._sequential(base)
        if variant == "symbolic_llm_union":
            return self._union(base)
        return HybridGenerationRunner(client=self.client).run(self._variant(base, variant))

    def _sequential(self, base: HybridGenerationConfig) -> HybridGenerationResult:
        symbolic_config = replace(
            base,
            strategy="symbolic",
            scheduler_mode="rule",
            max_llm_calls=0,
            run_id=base.run_id + "_symbolic",
        )
        symbolic = HybridGenerationRunner(client=self.client).run(symbolic_config)
        remaining_seconds = base.wall_time_minutes * 60 - symbolic.budget.elapsed_sec
        remaining_executions = (
            base.max_test_executions - symbolic.budget.test_executions
        )
        if symbolic.state.reached(base.statement_target, base.branch_target):
            return self._combine(base, symbolic, None, mode="sequential")
        if remaining_seconds <= 0 or remaining_executions <= 0:
            return self._combine(base, symbolic, None, mode="sequential")

        llm_config = replace(
            base,
            strategy="llm",
            scheduler_mode="rule",
            wall_time_minutes=remaining_seconds / 60,
            max_symbolic_attempts=0,
            max_test_executions=remaining_executions,
            run_id=base.run_id + "_llm",
        )
        llm = HybridGenerationRunner(client=self.client).run(
            llm_config,
            initial_statement_keys=set(symbolic.state.covered_statement_keys),
            initial_branch_keys=set(symbolic.state.covered_branch_keys),
            initial_seeds=self._successful_seeds(symbolic),
        )
        return self._combine(base, symbolic, llm, mode="sequential")

    def _union(self, base: HybridGenerationConfig) -> HybridGenerationResult:
        symbolic = HybridGenerationRunner(client=self.client).run(
            replace(
                base,
                strategy="symbolic",
                scheduler_mode="rule",
                max_llm_calls=0,
                run_id=base.run_id + "_symbolic",
            )
        )
        llm = HybridGenerationRunner(client=self.client).run(
            replace(
                base,
                strategy="llm",
                scheduler_mode="rule",
                max_symbolic_attempts=0,
                run_id=base.run_id + "_llm",
            )
        )
        return self._combine(base, symbolic, llm, mode="union")

    @staticmethod
    def _successful_seeds(result: HybridGenerationResult) -> list[SuccessfulSeed]:
        return [
            SuccessfulSeed(
                intent_id=attempt.intent.intent_id if attempt.intent else None,
                bindings=tuple(attempt.intent.bindings) if attempt.intent else (),
                result=attempt.execution,
            )
            for attempt in result.attempts
            if attempt.execution is not None and attempt.execution.passed
        ]

    @staticmethod
    def _combine(
        base: HybridGenerationConfig,
        first: HybridGenerationResult,
        second: HybridGenerationResult | None,
        *,
        mode: str,
    ) -> HybridGenerationResult:
        if second is not None and second.coverage_model.model_hash != first.coverage_model.model_hash:
            raise RuntimeError("coverage model changed between experiment phases")

        attempts: list[TestAttemptRecord] = []
        decisions: list[RouteDecision] = []
        elapsed_offset_ms = 0.0
        for phase in (first, second):
            if phase is None:
                continue
            iteration_offset = len(attempts)
            for attempt in phase.attempts:
                attempts.append(
                    attempt.model_copy(
                        deep=True,
                        update={
                            "iteration": attempt.iteration + iteration_offset,
                            "run_elapsed_ms": attempt.run_elapsed_ms + elapsed_offset_ms,
                        },
                    )
                )
            for decision in phase.route_decisions:
                decisions.append(
                    decision.model_copy(
                        deep=True,
                        update={"iteration": decision.iteration + iteration_offset},
                    )
                )
            elapsed_offset_ms += phase.budget.elapsed_sec * 1000

        state = ExactCoverageState(first.coverage_model)
        for attempt in attempts:
            if attempt.execution is not None:
                delta = state.add(attempt.execution)
                attempt.new_statement_keys = sorted(delta.new_statement_keys)
                attempt.new_branch_keys = sorted(delta.new_branch_keys)
                attempt.redundant = delta.redundant
            attempt.cumulative_statement_cov = state.statement_fraction
            attempt.cumulative_branch_cov = state.branch_fraction

        elapsed = first.budget.elapsed_sec + (
            second.budget.elapsed_sec if second is not None else 0.0
        )
        union_multiplier = 2 if mode == "union" else 1
        budget = RunBudget(
            wall_time_sec=base.wall_time_minutes * 60 * union_multiplier,
            max_llm_calls=base.max_llm_calls * union_multiplier,
            max_symbolic_attempts=base.max_symbolic_attempts * union_multiplier,
            max_test_executions=base.max_test_executions * union_multiplier,
            started_at=0.0,
            finished_at=elapsed,
            llm_calls=first.budget.llm_calls + (second.budget.llm_calls if second else 0),
            symbolic_attempts=first.budget.symbolic_attempts
            + (second.budget.symbolic_attempts if second else 0),
            test_executions=first.budget.test_executions
            + (second.budget.test_executions if second else 0),
        )
        reached = state.reached(base.statement_target, base.branch_target)
        last = second or first
        stop_reason = "coverage_target" if reached else last.stop_reason
        errors = [item.error for item in (first, second) if item is not None and item.error]
        return HybridGenerationResult(
            config=base,
            coverage_model=first.coverage_model,
            state=state,
            stop_reason=stop_reason,
            attempts=attempts,
            route_decisions=decisions,
            llm_interactions=[
                {"phase": mode, **interaction}
                for result in (first, second)
                if result is not None
                for interaction in result.llm_interactions
            ],
            budget=budget,
            error="; ".join(errors) or None,
            policy_metadata=last.policy_metadata,
        )

    @staticmethod
    def _write_summary(path: Path, records: list[ExperimentRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["experiment", "repeat", *SUMMARY_COLUMNS]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {"experiment": record.variant, "repeat": record.repeat, **record.row}
                )
