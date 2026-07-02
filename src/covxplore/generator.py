"""Core test-generation engine.

One call to :func:`generate` runs a single (function, prompt-variant) test
generation end-to-end: prefetch static CFG/context/source, kick off the crew,
deduce a stop reason from the achieved coverage, and reconcile token usage.

This module is app-only and carries no ablation/pipeline dependencies.
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass, field

from rich.console import Console

from covxplore.config import get_settings
from covxplore.generation.prompt_context import (
    fetch_static_prompt_data,
    seed_suite_conditions,
)
from covxplore.generation.stop_reasons import (
    StopReason,
    coverage_target_reached,
    infer_stop,
)
from covxplore.generation.tokens import choose_run_token_totals, reconcile_suite_tokens
from covxplore.observability import (
    extract_trace_id,
    fetch_trace_token_totals,
    flush_observability,
    get_trace_url,
    init_observability,
    trace_observation,
)
from covxplore.prompts.builder import PromptBuilder
from covxplore.prompts.registry import get_variant
from covxplore.types import TestSuite

_console = Console()

@dataclass
class GenerationConfig:
    """Fully specifies one generation run."""

    function_path: str
    prompt_variant: str
    max_iterations: int = field(default_factory=lambda: get_settings().max_iterations)
    mcdc_target: float = field(default_factory=lambda: get_settings().mcdc_target)
    redundant_streak_limit: int = field(
        default_factory=lambda: get_settings().redundant_streak_limit
    )
    fail_streak_limit: int = field(
        default_factory=lambda: get_settings().fail_streak_limit
    )
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
            "max_iterations": self.max_iterations,
            "mcdc_target": self.mcdc_target,
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
    tracing_url: str | None = None  # Langfuse trace URL if tracing was enabled
    llm_interactions: list[dict] = field(default_factory=list)  # legacy JSON key

    # ------------------------------------------------------------------ #
    # Derived metrics (computed from suite)                               #
    # ------------------------------------------------------------------ #

    @property
    def final_mcdc_pct(self) -> float:
        return round(self.suite.coverage.metrics(self.suite.tests).mcdc_pct, 4)

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
        return self.suite.coverage.metrics(self.suite.tests).covered_statements

    @property
    def covered_branches(self) -> int:
        return self.suite.coverage.metrics(self.suite.tests).covered_branches

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #

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
                "mcdc_coverage_pct": self.final_mcdc_pct,
                "covered_statements": self.covered_statements,
                "total_statements": metrics.total_statements,
                "covered_branches": self.covered_branches,
                "total_branches": metrics.total_branches,
                "covered_mcdc_pairs": metrics.covered_mcdc_pairs,
                "total_mcdc_pairs": metrics.total_mcdc_pairs,
                "redundancy_rate": self.redundancy_rate,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "elapsed_sec": self.elapsed_sec,
                "iterations_used": self.iterations_used,
            },
            "tracing_url": self.tracing_url,
            "llm_interactions": self.llm_interactions,
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
                }
                for t in self.suite.tests
            ],
        }


def generate(config: GenerationConfig) -> GenerationResult:
    """Execute a single generation run end-to-end."""
    from covxplore.crew import build_crew
    from covxplore.tools.execute_testcase import HardStop, RunContext

    cfg = get_settings()
    assert config.run_id is not None
    prompt_config = get_variant(config.prompt_variant)

    _console.rule(
        f"[bold cyan]Run {config.run_id} | variant={config.prompt_variant!r}"
    )

    # Create a per-run context for explicit run_id-keyed suite state.
    run_context = RunContext()

    suite: TestSuite = run_context.reset_suite(config.function_path, config.run_id)
    seed_suite_conditions(suite, _console)

    if not suite.coverage.has_mcdc:
        _console.print(
            "[yellow]No MC/DC conditions found — running for statement/branch coverage.[/]"
        )

    stop_reason: StopReason = "agent_done"
    error_msg = None
    crew_prompt_tokens = None
    crew_completion_tokens = None
    tracing_url = None
    crew_inst = None

    try:
        init_observability()
        static_prompt_data = fetch_static_prompt_data(config.function_path)
        crew_inst, builder = build_crew(
            prompt_config=prompt_config,
            max_iterations=config.max_iterations,
            run_context=run_context,
        )
        assert isinstance(builder, PromptBuilder)
        inputs = {
            "agent_backstory": builder.system_prompt(),
            "task_description": builder.task_description(
                function_path=config.function_path,
                suite=suite,
                remaining_iterations=config.max_iterations,
                static_conditions_text=static_prompt_data.conditions_text,
                static_context_text=static_prompt_data.context_text,
                static_source_text=static_prompt_data.source_text,
            ),
        }
        crew_obj = crew_inst.crew()
        with trace_observation(
            "covxplore.generate",
            run_id=config.run_id,
            function_path=config.function_path,
            prompt_variant=config.prompt_variant,
        ) as langfuse_url:
            tracing_url = langfuse_url
            crew_obj.kickoff(inputs=inputs)

    except HardStop as e:
        stop_reason = e.reason  # type: ignore[assignment]
        error_msg = None
        _console.print(f"[green]Hard stop ({e.reason}): {e}[/]")

    except BaseException as e:
        stop_reason = "error"
        error_msg = f"{type(e).__name__}: {e}"
        _console.print(f"[red]Error: {error_msg}[/]")
        traceback.print_exc()

    finally:
        token_totals = choose_run_token_totals(crew_inst)
        if tracing_url is None:
            tracing_url = get_trace_url()
        flush_observability()
        if token_totals.total == 0:
            token_totals = fetch_trace_token_totals(extract_trace_id(tracing_url))
        if token_totals.total > 0:
            crew_prompt_tokens = token_totals.prompt
            crew_completion_tokens = token_totals.completion
        final_suite = run_context.get_suite(config.run_id) or suite
        reconcile_suite_tokens(final_suite, token_totals)

    # Deduces final stop reason if the agent finished without a HardStop.
    if stop_reason != "agent_done":
        pass
    else:
        stop_reason = infer_stop(final_suite, config)
        if stop_reason == "max_iter":
            error_msg = None
            _console.print(
                f"[yellow]Stopped gracefully: max_iter after {final_suite.iteration_count} iterations[/]"
            )
        elif stop_reason == "redundant_streak":
            error_msg = None
            _console.print(
                f"[yellow]Stopped: {config.redundant_streak_limit} consecutive redundant tests[/]"
            )
        elif stop_reason == "coverage_target":
            error_msg = None
            _console.print("[green]Stopped gracefully: reached coverage target[/]")
        elif stop_reason == "fail_streak":
            error_msg = None
            fail_limit: int = getattr(config, "fail_streak_limit", 3)
            _console.print(
                f"[red]Hard stop: {fail_limit} consecutive failing tool iterations[/]"
            )

    result = GenerationResult(
        config=config,
        suite=final_suite,
        stop_reason=stop_reason,
        error_message=error_msg,
        crew_prompt_tokens=crew_prompt_tokens,
        crew_completion_tokens=crew_completion_tokens,
        tracing_url=tracing_url,
    )

    _print_result_summary(result)
    run_context.cleanup_suite(config.run_id)
    return result


_coverage_target_reached = coverage_target_reached


def _print_result_summary(r: GenerationResult) -> None:
    m = r.to_summary_dict()["metrics"]
    _console.print(
        f"  MC/DC: [bold]{m['mcdc_coverage_pct'] * 100:.0f}%[/] "
        f"({m['covered_mcdc_pairs']}/{m['total_mcdc_pairs']})  |  "
        f"redundancy: {m['redundancy_rate'] * 100:.0f}%  |  "
        f"tokens: {m['total_tokens']:,}  |  "
        f"time: {m['elapsed_sec']:.1f}s  |  "
        f"stop: {r.stop_reason}"
    )
