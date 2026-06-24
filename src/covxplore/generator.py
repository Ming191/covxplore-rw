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
from typing import Literal

from rich.console import Console

from covxplore.config import get_settings
from covxplore.crew import build_crew
from covxplore.driver import create_executor
from covxplore.llm_logger import LLMInteractionLogger
from covxplore.prompts.builder import PromptBuilder
from covxplore.prompts.registry import get_variant
from covxplore.status import TestStatus
from covxplore.tools.execute_testcase import HardStop, RunContext
from covxplore.types import TestSuite

_console = Console()

StopReason = Literal[
    "max_iter",
    "coverage_target",
    "redundant_streak",
    "agent_done",
    "error",
]


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
    run_id: str | None = None
    executor_backend: str = "akaut"
    gtest_source_root: str | None = None
    gtest_project_sources: list[str] = field(default_factory=list)
    gtest_extra_compile_flags: list[str] = field(default_factory=list)
    gtest_coverage_backend: str = "gcov"
    gtest_compiler: str | None = None

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
            "executor_backend": self.executor_backend,
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
    tracing_url: str | None = None  # CrewAI trace URL if tracing was enabled
    llm_interactions: list[dict] = field(
        default_factory=list
    )  # per-call thinking+answer log

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
    cfg = get_settings()
    assert config.run_id is not None
    prompt_config = get_variant(config.prompt_variant)

    _console.rule(
        f"[bold cyan]Run {config.run_id} | variant={config.prompt_variant!r}"
    )

    # Create a per-run context for explicit run_id-keyed suite state.
    run_context = RunContext()

    suite: TestSuite = run_context.reset_suite(config.function_path, config.run_id)
    is_gtest = config.executor_backend.strip().lower() in ("gtest", "gcov", "local")
    if is_gtest:
        suite.coverage.mcdc_execution_feedback = False
        if config.prompt_variant in ("baseline",):
            _console.print(
                "[yellow]gtest executor: variant 'baseline' lacks test-driver format rules; "
                "prefer 'no_reflection' or 'full'.[/]"
            )
        inferred = None
        try:
            from covxplore.driver.gtest_executor import infer_gtest_config

            inferred = infer_gtest_config(
                config.function_path,
                source_root=config.gtest_source_root,
                project_sources=config.gtest_project_sources or None,
                extra_compile_flags=config.gtest_extra_compile_flags or None,
            )
            _console.print(
                f"[dim]gtest: backend={config.gtest_coverage_backend} "
                f"| compiler={config.gtest_compiler or 'auto'} "
                f"| root={inferred['gtest_source_root']} "
                f"| link {len(inferred['gtest_project_sources'])} sibling source(s) "
                f"| flags={inferred['gtest_extra_compile_flags']}[/]"
            )
        except OSError as exc:
            _console.print(f"[yellow]gtest auto-config warning: {exc}[/]")

    _prefetch_conditions(suite)

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
    llm_logger = LLMInteractionLogger()

    try:
        static_conditions_text, static_context_text, static_source_text = (
            _prefetch_static_prompt_data(config.function_path)
        )
        crew_inst, builder = build_crew(
            prompt_config=prompt_config,
            max_iterations=config.max_iterations,
            run_context=run_context,
            executor=create_executor(
                config.executor_backend,
                function_path=config.function_path,
                gtest_source_root=config.gtest_source_root,
                gtest_project_sources=config.gtest_project_sources or None,
                gtest_extra_compile_flags=config.gtest_extra_compile_flags or None,
                gtest_coverage_backend=config.gtest_coverage_backend,
                gtest_compiler=config.gtest_compiler,
            ),
        )
        assert isinstance(builder, PromptBuilder)
        inputs = {
            "agent_backstory": builder.system_prompt(),
            "task_description": builder.task_description(
                function_path=config.function_path,
                run_id=config.run_id,
                suite=suite,
                remaining_iterations=config.max_iterations,
                static_conditions_text=static_conditions_text,
                static_context_text=static_context_text,
                static_source_text=static_source_text,
            ),
        }
        crew_obj = crew_inst.crew()
        llm_logger.attach()
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
        llm_logger.detach()
        if crew_inst is not None:
            try:
                metrics = crew_inst.crew().usage_metrics
                if metrics:
                    crew_prompt_tokens = metrics.prompt_tokens
                    crew_completion_tokens = metrics.completion_tokens
            except Exception:
                pass
            try:
                tracing_url = getattr(crew_inst.crew(), "_telemetry_url", None)
            except Exception:
                pass
        if not crew_prompt_tokens:
            crew_prompt_tokens = sum(
                i["usage"].get("prompt_tokens", 0) for i in llm_logger.interactions
            ) or None
        if not crew_completion_tokens:
            crew_completion_tokens = sum(
                i["usage"].get("completion_tokens", 0) for i in llm_logger.interactions
            ) or None
        final_suite = run_context.get_suite(config.run_id) or suite
        _reconcile_tokens(crew_inst, final_suite)

    if stop_reason == "agent_done":
        if final_suite.iteration_count == 0:
            # LLM returned a text answer without calling execute_testcase at all.
            # This is non-deterministic LLM behaviour; flag as error so callers
            # can retry rather than silently recording a 0-iteration result.
            stop_reason = "error"
            error_msg = (
                "LLM returned a final answer without calling execute_testcase "
                "(0 tool calls). Re-run to retry."
            )
            _console.print("[red]Agent produced no tool calls (0 iterations) — marking as error.[/]")
        elif _coverage_target_reached(final_suite, config):
            stop_reason = "coverage_target"
            _console.print("[green]Stopped gracefully: reached coverage target[/]")
        elif final_suite.coverage.consecutive_redundant >= config.redundant_streak_limit:
            stop_reason = "redundant_streak"
            _console.print(
                f"[yellow]Stopped: {config.redundant_streak_limit} consecutive redundant tests[/]"
            )
        elif final_suite.iteration_count >= config.max_iterations:
            stop_reason = "max_iter"
            _console.print(
                f"[yellow]Stopped gracefully: max_iter after {final_suite.iteration_count} iterations[/]"
            )

    result = GenerationResult(
        config=config,
        suite=final_suite,
        stop_reason=stop_reason,
        error_message=error_msg,
        crew_prompt_tokens=crew_prompt_tokens,
        crew_completion_tokens=crew_completion_tokens,
        tracing_url=tracing_url,
        llm_interactions=llm_logger.interactions,
    )

    _print_result_summary(result)
    run_context.cleanup_suite(config.run_id)
    return result


def _coverage_target_reached(suite: TestSuite, config: GenerationConfig) -> bool:
    """Return True when all applicable coverage targets are satisfied."""
    if suite.iteration_count == 0:
        return False
    metrics = suite.coverage.metrics(suite.tests)
    # Guard: require at least some statement or branch data before declaring
    # done.  Zero totals means no test has successfully measured coverage yet
    # (e.g. all tests compiled with errors).
    if metrics.total_statements == 0 and metrics.total_branches == 0:
        return False
    mcdc_done = (
        not suite.coverage.mcdc_execution_feedback
        or metrics.total_mcdc_pairs == 0
        or metrics.mcdc_pct >= config.mcdc_target
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


def _prefetch_conditions(suite: TestSuite) -> None:
    """Call /api/node/conditions before kickoff to pre-populate total_mcdc_pairs.

    This ensures stop-reason deduction is correct even when all generated tests
    fail to compile (otherwise total_mcdc_pairs stays 0 and the run is
    falsely marked as coverage_target).
    """
    from covxplore.api_client import AkaUTClient, AkaUTError

    try:
        with AkaUTClient() as client:
            result = client.get_node_conditions(suite.function_path)
        if result.total_mcdc_pairs > 0:
            suite.coverage.seed_conditions(result.conditions, result.total_mcdc_pairs)
            _console.print(
                f"[dim]Static CFG: {result.total_conditions} conditions "
                f"({result.total_mcdc_pairs} MC/DC pairs)[/]"
            )
    except AkaUTError as exc:
        _console.print(f"[yellow]Could not prefetch conditions: {exc}[/]")


def _format_conditions_for_prompt(result) -> str:
    lines = [
        f"Found {result.total_conditions} conditions "
        f"({result.total_mcdc_pairs} MC/DC pairs to cover):",
        "",
    ]
    for i, c in enumerate(result.conditions, start=1):
        if c.node_id is None:
            raise RuntimeError(
                "Backend payload missing nodeId in /api/node/conditions. "
                "nodeId is required for MC/DC identity."
            )
        line = c.line_in_function if c.line_in_function is not None else "?"
        start = c.start_offset if c.start_offset is not None else "?"
        end = c.end_offset if c.end_offset is not None else "?"
        lines.append(
            f"  {i}. [node:{c.node_id} line+{line}, offset {start}–{end}] {c.condition!r}"
        )
    return "\n".join(lines)


def _prefetch_static_prompt_data(function_path: str) -> tuple[str, str, str]:
    """Fetch static data once and return prompt-ready text blocks."""
    from covxplore.api_client import AkaUTClient
    from covxplore.tools.get_source import _number_lines

    with AkaUTClient() as client:
        cond = client.get_node_conditions(function_path)
        ctx = client.get_function_context(function_path)
        src = client.get_node_source(function_path)

    cond_text = _format_conditions_for_prompt(cond)
    ctx_text = ctx.context
    src_text = f"// Source: {function_path}\n{_number_lines(src.source)}"
    return cond_text, ctx_text, src_text


def _reconcile_tokens(crew_inst, suite: TestSuite) -> None:
    """Fallback token attribution after kickoff() completes.

    Distributes the overall metrics.prompt_tokens and metrics.completion_tokens
    evenly across all PASSED/RUNTIME_ERROR tests in the suite.
    """
    if crew_inst is None:
        return
    if suite.total_input_tokens > 0 or suite.total_output_tokens > 0:
        return
    try:
        metrics = crew_inst.crew().usage_metrics
        total_prompt = getattr(metrics, "prompt_tokens", 0) or 0
        total_completion = getattr(metrics, "completion_tokens", 0) or 0
    except Exception:
        return
    if total_prompt == 0 and total_completion == 0:
        return
    eligible = [
        t
        for t in suite.tests
        if t.status in {TestStatus.PASSED.value, TestStatus.RUNTIME_ERROR.value}
    ]
    if not eligible:
        return
    n = len(eligible)
    base_in, rem_in = divmod(total_prompt, n)
    base_out, rem_out = divmod(total_completion, n)
    for i, t in enumerate(eligible):
        t.token_input = base_in + (rem_in if i == n - 1 else 0)
        t.token_output = base_out + (rem_out if i == n - 1 else 0)


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
