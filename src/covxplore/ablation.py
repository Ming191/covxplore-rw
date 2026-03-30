from __future__ import annotations

import json
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table

from covxplore.config import get_settings
from covxplore.crew import build_crew
from covxplore.experiment import ExperimentConfig, ExperimentResult
from covxplore.llm_logger import LLMInteractionLogger
from covxplore.models import TestSuite
from covxplore.status import TestStatus
from covxplore.prompts.registry import VARIANTS, get_variant
from covxplore.prompts.builder import PromptBuilder
from covxplore.tools.execute_testcase import (
    get_shared_suite,
    reset_shared_suite,
    cleanup_suite,
)


_console = Console()


class AblationRunner:
    """Runs one or many experiments and collects results."""

    def run_one(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Execute a single experiment run end-to-end.
        """
        cfg = get_settings()
        assert config.run_id is not None
        prompt_config = get_variant(config.prompt_variant)

        _console.rule(
            f"[bold cyan]Run {config.run_id} | variant={config.prompt_variant!r}"
        )

        # Reset the shared suite for this run
        suite: TestSuite = reset_shared_suite(config.function_path, config.run_id)
        _prefetch_conditions(suite)

        stop_reason = "agent_done"
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
            )
            assert isinstance(builder, PromptBuilder)
            inputs = {
                "agent_backstory": builder.system_prompt(),
                "task_description": builder.task_description(
                    function_path=config.function_path,
                    suite=suite,
                    remaining_iterations=config.max_iterations,
                    static_conditions_text=static_conditions_text,
                    static_context_text=static_context_text,
                    static_source_text=static_source_text,
                ),
            }
            # Attach after build_crew() so CrewAI's tracing setup
            # (which resets litellm.callbacks) doesn't wipe our shim.
            llm_logger.attach()
            crew_inst.crew().kickoff(inputs=inputs)

        except Exception as e:
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
            final_suite = get_shared_suite() or suite
            _reconcile_tokens(crew_inst, final_suite)

        # We must deduce early stops manually based on the final achieved coverage
        if final_suite.iteration_count >= config.max_iterations:
            stop_reason = "max_iter"
            error_msg = None
            _console.print(
                f"[yellow]Stopped gracefully: max_iter after {final_suite.iteration_count} iterations[/]"
            )
        elif final_suite.consecutive_redundant >= config.redundant_streak_limit:
            stop_reason = "redundant_streak"
            error_msg = None
            _console.print(
                f"[yellow]Stopped: {config.redundant_streak_limit} consecutive redundant tests[/]"
            )
        elif final_suite.mcdc_coverage_pct >= config.mcdc_target or (
            final_suite.iteration_count > 0
            and final_suite.total_mcdc_conditions > 0
            and not final_suite.unvisited_summary()
        ):
            stop_reason = "coverage_target"
            error_msg = None
            _console.print("[green]Stopped gracefully: reached coverage target[/]")

        result = ExperimentResult(
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
        cleanup_suite(config.run_id)
        return result

    def run_matrix(
        self,
        function_path: str,
        variants: list[str] | None = None,
        repeat: int | None = None,
    ) -> list[ExperimentResult]:
        """Run every requested variant ``repeat`` times and collect results.

        Parameters
        ----------
        function_path:
            Target function for all runs.
        variants:
            List of variant names from PromptRegistry. Defaults to all variants.
        repeat:
            Number of repetitions per variant. Defaults to ``Settings.ablation_repeat``.
        """
        cfg = get_settings()
        variants = variants or list(VARIANTS.keys())
        repeat = repeat if repeat is not None else cfg.ablation_repeat

        results: list[ExperimentResult] = []
        total = len(variants) * repeat

        _console.rule(
            f"[bold]Ablation matrix: {len(variants)} variants × {repeat} repeats "
            f"= {total} runs[/]"
        )

        for variant in variants:
            for rep in range(repeat):
                exp_cfg = ExperimentConfig(
                    function_path=function_path,
                    prompt_variant=variant,
                    max_iterations=cfg.max_iterations,
                    mcdc_target=cfg.mcdc_target,
                )
                result = self.run_one(exp_cfg)
                results.append(result)

        _print_matrix_summary(results)
        return results

    def export_results(
        self,
        results: list[ExperimentResult],
        out_dir: Path,
        *,
        prefix: str = "ablation",
    ) -> None:
        """Write results to ``out_dir`` as JSON (full) and CSV (flat rows).

        Creates ``out_dir`` if it does not exist.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # Full JSON (one file per run and one aggregate)
        all_summaries = []
        for r in results:
            summary = r.to_summary_dict()
            all_summaries.append(summary)
            run_path = out_dir / f"{prefix}_{r.config.run_id}.json"
            run_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        agg_path = out_dir / f"{prefix}_all.json"
        agg_path.write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Flat CSV
        try:
            import pandas as pd

            rows = [r.to_flat_row() for r in results]
            df = pd.DataFrame(rows)
            csv_path = out_dir / f"{prefix}_summary.csv"
            df.to_csv(csv_path, index=False)
            _console.print(f"[green]Exported {len(results)} results to {out_dir}/[/]")
        except ImportError:
            _console.print(
                "[yellow]pandas not installed — CSV export skipped. "
                "Install with: pip install pandas[/]"
            )


def _prefetch_conditions(suite: "TestSuite") -> None:
    """Call /api/node/conditions before kickoff to pre-populate total_mcdc_conditions.

    This ensures stop-reason deduction is correct even when all generated tests
    fail to compile (otherwise total_mcdc_conditions stays 0 and the run is
    falsely marked as coverage_target).
    """
    from covxplore.api_client import AkaUTClient, AkaUTError

    try:
        with AkaUTClient() as client:
            result = client.get_node_conditions(suite.function_path)
        if result.total_mcdc_pairs > 0:
            suite.total_mcdc_conditions = result.total_mcdc_pairs
            suite.all_conditions = [c.condition for c in result.conditions]
            for c in result.conditions:
                if c.node_id is None:
                    raise RuntimeError(
                        "Backend payload missing nodeId in /api/node/conditions. "
                        "nodeId is required for MC/DC identity."
                    )
                cid = c.node_id
                suite.condition_id_to_text[cid] = c.condition
                suite.condition_id_to_line[cid] = c.line_in_function
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


def _reconcile_tokens(crew_inst, suite: "TestSuite") -> None:
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


def _print_result_summary(r: ExperimentResult) -> None:
    m = r.to_summary_dict()["metrics"]
    _console.print(
        f"  MC/DC: [bold]{m['mcdc_coverage_pct'] * 100:.0f}%[/] "
        f"({m['covered_mcdc_pairs']}/{m['total_mcdc_pairs']})  |  "
        f"redundancy: {m['redundancy_rate'] * 100:.0f}%  |  "
        f"tokens: {m['total_tokens']:,}  |  "
        f"time: {m['elapsed_sec']:.1f}s  |  "
        f"stop: {r.stop_reason}"
    )


def _print_matrix_summary(results: list[ExperimentResult]) -> None:
    _console.rule("[bold]Ablation Summary[/]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Variant", style="cyan")
    table.add_column("MC/DC %", justify="right")
    table.add_column("Redundancy %", justify="right")
    table.add_column("Tokens (total)", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Stop reason")

    for r in results:
        m = r.to_summary_dict()["metrics"]
        table.add_row(
            r.config.prompt_variant,
            f"{m['mcdc_coverage_pct'] * 100:.0f}%",
            f"{m['redundancy_rate'] * 100:.0f}%",
            f"{m['total_tokens']:,}",
            f"{m['elapsed_sec']:.1f}",
            r.stop_reason,
        )

    _console.print(table)
