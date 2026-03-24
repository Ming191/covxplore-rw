"""AblationRunner — orchestrates single and matrix experiment runs.

Usage (single run)::

    from covxplore.ablation import AblationRunner
    from covxplore.experiment import ExperimentConfig

    runner = AblationRunner()
    cfg = ExperimentConfig(
        function_path="/project/src/foo.cpp\\\\MyNS::bar(int)",
        prompt_variant="full",
    )
    result = runner.run_one(cfg)
    print(result.to_summary_dict())

Usage (full ablation matrix)::

    results = runner.run_matrix(
        function_path="...",
        variants=["baseline", "s3_coverage", "no_cot", "full"],
        repeat=3,
    )
    runner.export_results(results, Path("results/"))
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table

from covxplore.config import get_settings
from covxplore.crew import _StopGeneration, build_crew
from covxplore.experiment import ExperimentConfig, ExperimentResult
from covxplore.models import TestSuite
from covxplore.prompts.registry import VARIANTS, get_variant
from covxplore.tools.execute_testcase import get_shared_suite, reset_shared_suite

_console = Console()


class AblationRunner:
    """Runs one or many experiments and collects results."""

    def run_one(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute a single experiment run end-to-end.

        Handles the crew lifecycle including early stopping via
        ``_StopGeneration`` and unexpected errors.
        """
        cfg = get_settings()
        prompt_config = get_variant(config.prompt_variant)

        _console.rule(
            f"[bold cyan]Run {config.run_id} | variant={config.prompt_variant!r}"
        )

        # Reset the shared suite for this run
        suite: TestSuite = reset_shared_suite(config.function_path)

        stop_reason = "agent_done"
        error_msg = None

        try:
            crew_inst, builder = build_crew(
                function_path=config.function_path,
                prompt_config=prompt_config,
                suite=suite,
            )
            inputs = {
                "agent_backstory": builder.system_prompt(),
                "task_description": builder.task_description(
                    function_path=config.function_path,
                    suite=suite,
                    remaining_iterations=config.max_iterations,
                ),
            }
            crew_inst.crew().kickoff(inputs=inputs)

        except _StopGeneration as e:
            stop_reason = e.reason
            _console.print(
                f"[yellow]Stopped: {stop_reason} after {suite.iteration_count} iterations[/]"
            )

        except Exception as e:
            stop_reason = "error"
            error_msg = f"{type(e).__name__}: {e}"
            _console.print(f"[red]Error: {error_msg}[/]")
            traceback.print_exc()

        # Retrieve the final suite (may have been updated by callbacks)
        final_suite = get_shared_suite() or suite

        # CrewAI swallows exceptions from step_callback, so we must deduce early stops manually
        if final_suite.iteration_count >= config.max_iterations:
            stop_reason = "max_iter"
            error_msg = None  # Clear the artificial error message
            _console.print(f"[yellow]Stopped gracefully: max_iter after {final_suite.iteration_count} iterations[/]")
        elif final_suite.mcdc_coverage_pct >= config.mcdc_target or final_suite.iteration_count > 0 and not final_suite.unvisited_summary():
            stop_reason = "coverage_target"
            error_msg = None
            _console.print("[green]Stopped gracefully: reached coverage target[/]")

        result = ExperimentResult(
            config=config,
            suite=final_suite,
            stop_reason=stop_reason,
            error_message=error_msg,
        )

        _print_result_summary(result)
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

        # Full JSON (one file per run + one aggregate)
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


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_result_summary(r: ExperimentResult) -> None:
    m = r.to_summary_dict()["metrics"]
    _console.print(
        f"  MC/DC: [bold]{m['mcdc_coverage_pct']*100:.0f}%[/] "
        f"({m['covered_mcdc_pairs']}/{m['total_mcdc_pairs']})  |  "
        f"redundancy: {m['redundancy_rate']*100:.0f}%  |  "
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
            f"{m['mcdc_coverage_pct']*100:.0f}%",
            f"{m['redundancy_rate']*100:.0f}%",
            f"{m['total_tokens']:,}",
            f"{m['elapsed_sec']:.1f}",
            r.stop_reason,
        )

    _console.print(table)
