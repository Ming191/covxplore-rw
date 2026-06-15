from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from covxplore.config import get_settings
from covxplore.experiment import flat_row
from covxplore.generator import GenerationConfig, GenerationResult, generate
from covxplore.prompts.registry import VARIANTS


_console = Console()


class AblationRunner:
    """Runs one or many experiments and collects results."""

    def run_one(self, config: GenerationConfig) -> GenerationResult:
        """Execute a single experiment run end-to-end."""
        return generate(config)

    def run_matrix(
        self,
        function_path: str,
        variants: list[str] | None = None,
        repeat: int | None = None,
    ) -> list[GenerationResult]:
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

        results: list[GenerationResult] = []
        total = len(variants) * repeat

        _console.rule(
            f"[bold]Ablation matrix: {len(variants)} variants × {repeat} repeats "
            f"= {total} runs[/]"
        )

        for variant in variants:
            for rep in range(repeat):
                exp_cfg = GenerationConfig(
                    function_path=function_path,
                    prompt_variant=variant,
                    max_iterations=cfg.max_iterations,
                    mcdc_target=cfg.mcdc_target,
                )
                results.append(generate(exp_cfg))

        _print_matrix_summary(results)
        return results

    def export_results(
        self,
        results: list[GenerationResult],
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

            rows = [flat_row(r) for r in results]
            df = pd.DataFrame(rows)
            csv_path = out_dir / f"{prefix}_summary.csv"
            df.to_csv(csv_path, index=False)
            _console.print(f"[green]Exported {len(results)} results to {out_dir}/[/]")
        except ImportError:
            _console.print(
                "[yellow]pandas not installed — CSV export skipped. "
                "Install with: pip install pandas[/]"
            )


def _print_matrix_summary(results: list[GenerationResult]) -> None:
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
