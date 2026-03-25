"""Parallel pipeline — run ablation for multiple C/C++ functions concurrently.

Usage:

    from pathlib import Path
    from covxplore.pipeline import ParallelPipeline

    pipeline = ParallelPipeline(
        paths_file=Path("function_paths.txt"),
        out_dir=Path("results/"),
        variants=["full", "baseline"],
        repeat=1,
        max_workers=3,
    )
    summary_path = pipeline.run()
    print(f"Done. Summary: {summary_path}")

CLI entry point: ``covxplore-pipeline`` (see main.py).
"""
from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console

from covxplore.ablation import AblationRunner
from covxplore.experiment import ExperimentResult

_console = Console()

def parse_function_paths(paths_file: Path) -> list[str]:
    """
    Read a paths file and return non-comment, non-empty lines.

    Lines starting with ``#`` and blank lines are ignored.
    Raises ``FileNotFoundError`` if the file does not exist, and
    ``ValueError`` if no valid paths are found.
    """
    lines = paths_file.read_text(encoding="utf-8").splitlines()
    paths = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not paths:
        raise ValueError(f"No function paths found in {paths_file}")
    return paths


def _safe_name(function_path: str) -> str:
    """
    Derive a filesystem-safe subdirectory name from a function path.

    Takes the last two ``/``-delimited segments (e.g. ``hjson_decode.cpp``
    and ``Hjson::_readMLString(Parser*)``), sanitizes each, and joins with
    ``__``.  Truncated to 80 characters.
    """
    parts = function_path.rstrip("/").split("/")
    segments = parts[-2:] if len(parts) >= 2 else parts[-1:]
    sanitized = [re.sub(r"[^A-Za-z0-9_-]+", "_", seg).strip("_") for seg in segments]
    name = "__".join(sanitized)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80]

class ParallelPipeline:
    """
    Runs ablation for every function in ``paths_file`` using a thread pool.

    Each function is processed in its own thread (and therefore its own
    thread-local ``TestSuite``).  Results are written to per-function
    subdirectories under ``out_dir``, and a combined ``pipeline_summary.csv``
    is written to ``out_dir`` when all threads finish.
    """

    def __init__(
        self,
        paths_file: Path,
        out_dir: Path,
        variants: list[str] | None = None,
        repeat: int | None = None,
        max_workers: int = 3,
    ):
        self.paths_file = paths_file
        self.out_dir = out_dir
        self.variants = variants
        self.repeat = repeat
        self.max_workers = max_workers

    def run(self) -> Path:
        """Execute the parallel pipeline and return the path to the combined CSV."""
        function_paths = parse_function_paths(self.paths_file)

        _console.rule(
            f"[bold]Pipeline: {len(function_paths)} functions, "
            f"max_workers={self.max_workers}[/]"
        )
        for fp in function_paths:
            _console.print(f"  • {fp}")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        all_results: list[ExperimentResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._run_one, fp): fp
                for fp in function_paths
            }
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    _console.print(f"[green][done][/] {fp}")
                except Exception as exc:
                    _console.print(f"[red][error][/] {fp}: {exc}")

        summary_path = _write_pipeline_summary(all_results, self.out_dir)
        _console.rule(f"[bold green]Pipeline complete[/]")
        _console.print(f"Combined summary: {summary_path}")
        return summary_path

    def _run_one(self, function_path: str) -> list[ExperimentResult]:
        """Worker: run the full ablation matrix for one function."""
        func_name = _safe_name(function_path)
        func_out = self.out_dir / func_name
        func_out.mkdir(parents=True, exist_ok=True)

        runner = AblationRunner()
        results = runner.run_matrix(
            function_path=function_path,
            variants=self.variants,
            repeat=self.repeat,
        )
        runner.export_results(results, func_out, prefix=func_name)
        return results

def _write_pipeline_summary(
    results: list[ExperimentResult],
    out_dir: Path,
) -> Path:
    """Write a combined flat CSV for all results across all functions."""
    rows = []
    for r in results:
        row = r.to_flat_row()
        row["function_name"] = _safe_name(r.config.function_path)
        rows.append(row)

    rows.sort(key=lambda r: (r["function_name"], r["prompt_variant"], r["run_id"]))

    csv_path = out_dir / "pipeline_summary.csv"

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        if not df.empty and "function_name" in df.columns:
            cols = ["function_name"] + [c for c in df.columns if c != "function_name"]
            df = df[cols]
        df.to_csv(csv_path, index=False)
    except ImportError:
        if rows:
            fieldnames = ["function_name"] + [k for k in rows[0] if k != "function_name"]
        else:
            fieldnames = [
                "function_name", "run_id", "function_path", "prompt_variant",
                "stop_reason", "mcdc_coverage_pct", "covered_mcdc_pairs",
                "total_mcdc_pairs", "redundancy_rate", "total_input_tokens",
                "total_output_tokens", "total_tokens", "elapsed_sec",
                "iterations_used", "num_tests", "num_passing", "num_redundant", "error",
            ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return csv_path
