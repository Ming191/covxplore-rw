from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_FLAT_FIELDNAMES = [
    "run_id",
    "function_path",
    "prompt_variant",
    "stop_reason",
    "statement_coverage_pct",
    "branch_coverage_pct",
    "mcdc_coverage_pct",
    "covered_statements",
    "total_statements",
    "covered_branches",
    "total_branches",
    "covered_mcdc_pairs",
    "total_mcdc_pairs",
    "redundancy_rate",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "elapsed_sec",
    "iterations_used",
    "num_tests",
    "num_passing",
    "num_redundant",
    "llm_call_count",
    "total_llm_elapsed_ms",
    "avg_llm_elapsed_ms",
    "total_execute_elapsed_ms",
    "avg_execute_elapsed_ms",
    "statements_per_llm_call",
    "branches_per_llm_call",
    "mcdc_pairs_per_llm_call",
    "error",
]


def write_experiment_results(
    results: Sequence[Any],
    out_dir: Path,
    *,
    prefix: str = "ablation",
    console: Any | None = None,
) -> None:
    """Write per-run JSON, aggregate JSON, and flat CSV for experiment results."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for result in results:
        summary = result.to_summary_dict()
        all_summaries.append(summary)
        run_path = out_dir / f"{prefix}_{result.config.run_id}.json"
        run_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    aggregate_path = out_dir / f"{prefix}_all.json"
    aggregate_path.write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = [result.to_flat_row() for result in results]
    write_rows_csv(
        rows,
        out_dir / f"{prefix}_summary.csv",
        default_fieldnames=DEFAULT_FLAT_FIELDNAMES,
    )
    if console is not None:
        console.print(f"[green]Exported {len(results)} results to {out_dir}/[/]")


def write_rows_csv(
    rows: Sequence[dict],
    csv_path: Path,
    *,
    default_fieldnames: Sequence[str] = DEFAULT_FLAT_FIELDNAMES,
    leading_columns: Sequence[str] = (),
) -> Path:
    """Write flat row dictionaries to CSV using pandas when available."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

    try:
        import pandas as pd

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=default_fieldnames)
        for column in reversed(leading_columns):
            if column in df.columns:
                cols = [column] + [c for c in df.columns if c != column]
                df = df[cols]
        df.to_csv(csv_path, index=False)
    except ImportError:
        fieldnames = _fieldnames(rows, default_fieldnames, leading_columns)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return csv_path


def _fieldnames(
    rows: Sequence[dict],
    default_fieldnames: Sequence[str],
    leading_columns: Sequence[str],
) -> list[str]:
    if rows:
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    else:
        names = list(default_fieldnames)

    for column in reversed(leading_columns):
        if column in names:
            names = [column] + [name for name in names if name != column]
    return names
