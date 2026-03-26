#!/usr/bin/env python
"""covxplore CLI entry points.

Commands
--------
covxplore-gen
    Run a single test generation for one function with one prompt variant.
    Example:
        covxplore-gen \\
            --path "/project/src/foo.cpp\\MyNS::bar(int)" \\
            --variant full \\
            --out results/

covxplore-ablate
    Run the full ablation matrix (all or selected variants, N repeats).
    Example:
        covxplore-ablate \\
            --path "/project/src/foo.cpp\\MyNS::bar(int)" \\
            --variants full no_cot no_coverage baseline \\
            --repeat 3 \\
            --out results/
        covxplore-ablate \\
            --path "/project/src/foo.cpp\\MyNS::bar(int)" \\
            --leave-one-out \\
            --repeat 3 \\
            --out results/
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from dotenv import load_dotenv
import os

load_dotenv()

print("Tracing:", os.getenv("CREWAI_TRACING"))
print("Project:", os.getenv("CREWAI_PROJECT_NAME"))
print("Key:", os.getenv("CREWAI_API_KEY")[:8])


# ---------------------------------------------------------------------------
# run_generation  (covxplore-gen)
# ---------------------------------------------------------------------------

def run_generation() -> None:
    """Single generation run — one function, one prompt variant."""
    parser = argparse.ArgumentParser(
        prog="covxplore-gen",
        description="Generate MC/DC-covering tests for one C/C++ function.",
    )
    parser.add_argument(
        "--path", "-p", required=True,
        help="Absolute path of the function node (as returned by /api/search).",
    )
    parser.add_argument(
        "--variant", "-v", default=None,
        help="Prompt variant name (default: value from Settings.default_prompt_variant).",
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Directory to write the summary JSON. Defaults to current directory.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=None,
        help="Override max iterations (default: Settings.max_iterations).",
    )
    parser.add_argument(
        "--mcdc-target", type=float, default=None,
        help="Override MC/DC target 0.0–1.0 (default: Settings.mcdc_target).",
    )
    args = parser.parse_args()

    from covxplore.ablation import AblationRunner
    from covxplore.config import get_settings
    from covxplore.experiment import ExperimentConfig

    cfg = get_settings()
    variant = args.variant or cfg.default_prompt_variant

    exp_cfg = ExperimentConfig(
        function_path=args.path,
        prompt_variant=variant,
        max_iterations=args.max_iter if args.max_iter is not None else cfg.max_iterations,
        mcdc_target=args.mcdc_target if args.mcdc_target is not None else cfg.mcdc_target,
    )

    runner = AblationRunner()
    result = runner.run_one(exp_cfg)

    summary = result.to_summary_dict()
    out_dir = Path(args.out) if args.out else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gen_{exp_cfg.run_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary written to: {out_path}")


# ---------------------------------------------------------------------------
# run_ablation  (covxplore-ablate)
# ---------------------------------------------------------------------------

def run_ablation() -> None:
    """Full ablation matrix run."""
    parser = argparse.ArgumentParser(
        prog="covxplore-ablate",
        description="Run the ablation study across prompt variants.",
    )
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Variant names to include. Defaults to all variants.",
    )
    parser.add_argument(
        "--leave-one-out", "--loo",
        action="store_true",
        help="Run leave-one-out preset: full, full_shots, and all variants starting with 'no_'.",
    )
    parser.add_argument("--repeat", "-r", type=int, default=None)
    parser.add_argument("--out", "-o", default="results")
    args = parser.parse_args()

    from covxplore.ablation import AblationRunner
    from covxplore.prompts.registry import get_leave_one_out_variants

    if args.leave_one_out and args.variants:
        parser.error("Use either --variants or --leave-one-out, not both.")

    variants = get_leave_one_out_variants() if args.leave_one_out else args.variants

    runner = AblationRunner()
    results = runner.run_matrix(
        function_path=args.path,
        variants=variants,
        repeat=args.repeat,
    )
    runner.export_results(results, Path(args.out))


# ---------------------------------------------------------------------------
# run_parallel  (covxplore-pipeline)
# ---------------------------------------------------------------------------


def run_parallel() -> None:
    """Parallel pipeline — run ablation for every function in a paths file."""
    parser = argparse.ArgumentParser(
        prog="covxplore-pipeline",
        description=(
            "Run MC/DC ablation for every function listed in a paths file, "
            "in parallel. Results land in per-function subdirectories under --out, "
            "and a combined pipeline_summary.csv is written to --out."
        ),
    )
    parser.add_argument(
        "--paths-file", "-f", required=True,
        help="Text file listing function paths (one per line; lines starting with # are ignored).",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=3,
        help="Maximum number of functions to process concurrently (default: 3).",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Prompt variant names to run. Defaults to all variants.",
    )
    parser.add_argument(
        "--leave-one-out", "--loo",
        action="store_true",
        help="Run leave-one-out preset: full, full_shots, and all variants starting with 'no_'.",
    )
    parser.add_argument(
        "--repeat", "-r", type=int, default=None,
        help="Repetitions per variant. Defaults to Settings.ablation_repeat.",
    )
    parser.add_argument(
        "--out", "-o", default="results",
        help="Root output directory. Each function gets a subdirectory (default: results).",
    )
    args = parser.parse_args()

    from covxplore.pipeline import ParallelPipeline
    from covxplore.prompts.registry import get_leave_one_out_variants

    if args.leave_one_out and args.variants:
        parser.error("Use either --variants or --leave-one-out, not both.")

    variants = get_leave_one_out_variants() if args.leave_one_out else args.variants

    pipeline = ParallelPipeline(
        paths_file=Path(args.paths_file),
        out_dir=Path(args.out),
        variants=variants,
        repeat=args.repeat,
        max_workers=args.workers,
    )
    summary_path = pipeline.run()
    print(f"\nPipeline complete. Summary: {summary_path}")

