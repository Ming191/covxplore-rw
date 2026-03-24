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
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


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
        max_iterations=args.max_iter or cfg.max_iterations,
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
    parser.add_argument("--repeat", "-r", type=int, default=None)
    parser.add_argument("--out", "-o", default="results")
    args = parser.parse_args()

    from covxplore.ablation import AblationRunner

    runner = AblationRunner()
    results = runner.run_matrix(
        function_path=args.path,
        variants=args.variants,
        repeat=args.repeat,
    )
    runner.export_results(results, Path(args.out))


# ---------------------------------------------------------------------------
# Legacy crewai entry points (kept for backward compatibility)
# ---------------------------------------------------------------------------

def run() -> None:
    """Default entry point — delegates to run_generation."""
    run_generation()


def train() -> None:
    cfg_args = sys.argv[1:]
    if len(cfg_args) < 2:
        print("Usage: train <n_iterations> <filename>")
        sys.exit(1)
    print("Training not supported in covxplore v0.2; use run_generation instead.")


def replay() -> None:
    print("Replay not supported in covxplore v0.2.")


def test() -> None:
    print("Use covxplore-gen or covxplore-ablate instead.")
