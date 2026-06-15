#!/usr/bin/env python
"""covxplore CLI entry point.

covxplore-gen
    Run a single test generation for one function with one prompt variant.
    Example:
        covxplore-gen \\
            --path "/project/src/foo.cpp\\MyNS::bar(int)" \\
            --variant full \\
            --out results/
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from dotenv import load_dotenv

load_dotenv()


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

    from covxplore.config import get_settings
    from covxplore.generator import GenerationConfig, generate

    cfg = get_settings()
    variant = args.variant or cfg.default_prompt_variant

    exp_cfg = GenerationConfig(
        function_path=args.path,
        prompt_variant=variant,
        max_iterations=args.max_iter if args.max_iter is not None else cfg.max_iterations,
        mcdc_target=args.mcdc_target if args.mcdc_target is not None else cfg.mcdc_target,
    )

    result = generate(exp_cfg)

    summary = result.to_summary_dict()
    out_dir = Path(args.out) if args.out else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gen_{exp_cfg.run_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary written to: {out_path}")
