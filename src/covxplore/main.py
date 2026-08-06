from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# run_generation  (covxplore-gen)
# ---------------------------------------------------------------------------

def run_generation() -> None:
    """Single generation run — one function, one prompt variant."""
    parser = argparse.ArgumentParser(
        prog="covxplore-gen",
        description="Generate statement- and branch-covering tests for one C/C++ function.",
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
        "--context-version", choices=("v1", "v2"), default=None,
        help="Static function context provider (default: Settings.context_version).",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Override max batches (default: Settings.max_batches).",
    )
    args = parser.parse_args()

    from covxplore.config import get_settings
    from covxplore.generator import GenerationConfig, generate

    cfg = get_settings()
    variant = args.variant or cfg.default_prompt_variant

    exp_cfg = GenerationConfig(
        function_path=args.path,
        prompt_variant=variant,
        context_version=args.context_version or cfg.context_version,
        max_batches=args.max_batches if args.max_batches is not None else cfg.max_batches,
    )

    result = generate(exp_cfg)

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
# run_parallel  (covxplore-pipeline)
# ---------------------------------------------------------------------------


def run_parallel() -> None:
    """Parallel pipeline — run ablation for every function in a paths file."""
    parser = argparse.ArgumentParser(
        prog="covxplore-pipeline",
        description=(
            "Run statement/branch ablation for every function listed in a paths file, "
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
        "--repeat", "-r", type=int, default=None,
        help="Repetitions per variant. Defaults to Settings.ablation_repeat.",
    )
    parser.add_argument(
        "--out", "-o", default="results",
        help="Root output directory. Each function gets a subdirectory (default: results).",
    )
    args = parser.parse_args()

    from covxplore.pipeline import ParallelPipeline

    pipeline = ParallelPipeline(
        paths_file=Path(args.paths_file),
        out_dir=Path(args.out),
        variants=args.variants,
        repeat=args.repeat,
        max_workers=args.workers,
    )
    summary_path = pipeline.run()
    print(f"\nPipeline complete. Summary: {summary_path}")

