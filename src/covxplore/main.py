from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
load_dotenv()


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy", choices=("llm", "symbolic", "hybrid"), default=None
    )
    parser.add_argument("--statement-target", type=float, default=None)
    parser.add_argument("--branch-target", type=float, default=None)
    parser.add_argument(
        "--scheduler", choices=("rule", "collect", "frozen"), default=None
    )
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--wall-time-minutes", type=float, default=None)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-symbolic-attempts", type=int, default=None)
    parser.add_argument("--max-test-executions", type=int, default=None)


def _generation_config(args: argparse.Namespace, function_path: str):
    from covxplore.config import get_settings
    from covxplore.hybrid.runner import HybridGenerationConfig

    settings = get_settings()
    return HybridGenerationConfig(
        function_path=function_path,
        strategy=args.strategy or settings.strategy,
        statement_target=(
            args.statement_target
            if args.statement_target is not None
            else settings.statement_target
        ),
        branch_target=(
            args.branch_target
            if args.branch_target is not None
            else settings.branch_target
        ),
        scheduler_mode=args.scheduler or settings.scheduler_mode,
        policy_path=args.policy,
        wall_time_minutes=(
            args.wall_time_minutes
            if args.wall_time_minutes is not None
            else settings.wall_time_minutes
        ),
        max_llm_calls=(
            args.max_llm_calls
            if args.max_llm_calls is not None
            else settings.max_llm_calls
        ),
        max_symbolic_attempts=(
            args.max_symbolic_attempts
            if args.max_symbolic_attempts is not None
            else settings.max_symbolic_attempts
        ),
        max_test_executions=(
            args.max_test_executions
            if args.max_test_executions is not None
            else settings.max_test_executions
        ),
    )


def run_generation() -> None:
    """Run COV127 generation for one C/C++ focal method."""
    parser = argparse.ArgumentParser(
        prog="covxplore-gen",
        description="Generate statement/branch-targeted tests for one C/C++ function.",
    )
    parser.add_argument(
        "--path", "-p", required=True,
        help="Absolute function path returned by AkaUT /api/search.",
    )
    parser.add_argument(
        "--out", "-o", default="results",
        help="Directory for JSON, per-run CSV, and cumulative summary.csv.",
    )
    _add_generation_arguments(parser)
    args = parser.parse_args()

    from covxplore.api_client import AkaUTClient
    from covxplore.hybrid.artifacts import write_artifacts
    from covxplore.hybrid.runner import HybridGenerationRunner

    config = _generation_config(args, args.path)
    with AkaUTClient() as client:
        result = HybridGenerationRunner(client=client).run(config)

    json_path, csv_path, summary_path = write_artifacts(result, Path(args.out))
    metrics = result.to_summary_dict()["metrics"]
    print(
        "\n"
        f"Statement: {metrics['statement_cov'] * 100:.2f}% "
        f"({metrics['covered_statement']}/{metrics['total_statement']}) | "
        f"Branch: {metrics['branch_cov'] * 100:.2f}% "
        f"({metrics['covered_branch']}/{metrics['total_branch']}) | "
        f"stop: {result.stop_reason}"
    )
    print(
        f"Tokens: {metrics['total_tokens']} | "
        f"time: {metrics['elapsed_sec']:.2f}s"
    )
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Cumulative CSV: {summary_path}")


def run_ablation() -> None:
    """Run COV127 baselines and ablations for one C++ focal method."""
    parser = argparse.ArgumentParser(
        prog="covxplore-ablate",
        description="Run COV127 strategy baselines and component ablations.",
    )
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Experiment names; defaults to variants that do not require a policy.",
    )
    parser.add_argument("--repeat", "-r", type=int, default=None)
    parser.add_argument("--out", "-o", default="results")
    _add_generation_arguments(parser)
    args = parser.parse_args()

    from covxplore.api_client import AkaUTClient
    from covxplore.config import get_settings
    from covxplore.hybrid.experiments import (
        EXPERIMENT_VARIANTS,
        HybridExperimentRunner,
    )

    variants = args.variants or [
        name for name in EXPERIMENT_VARIANTS if name != "hybrid_frozen"
    ]
    if "hybrid_frozen" in variants and args.policy is None:
        parser.error("hybrid_frozen requires --policy")
    repeat = args.repeat or get_settings().ablation_repeat
    with AkaUTClient() as client:
        summary = HybridExperimentRunner(client).run_matrix(
            _generation_config(args, args.path),
            variants=variants,
            repeat=repeat,
            out_dir=Path(args.out),
        )
    print(f"Experiment summary: {summary}")


def run_batch() -> None:
    """Run COV127 sequentially for selected C++ focal methods."""
    parser = argparse.ArgumentParser(
        prog="covxplore-batch",
        description=(
            "Generate tests sequentially because AkaUT owns process-global execution state."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--paths-file", "-f", type=Path,
        help="UTF-8 file containing one AkaUT absolute function path per line.",
    )
    source.add_argument(
        "--source-files", nargs="+",
        help="C/C++ source names or paths used to filter AkaUT search results.",
    )
    parser.add_argument("--out", "-o", default="results")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not skip completed function/strategy/scheduler identities.",
    )
    _add_generation_arguments(parser)
    args = parser.parse_args()

    from covxplore.api_client import AkaUTClient
    from covxplore.hybrid.batch import BatchConfig, HybridBatchRunner

    function_paths = (
        HybridBatchRunner.load_paths(args.paths_file) if args.paths_file else []
    )
    with AkaUTClient() as client:
        summary = HybridBatchRunner(client).run(
            BatchConfig(
                generation=_generation_config(args, "__batch__"),
                out_dir=Path(args.out),
                function_paths=function_paths,
                source_files=args.source_files or [],
                resume=not args.no_resume,
            )
        )
    print(f"Batch summary: {summary}")


def run_parallel() -> None:
    """Backward-compatible alias; COV127 execution is deliberately sequential."""
    run_batch()


def run_policy() -> None:
    """Inspect or freeze a collected COV127 LinUCB policy."""
    parser = argparse.ArgumentParser(prog="covxplore-policy")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Write an immutable evaluation policy.")
    freeze.add_argument("--input", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Print policy metadata.")
    inspect.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()

    from covxplore.hybrid.scheduler import LinUCBPolicy

    if args.command == "freeze":
        policy = LinUCBPolicy.load(args.input, frozen=True)
        policy.freeze_to(args.out)
        print(f"Frozen policy: {args.out}")
        return
    policy = LinUCBPolicy.load(args.policy)
    print(
        f"algorithm=LinUCB frozen={policy.frozen} alpha={policy.alpha} "
        f"ridge={policy.ridge} observations="
        + ",".join(
            f"{route.value}:{policy.observations[route]}" for route in policy.observations
        )
    )
