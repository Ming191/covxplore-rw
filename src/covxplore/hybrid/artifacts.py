from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path

from covxplore.hybrid.runner import HybridGenerationResult


SUMMARY_COLUMNS = [
    "function_name",
    "function_path",
    "run_id",
    "config_hash",
    "strategy",
    "scheduler",
    "stop_reason",
    "statement_cov",
    "branch_cov",
    "covered_statement",
    "total_statement",
    "covered_branch",
    "total_branch",
    "statement_coverage_time_auc",
    "branch_coverage_time_auc",
    "total_time_used",
    "total_input_token",
    "total_output_token",
    "iteration_used",
    "num_testcases",
    "num_passing",
    "num_compile_error",
    "num_runtime_error",
    "num_infra_error",
    "redundancy_rate",
    "llm_calls",
    "symbolic_attempts",
    "z3_calls",
    "z3_solved",
    "z3_unsat",
    "z3_unsupported",
    "z3_timeout",
    "z3_error",
    "z3_solve_rate",
    "z3_time_used",
    "symbolic_time_used",
    "z3_binding_count",
    "symbolic_assisted_tests",
    "symbolic_assisted_passing",
    "symbolic_target_reached",
    "symbolic_useful_attempts",
    "symbolic_useful_rate",
    "symbolic_assisted_new_statements",
    "symbolic_assisted_new_branches",
    "symbolic_new_statements",
    "symbolic_new_branches",
    "hybrid_assisted_new_statements",
    "hybrid_assisted_new_branches",
    "partial_to_passing_rate",
    "error",
]


def result_row(result: HybridGenerationResult) -> dict[str, object]:
    summary = result.to_summary_dict()
    metrics = summary["metrics"]
    statuses = [
        attempt.execution.status.upper()
        for attempt in result.attempts
        if attempt.execution is not None
    ]
    return {
        "function_name": result.coverage_model.function_name,
        "function_path": result.config.function_path,
        "run_id": result.config.run_id,
        "config_hash": configuration_hash(result.config),
        "strategy": result.config.strategy,
        "scheduler": result.config.scheduler_mode,
        "stop_reason": result.stop_reason,
        "statement_cov": metrics["statement_cov"],
        "branch_cov": metrics["branch_cov"],
        "covered_statement": metrics["covered_statement"],
        "total_statement": metrics["total_statement"],
        "covered_branch": metrics["covered_branch"],
        "total_branch": metrics["total_branch"],
        "statement_coverage_time_auc": metrics["statement_coverage_time_auc"],
        "branch_coverage_time_auc": metrics["branch_coverage_time_auc"],
        "total_time_used": metrics["elapsed_sec"],
        "total_input_token": metrics["total_input_tokens"],
        "total_output_token": metrics["total_output_tokens"],
        "iteration_used": metrics["iteration_used"],
        "num_testcases": metrics["num_testcases"],
        "num_passing": metrics["num_passing"],
        "num_compile_error": statuses.count("COMPILE_ERROR"),
        "num_runtime_error": statuses.count("RUNTIME_ERROR"),
        "num_infra_error": statuses.count("INFRA_ERROR"),
        "redundancy_rate": metrics["redundancy_rate"],
        "llm_calls": metrics["llm_calls"],
        "symbolic_attempts": metrics["symbolic_attempts"],
        "z3_calls": metrics["z3_calls"],
        "z3_solved": metrics["z3_solved"],
        "z3_unsat": metrics["z3_unsat"],
        "z3_unsupported": metrics["z3_unsupported"],
        "z3_timeout": metrics["z3_timeout"],
        "z3_error": metrics["z3_error"],
        "z3_solve_rate": metrics["z3_solve_rate"],
        "z3_time_used": metrics["z3_time_used"],
        "symbolic_time_used": metrics["symbolic_time_used"],
        "z3_binding_count": metrics["z3_binding_count"],
        "symbolic_assisted_tests": metrics["symbolic_assisted_tests"],
        "symbolic_assisted_passing": metrics["symbolic_assisted_passing"],
        "symbolic_target_reached": metrics["symbolic_target_reached"],
        "symbolic_useful_attempts": metrics["symbolic_useful_attempts"],
        "symbolic_useful_rate": metrics["symbolic_useful_rate"],
        "symbolic_assisted_new_statements": metrics[
            "symbolic_assisted_new_statements"
        ],
        "symbolic_assisted_new_branches": metrics[
            "symbolic_assisted_new_branches"
        ],
        "symbolic_new_statements": metrics["symbolic_new_statements"],
        "symbolic_new_branches": metrics["symbolic_new_branches"],
        "hybrid_assisted_new_statements": metrics[
            "hybrid_assisted_new_statements"
        ],
        "hybrid_assisted_new_branches": metrics[
            "hybrid_assisted_new_branches"
        ],
        "partial_to_passing_rate": metrics["partial_to_passing_rate"],
        "error": result.error or "",
    }


def configuration_hash(config) -> str:
    policy_hash = ""
    if config.policy_path is not None and config.policy_path.is_file():
        policy_hash = hashlib.sha256(config.policy_path.read_bytes()).hexdigest()
    canonical = {
        "strategy": config.strategy,
        "statement_target": config.statement_target,
        "branch_target": config.branch_target,
        "scheduler_mode": config.scheduler_mode,
        "policy_hash": policy_hash,
        "wall_time_minutes": config.wall_time_minutes,
        "max_llm_calls": config.max_llm_calls,
        "max_symbolic_attempts": config.max_symbolic_attempts,
        "max_test_executions": config.max_test_executions,
        "random_seed": config.random_seed,
        "use_nearest_seed": config.use_nearest_seed,
        "use_divergence_repair": config.use_divergence_repair,
        "use_symbolic_partial": config.use_symbolic_partial,
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_artifacts(
    result: HybridGenerationResult,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    function = _safe_name(result.coverage_model.function_name)
    stem = f"gen_{result.config.run_id}_{function}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    summary_path = out_dir / "summary.csv"

    result.write_json(json_path)
    row = result_row(result)
    _write_csv(csv_path, row, append=False)
    _write_csv(summary_path, row, append=True)
    return json_path, csv_path, summary_path


def append_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, row, append=True)


def _write_csv(path: Path, row: dict[str, object], *, append: bool) -> None:
    if append:
        _migrate_csv_schema(path)
    has_header = append and path.is_file() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def _migrate_csv_schema(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames == SUMMARY_COLUMNS:
            return
        rows = list(reader)

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=SUMMARY_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if attempt < 4:
                time.sleep(0.2)
                continue
            temporary.unlink(missing_ok=True)
            raise PermissionError(
                f"Cannot migrate {path}; close Excel or any program that has "
                "the CSV open, then resume the batch"
            ) from error


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return (normalized or "function")[:100]
