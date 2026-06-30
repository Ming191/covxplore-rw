from __future__ import annotations

from covxplore.generator import GenerationConfig, GenerationResult, StopReason
from covxplore.status import TestStatus

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "StopReason",
    "flat_row",
]


def flat_row(result: GenerationResult) -> dict:
    """One-row dict for CSV / DataFrame export."""
    suite = result.suite
    metrics = suite.coverage.metrics(suite.tests)
    return {
        "run_id": result.config.run_id,
        "function_path": result.config.function_path,
        "prompt_variant": result.config.prompt_variant,
        "stop_reason": result.stop_reason,
        "statement_coverage_pct": round(metrics.statement_pct, 4),
        "branch_coverage_pct": round(metrics.branch_pct, 4),
        "mcdc_coverage_pct": result.final_mcdc_pct,
        "covered_statements": result.covered_statements,
        "total_statements": metrics.total_statements,
        "covered_branches": result.covered_branches,
        "total_branches": metrics.total_branches,
        "covered_mcdc_pairs": metrics.covered_mcdc_pairs,
        "total_mcdc_pairs": metrics.total_mcdc_pairs,
        "redundancy_rate": result.redundancy_rate,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "total_tokens": result.total_input_tokens + result.total_output_tokens,
        "tracing_url": result.tracing_url or "",
        "elapsed_sec": result.elapsed_sec,
        "iterations_used": result.iterations_used,
        "num_tests": len(suite.tests),
        "num_passing": sum(
            1 for t in suite.tests if t.status == TestStatus.PASSED.value
        ),
        "num_redundant": sum(1 for t in suite.tests if t.is_redundant),
        "error": result.error_message or "",
    }
