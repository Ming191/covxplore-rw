from __future__ import annotations

from dataclasses import asdict

from covxplore.generator import GenerationConfig, GenerationResult
from covxplore.generation.stop_reasons import StopReason
from covxplore.status import TestStatus

__all__ = ["GenerationConfig", "GenerationResult", "StopReason", "flat_row"]


def flat_row(result: GenerationResult) -> dict:
    """Flatten one run for CSV export and router training."""
    suite = result.suite
    metrics = suite.coverage.metrics(suite.tests)
    total_tokens = result.total_input_tokens + result.total_output_tokens
    metadata = {
        f"experiment_{key}": value for key, value in asdict(result.experiment).items()
    }
    return {
        "run_id": result.config.run_id,
        "function_path": result.config.function_path,
        "prompt_variant": result.config.prompt_variant,
        "context_version": result.config.context_version,
        **metadata,
        "stop_reason": result.stop_reason,
        "statement_coverage_pct": round(metrics.statement_pct, 4),
        "branch_coverage_pct": round(metrics.branch_pct, 4),
        "covered_statements": result.covered_statements,
        "total_statements": metrics.total_statements,
        "covered_branches": result.covered_branches,
        "total_branches": metrics.total_branches,
        "redundancy_rate": result.redundancy_rate,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "total_tokens": total_tokens,
        "elapsed_sec": result.elapsed_sec,
        "batches_used": result.batches_used,
        "accepted_test_count": result.accepted_test_count,
        "passing_test_count": sum(
            1 for test in suite.tests if test.status == TestStatus.PASSED.value
        ),
        "rejected_candidate_count": result.rejected_candidate_count,
        "candidate_count": result.candidate_count,
        "tokens_per_batch": round(total_tokens / max(result.batches_used, 1), 2),
        "tokens_per_candidate": round(total_tokens / max(result.candidate_count, 1), 2),
        "error": result.error_message or "",
    }
