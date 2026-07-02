from __future__ import annotations

from typing import Literal


StopReason = Literal[
    "max_iter",
    "coverage_target",
    "redundant_streak",
    "fail_streak",
    "agent_done",
    "error",
]


def coverage_target_reached(suite, config) -> bool:
    """Return True when all applicable coverage targets are satisfied."""
    if suite.iteration_count == 0:
        return False
    metrics = suite.coverage.metrics(suite.tests)
    mcdc_done = (
        metrics.total_mcdc_pairs == 0
        or metrics.mcdc_pct >= config.mcdc_target
        or not suite.coverage.unvisited_summary()
    )
    stmt_done = (
        metrics.total_statements == 0
        or metrics.covered_statements >= metrics.total_statements
    )
    branch_done = (
        metrics.total_branches == 0 or metrics.covered_branches >= metrics.total_branches
    )
    return mcdc_done and stmt_done and branch_done


def infer_stop(suite, config) -> StopReason:
    if coverage_target_reached(suite, config):
        return "coverage_target"
    if suite.coverage.consecutive_redundant >= config.redundant_streak_limit:
        return "redundant_streak"
    fail_limit: int = getattr(config, "fail_streak_limit", 3)
    if suite.consecutive_failures() >= fail_limit:
        return "fail_streak"
    if suite.iteration_count >= config.max_iterations:
        return "max_iter"
    return "agent_done"
