from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


StopReason = Literal[
    "max_batches",
    "coverage_target",
    "redundant_streak",
    "fail_streak",
    "agent_done",
    "guardrail_incomplete",
    "error",
]


@dataclass(frozen=True)
class StopPolicy:
    max_batches: int
    redundant_streak_limit: int
    fail_streak_limit: int

    @classmethod
    def from_config(cls, config) -> "StopPolicy":
        return cls(
            max_batches=int(getattr(config, "max_batches", 10**9)),
            redundant_streak_limit=int(getattr(config, "redundant_streak_limit", 3)),
            fail_streak_limit=int(getattr(config, "fail_streak_limit", 3)),
        )

    def coverage_target_reached(
        self,
        suite,
        *,
        require_structural_totals: bool = False,
    ) -> bool:
        if suite.batch_count == 0:
            return False
        metrics = suite.coverage.metrics(suite.tests)
        if require_structural_totals and not suite.coverage._structural_totals_known:
            return False
        statement_done = metrics.total_statements == 0 or metrics.covered_statements >= metrics.total_statements
        branch_done = metrics.total_branches == 0 or metrics.covered_branches >= metrics.total_branches
        return statement_done and branch_done

    def hard_stop_reason(self, suite) -> StopReason | None:
        if suite.coverage.consecutive_redundant >= self.redundant_streak_limit:
            return "redundant_streak"
        if suite.consecutive_failures() >= self.fail_streak_limit:
            return "fail_streak"
        if self.coverage_target_reached(suite, require_structural_totals=True):
            return "coverage_target"
        return None

    def terminal_reason(self, suite) -> StopReason:
        if self.coverage_target_reached(suite, require_structural_totals=True):
            return "coverage_target"
        if suite.coverage.consecutive_redundant >= self.redundant_streak_limit:
            return "redundant_streak"
        if suite.consecutive_failures() >= self.fail_streak_limit:
            return "fail_streak"
        if suite.batch_count >= self.max_batches:
            return "max_batches"
        return "agent_done"

    def hard_stop_message(self, reason: StopReason) -> str:
        if reason == "redundant_streak":
            return (
                f"Hard stop: {self.redundant_streak_limit} consecutive redundant tests — "
                "agent loop terminated."
            )
        if reason == "fail_streak":
            return (
                f"Hard stop: {self.fail_streak_limit} consecutive failing batches — "
                "agent loop terminated."
            )
        if reason == "coverage_target":
            return "Hard stop: all coverage targets met — agent loop terminated."
        return f"Hard stop: {reason}"
