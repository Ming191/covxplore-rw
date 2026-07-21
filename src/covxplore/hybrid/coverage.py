from __future__ import annotations

from dataclasses import dataclass, field

from covxplore.hybrid.models import CoverageModel, ExecutionResult


@dataclass(frozen=True)
class CoverageDelta:
    new_statement_keys: frozenset[str] = frozenset()
    new_branch_keys: frozenset[str] = frozenset()
    accepted: bool = False

    @property
    def redundant(self) -> bool:
        return self.accepted and not self.new_statement_keys and not self.new_branch_keys


@dataclass
class ExactCoverageState:
    """Suite coverage represented only by stable keys returned by AkaUT."""

    model: CoverageModel
    covered_statement_keys: set[str] = field(default_factory=set)
    covered_branch_keys: set[str] = field(default_factory=set)

    def add(self, result: ExecutionResult) -> CoverageDelta:
        if not self._accepts_coverage(result):
            return CoverageDelta(accepted=False)

        valid_statements = set(result.visited_statement_keys) & self.model.statement_keys
        valid_branches = set(result.visited_branch_keys) & self.model.branch_keys
        new_statements = valid_statements - self.covered_statement_keys
        new_branches = valid_branches - self.covered_branch_keys
        self.covered_statement_keys.update(valid_statements)
        self.covered_branch_keys.update(valid_branches)
        return CoverageDelta(
            new_statement_keys=frozenset(new_statements),
            new_branch_keys=frozenset(new_branches),
            accepted=True,
        )

    @staticmethod
    def _accepts_coverage(result: ExecutionResult) -> bool:
        status = result.status.upper()
        if status == "PASSED":
            return True
        if status == "RUNTIME_ERROR":
            return bool(result.visited_statement_keys or result.visited_branch_keys)
        return False

    @property
    def uncovered_statement_keys(self) -> set[str]:
        return self.model.statement_keys - self.covered_statement_keys

    @property
    def uncovered_branch_keys(self) -> set[str]:
        return self.model.branch_keys - self.covered_branch_keys

    @staticmethod
    def _fraction(covered: int, total: int) -> float:
        return 1.0 if total == 0 else covered / total

    @property
    def statement_fraction(self) -> float:
        return self._fraction(
            len(self.covered_statement_keys), len(self.model.statement_keys)
        )

    @property
    def branch_fraction(self) -> float:
        return self._fraction(len(self.covered_branch_keys), len(self.model.branch_keys))

    def reached(self, statement_target: float, branch_target: float) -> bool:
        return (
            self.statement_fraction >= statement_target
            and self.branch_fraction >= branch_target
        )

    def summary(self) -> dict[str, float | int]:
        return {
            "statement_cov": self.statement_fraction,
            "branch_cov": self.branch_fraction,
            "covered_statement": len(self.covered_statement_keys),
            "total_statement": len(self.model.statement_keys),
            "covered_branch": len(self.covered_branch_keys),
            "total_branch": len(self.model.branch_keys),
        }
