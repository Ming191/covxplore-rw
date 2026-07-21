from __future__ import annotations

from dataclasses import dataclass

from covxplore.hybrid.coverage import ExactCoverageState
from covxplore.hybrid.models import (
    BranchEdge,
    ExecutionResult,
    Seed,
    Statement,
    Target,
    TargetKind,
    TestBinding,
)


def longest_common_prefix(left: list[str], right: list[str]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        count += 1
    return count


@dataclass(frozen=True)
class SuccessfulSeed:
    intent_id: str | None
    bindings: tuple[TestBinding, ...]
    result: ExecutionResult

    def to_contract(self) -> Seed:
        return Seed(
            intent_id=self.intent_id,
            test_name=self.result.test_name,
            visited_statement_keys=self.result.visited_statement_keys,
            visited_branch_keys=self.result.visited_branch_keys,
            ordered_trace=self.result.ordered_trace,
            bindings=list(self.bindings),
        )


@dataclass(frozen=True)
class TargetSelection:
    target: Target
    expected_path: tuple[str, ...]
    path_depth: int
    loop_depth: int
    seed: Seed | None


class TargetSelector:
    def select(
        self,
        state: ExactCoverageState,
        seeds: list[SuccessfulSeed],
        excluded_keys: set[str] | None = None,
        *,
        include_branches: bool = True,
        include_statements: bool = True,
    ) -> TargetSelection | None:
        excluded = excluded_keys or set()
        branch_choice = (
            self._select_branch(state, seeds, excluded)
            if include_branches
            else None
        )
        if branch_choice is not None:
            branch, seed = branch_choice
            path = branch.expected_path
            return TargetSelection(
                target=Target(
                    kind=TargetKind.BRANCH_EDGE,
                    key=branch.key,
                    desired_outcome=self._bool_outcome(branch.outcome),
                ),
                expected_path=tuple(path),
                path_depth=branch.path_depth,
                loop_depth=branch.loop_depth,
                seed=seed,
            )

        statement_choice = (
            self._select_statement(state, seeds, excluded)
            if include_statements
            else None
        )
        if statement_choice is None:
            return None
        statement, seed = statement_choice
        path = statement.expected_path
        return TargetSelection(
            target=Target(kind=TargetKind.STATEMENT, key=statement.key),
            expected_path=tuple(path),
            path_depth=statement.path_depth,
            loop_depth=statement.loop_depth,
            seed=seed,
        )

    @staticmethod
    def _bool_outcome(outcome: str) -> bool | None:
        normalized = outcome.upper()
        if normalized == "TRUE":
            return True
        if normalized == "FALSE":
            return False
        return None

    @staticmethod
    def _select_branch(
        state: ExactCoverageState,
        seeds: list[SuccessfulSeed],
        excluded_keys: set[str],
    ) -> tuple[BranchEdge, Seed | None] | None:
        candidates = [
            edge
            for edge in state.model.branch_edges
            if edge.key in state.uncovered_branch_keys and edge.key not in excluded_keys
        ]
        if not candidates:
            return None
        uncovered = state.uncovered_statement_keys

        def rank(edge: BranchEdge) -> tuple[int, int, int, str]:
            _, prefix = TargetSelector._nearest_seed(edge.expected_path, seeds)
            return (
                -len(set(edge.downstream_statement_keys) & uncovered),
                -prefix,
                edge.path_depth,
                edge.key,
            )

        candidates.sort(key=rank)
        selected = candidates[0]
        seed, _ = TargetSelector._nearest_seed(selected.expected_path, seeds)
        return selected, seed

    @staticmethod
    def _select_statement(
        state: ExactCoverageState,
        seeds: list[SuccessfulSeed],
        excluded_keys: set[str],
    ) -> tuple[Statement, Seed | None] | None:
        candidates = [
            statement
            for statement in state.model.statements
            if statement.key in state.uncovered_statement_keys
            and statement.key not in excluded_keys
        ]
        if not candidates:
            return None

        def rank(statement: Statement) -> tuple[int, int, str]:
            _, prefix = TargetSelector._nearest_seed(statement.expected_path, seeds)
            return (-prefix, statement.path_depth, statement.key)

        candidates.sort(key=rank)
        selected = candidates[0]
        seed, _ = TargetSelector._nearest_seed(selected.expected_path, seeds)
        return selected, seed

    @staticmethod
    def _nearest_seed(
        path: list[str], seeds: list[SuccessfulSeed]
    ) -> tuple[Seed | None, int]:
        if not seeds:
            return None, 0

        def seed_key(seed: SuccessfulSeed) -> tuple[int, str]:
            trace = [step.key for step in seed.result.ordered_trace]
            return (-longest_common_prefix(path, trace), seed.result.test_name)

        selected = min(seeds, key=seed_key)
        trace = [step.key for step in selected.result.ordered_trace]
        return selected.to_contract(), longest_common_prefix(path, trace)
