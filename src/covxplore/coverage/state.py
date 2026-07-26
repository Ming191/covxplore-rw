from __future__ import annotations

from dataclasses import dataclass, field

from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.test_result import TestResult
from covxplore.types.coverage_gap_input import CoverageGapInput
from covxplore.types.coverage_metrics import CoverageMetrics
from covxplore.status import TestStatus, normalize_test_status


@dataclass
class CoverageState:
    """Accumulate coverage metrics from executed test results.

    Public interface::

        cs = CoverageState()
        cs.add_result(result, prior_results=tests, min_suite_size=3)
        m = cs.metrics(tests)           # -> CoverageMetrics
        gi = cs.gap_input(tests, batch_count)  # -> CoverageGapInput

    Internal fields (prefixed ``_``) are for :meth:`add_result` and DTO
    producers only; external callers should use :meth:`metrics`,
    :meth:`gap_input`, and the public properties.
    """

    # --- internal accumulation (private) ---
    _cumulative_uncovered_stmt_ids: set[int] | None = None
    _cumulative_uncovered_branch_keys: set[tuple[int, bool]] | None = None
    _stmt_node_info: dict[int, UnvisitedStatement] = field(default_factory=dict)
    _branch_node_info: dict[int, UnvisitedBranch] = field(default_factory=dict)
    _total_statements: int = 0
    _total_branches: int = 0
    _best_statement_visited: int = 0
    _best_branch_visited: int = 0
    consecutive_redundant: int = 0
    """Counter of back-to-back redundant tests; reset on progress."""

    def add_result(
        self,
        result: TestResult,
        *,
        prior_results: list[TestResult],
        min_suite_size: int = 3,
    ) -> None:
        """Update cumulative coverage from one result."""
        normalized = normalize_test_status(result.status)
        if normalized not in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}:
            return

        result.new_structural_coverage = self.structural_gain(result)
        result.is_redundant = (
            result.new_structural_coverage == 0 and len(prior_results) >= min_suite_size
        )
        self.consecutive_redundant = self.consecutive_redundant + 1 if result.is_redundant else 0

        self._update_totals(result)
        self._update_statement_intersection(result)
        self._update_branch_intersection(result)

    def metrics(self, tests: list[TestResult]) -> CoverageMetrics:
        """Return frozen summary suitable for serialisation and reporting."""
        covered_stmts = self._covered_statements(tests)
        covered_brs = self._covered_branches(tests)
        return CoverageMetrics(
            statement_pct=self._statement_pct(tests),
            branch_pct=self._branch_pct(tests),
            covered_statements=covered_stmts,
            total_statements=self._total_statements,
            covered_branches=covered_brs,
            total_branches=self._total_branches,
        )

    def gap_input(
        self, tests: list[TestResult], batch_count: int
    ) -> CoverageGapInput:
        """Produce the DTO consumed by :class:`GapAnalyzer`."""
        return CoverageGapInput(
            tests=tests,
            metrics=self.metrics(tests),
            cumulative_unvisited_statements=self._cumulative_unvisited_stmts(),
            cumulative_unvisited_branches=self._cumulative_unvisited_brs(),
            consecutive_redundant=self.consecutive_redundant,
            batch_count=batch_count,
        )

    def is_branch_satisfied(self, node_id: int, polarity: bool) -> bool | None:
        """Whether ``node_id``'s ``polarity`` side is already covered by the suite so far.

        Returns ``None`` when branch coverage hasn't been established yet (no accepted result
        has carried node-id-bearing ``unvisited_branches``), so callers can treat "unknown" as
        "not confirmed covered" rather than silently skipping a possibly-still-needed candidate.

        Returns ``False`` for a ``node_id`` never seen in suite coverage data (e.g. agent
        confused ``line_in_function`` with CFG ``nodeId``) — never treat phantom IDs as covered.
        """
        if self._cumulative_uncovered_branch_keys is None:
            return None
        if (node_id, polarity) in self._cumulative_uncovered_branch_keys:
            return False
        known = (
            node_id in self._branch_node_info
            or (node_id, True) in self._cumulative_uncovered_branch_keys
            or (node_id, False) in self._cumulative_uncovered_branch_keys
        )
        if not known:
            return False
        return True

    def structural_gain(self, result: TestResult) -> int:
        """Count newly covered statements and branch sides.

        Node IDs provide exact cumulative set differences. AkaUT may omit IDs;
        then monotonic coverage counts are the safe fallback.
        """
        if normalize_test_status(result.status) not in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}:
            return 0
        return self._statement_gain(result) + self._branch_gain(result)

    def _statement_gain(self, result: TestResult) -> int:
        uncovered = self._uncovered_statement_ids(result)
        if uncovered is None:
            return max(result.statement_coverage.visited - self._best_statement_visited, 0)
        if self._cumulative_uncovered_stmt_ids is None:
            return max(result.statement_coverage.visited, 0)
        return len(self._cumulative_uncovered_stmt_ids - uncovered)

    def _branch_gain(self, result: TestResult) -> int:
        uncovered = self._uncovered_branch_keys(result)
        if uncovered is None:
            return max(result.branch_coverage.visited - self._best_branch_visited, 0)
        if self._cumulative_uncovered_branch_keys is None:
            return max(result.branch_coverage.visited, 0)
        return len(self._cumulative_uncovered_branch_keys - uncovered)

    @staticmethod
    def _uncovered_statement_ids(result: TestResult) -> set[int] | None:
        ids = {statement.node_id for statement in result.unvisited_statements if statement.node_id is not None}
        return ids if ids or not result.unvisited_statements else None

    @staticmethod
    def _uncovered_branch_keys(result: TestResult) -> set[tuple[int, bool]] | None:
        if result.unvisited_branches and not any(branch.node_id is not None for branch in result.unvisited_branches):
            return None
        return {
            (branch.node_id, side)
            for branch in result.unvisited_branches
            if branch.node_id is not None
            for side, visited in ((True, branch.true_visited), (False, branch.false_visited))
            if not visited
        }

    def _covered_statements(self, tests: list[TestResult]) -> int:
        if self._total_statements == 0:
            return 0
        best_single = 0
        for test in tests:
            normalized = normalize_test_status(test.status)
            if normalized in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}:
                best_single = max(best_single, test.statement_coverage.visited)
        derived = None
        if self._cumulative_uncovered_stmt_ids is not None:
            uncovered = len(self._cumulative_uncovered_stmt_ids)
            if uncovered <= self._total_statements:
                derived = self._total_statements - uncovered
        covered = best_single if derived is None else max(best_single, derived)
        return min(self._total_statements, max(0, covered))

    def _covered_branches(self, tests: list[TestResult]) -> int:
        if self._total_branches == 0:
            return 0
        best_single = 0
        for test in tests:
            normalized = normalize_test_status(test.status)
            if normalized in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}:
                best_single = max(best_single, test.branch_coverage.visited)
        derived = None
        if self._cumulative_uncovered_branch_keys is not None:
            uncovered = len(self._cumulative_uncovered_branch_keys)
            if uncovered <= self._total_branches:
                derived = self._total_branches - uncovered
        covered = best_single if derived is None else max(best_single, derived)
        return min(self._total_branches, max(0, covered))

    def _statement_pct(self, tests: list[TestResult]) -> float:
        if self._total_statements == 0:
            return 0.0
        return self._covered_statements(tests) / self._total_statements

    def _branch_pct(self, tests: list[TestResult]) -> float:
        if self._total_branches == 0:
            return 0.0
        return self._covered_branches(tests) / self._total_branches

    def _cumulative_unvisited_stmts(self) -> list[UnvisitedStatement]:
        if self._cumulative_uncovered_stmt_ids is None:
            return []
        return [
            self._stmt_node_info[node_id]
            for node_id in self._cumulative_uncovered_stmt_ids
            if node_id in self._stmt_node_info
        ]

    def _cumulative_unvisited_brs(self) -> list[UnvisitedBranch]:
        if self._cumulative_uncovered_branch_keys is None:
            return []
        node_missing: dict[int, tuple[bool, bool]] = {}
        for node_id, is_true_side in self._cumulative_uncovered_branch_keys:
            true_missing, false_missing = node_missing.get(node_id, (False, False))
            if is_true_side:
                node_missing[node_id] = (True, false_missing)
            else:
                node_missing[node_id] = (true_missing, True)
        result: list[UnvisitedBranch] = []
        for node_id, (true_missing, false_missing) in node_missing.items():
            if node_id in self._branch_node_info:
                info = self._branch_node_info[node_id]
                result.append(
                    UnvisitedBranch(
                        node_id=node_id,
                        condition=info.condition,
                        true_visited=not true_missing,
                        false_visited=not false_missing,
                        line_in_function=info.line_in_function,
                        start_offset=info.start_offset,
                        end_offset=info.end_offset,
                    )
                )
        return result

    def _update_totals(self, result: TestResult) -> None:
        if result.statement_coverage.total > 0:
            self._total_statements = result.statement_coverage.total
        if result.branch_coverage.total > 0:
            self._total_branches = result.branch_coverage.total
        self._best_statement_visited = max(self._best_statement_visited, result.statement_coverage.visited)
        self._best_branch_visited = max(self._best_branch_visited, result.branch_coverage.visited)

    def _update_statement_intersection(self, result: TestResult) -> None:
        uncovered = self._uncovered_statement_ids(result)
        if uncovered is None:
            return
        for statement in result.unvisited_statements:
            if statement.node_id is not None:
                self._stmt_node_info[statement.node_id] = statement
        if self._cumulative_uncovered_stmt_ids is None:
            self._cumulative_uncovered_stmt_ids = uncovered
        else:
            self._cumulative_uncovered_stmt_ids &= uncovered

    def _update_branch_intersection(self, result: TestResult) -> None:
        uncovered = self._uncovered_branch_keys(result)
        if uncovered is None:
            return
        for branch in result.unvisited_branches:
            if branch.node_id is not None:
                self._branch_node_info[branch.node_id] = branch
        if self._cumulative_uncovered_branch_keys is None:
            self._cumulative_uncovered_branch_keys = uncovered
        else:
            self._cumulative_uncovered_branch_keys &= uncovered
