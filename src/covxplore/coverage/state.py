from __future__ import annotations

from dataclasses import dataclass, field

from covxplore.types.condition_key import ConditionKey
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.test_result import TestResult
from covxplore.types.coverage_gap_input import CoverageGapInput
from covxplore.types.coverage_metrics import CoverageMetrics
from covxplore.types.mcdc_obligation import McdcObligation
from covxplore.status import TestStatus, normalize_test_status


@dataclass
class CoverageState:
    """Accumulate coverage metrics from executed test results.

    Public interface::

        cs = CoverageState()
        cs.add_result(result, prior_results=tests, min_suite_size=3)
        m = cs.metrics(tests)           # -> CoverageMetrics
        gi = cs.gap_input(tests, batch_count)  # -> CoverageGapInput
        cs.seed_conditions(...)

    Internal fields (prefixed ``_``) are for :meth:`add_result` and DTO
    producers only; external callers should use :meth:`metrics`,
    :meth:`gap_input`, and the public properties.
    """

    # --- internal accumulation (private) ---
    _covered_keys: set[ConditionKey] = field(default_factory=set)
    _condition_id_to_text: dict[int, str] = field(default_factory=dict)
    _condition_id_to_line: dict[int, int | None] = field(default_factory=dict)
    _all_conditions: list[str] = field(default_factory=list)
    _cumulative_uncovered_stmt_ids: set[int] | None = None
    _cumulative_uncovered_branch_keys: set[tuple[int, bool]] | None = None
    _stmt_node_info: dict[int, UnvisitedStatement] = field(default_factory=dict)
    _branch_node_info: dict[int, UnvisitedBranch] = field(default_factory=dict)
    _total_statements: int = 0
    _total_branches: int = 0

    total_mcdc_pairs: int = 0
    """Total MC/DC pairs to cover (0 means no MC/DC data).
    Set via :meth:`seed_conditions` or discovered from test results."""

    consecutive_redundant: int = 0
    """Counter of back-to-back redundant tests; reset on progress."""

    @property
    def has_mcdc(self) -> bool:
        """Whether the function under test has MC/DC conditions."""
        return self.total_mcdc_pairs > 0

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

        raw_keys = result.condition_keys()
        new_keys = raw_keys - self._covered_keys
        result.new_mcdc_pairs_covered = len(new_keys)
        result.is_redundant = (
            self.total_mcdc_pairs > 0
            and len(new_keys) == 0
            and len(prior_results) >= min_suite_size
        )
        self._covered_keys |= new_keys

        if result.is_redundant:
            self.consecutive_redundant += 1
        else:
            self.consecutive_redundant = 0

        self._discover_conditions(result)
        self._update_totals(result)
        self._update_statement_intersection(result)
        self._update_branch_intersection(result)

    def seed_conditions(self, conditions, total_mcdc_pairs: int | None = None) -> None:
        """Pre-populate condition metadata from static CFG before generation."""
        if total_mcdc_pairs is not None:
            self.total_mcdc_pairs = total_mcdc_pairs
        self._all_conditions = [c.condition for c in conditions]
        for condition in conditions:
            if condition.node_id is None:
                raise RuntimeError(
                    "Backend payload missing nodeId in /api/node/conditions. "
                    "nodeId is required for MC/DC identity."
                )
            condition_id = condition.node_id
            self._condition_id_to_text[condition_id] = condition.condition
            self._condition_id_to_line[condition_id] = condition.line_in_function

    def metrics(self, tests: list[TestResult]) -> CoverageMetrics:
        """Return frozen summary suitable for serialisation and reporting."""
        covered_stmts = self._covered_statements(tests)
        covered_brs = self._covered_branches(tests)
        return CoverageMetrics(
            statement_pct=self._statement_pct(tests),
            branch_pct=self._branch_pct(tests),
            mcdc_pct=self._mcdc_pct(),
            covered_statements=covered_stmts,
            total_statements=self._total_statements,
            covered_branches=covered_brs,
            total_branches=self._total_branches,
            covered_mcdc_pairs=len(self._covered_keys),
            total_mcdc_pairs=self.total_mcdc_pairs,
        )

    def gap_input(
        self, tests: list[TestResult], batch_count: int
    ) -> CoverageGapInput:
        """Produce the DTO consumed by :class:`GapAnalyzer`."""
        obligations = [
            McdcObligation(
                condition_id=item["condition_id"],
                condition=item["condition"],
                line_in_function=item["line_in_function"],
                needs_true=item["needs_true"],
                needs_false=item["needs_false"],
            )
            for item in self.unvisited_summary()
        ]
        return CoverageGapInput(
            tests=tests,
            metrics=self.metrics(tests),
            obligations=obligations,
            observed_polarity_counts=self._observed_polarity_counts(tests),
            cumulative_unvisited_statements=self._cumulative_unvisited_stmts(),
            cumulative_unvisited_branches=self._cumulative_unvisited_brs(),
            all_conditions_count=len(self._all_conditions),
            unique_condition_ids=len(self._condition_id_to_text),
            consecutive_redundant=self.consecutive_redundant,
            batch_count=batch_count,
        )

    def _mcdc_pct(self) -> float:
        if self.total_mcdc_pairs == 0:
            return 0.0
        return len(self._covered_keys) / self.total_mcdc_pairs

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

    def _discover_conditions(self, result: TestResult) -> None:
        discovered_ids: dict[int, str] = {}
        if result.condition_trace:
            for entry in result.condition_trace:
                condition_id = entry.identity()
                discovered_ids[condition_id] = entry.condition.strip()
                self._condition_id_to_line[condition_id] = entry.line_in_function
        if result.unvisited_mcdc:
            for entry in result.unvisited_mcdc:
                discovered_ids[entry.identity()] = entry.condition.strip()

        for condition_id, condition_text in discovered_ids.items():
            if condition_id not in self._condition_id_to_text:
                self._condition_id_to_text[condition_id] = condition_text

        self._all_conditions = [
            self._condition_id_to_text[cid] for cid in self._condition_id_to_text
        ]

    def _update_totals(self, result: TestResult) -> None:
        if self.total_mcdc_pairs == 0 and result.mcdc_coverage.total > 0:
            self.total_mcdc_pairs = result.mcdc_coverage.total
        if result.statement_coverage.total > 0:
            self._total_statements = result.statement_coverage.total
        if result.branch_coverage.total > 0:
            self._total_branches = result.branch_coverage.total

    def _update_statement_intersection(self, result: TestResult) -> None:
        test_uncovered_stmt_ids: set[int] = set()
        for statement in result.unvisited_statements:
            if statement.node_id is not None:
                test_uncovered_stmt_ids.add(statement.node_id)
                self._stmt_node_info[statement.node_id] = statement
        if self._cumulative_uncovered_stmt_ids is None:
            self._cumulative_uncovered_stmt_ids = test_uncovered_stmt_ids
        else:
            self._cumulative_uncovered_stmt_ids &= test_uncovered_stmt_ids

    def _update_branch_intersection(self, result: TestResult) -> None:
        test_uncovered_branch_keys: set[tuple[int, bool]] = set()
        for branch in result.unvisited_branches:
            if branch.node_id is None:
                continue
            if not branch.true_visited:
                test_uncovered_branch_keys.add((branch.node_id, True))
            if not branch.false_visited:
                test_uncovered_branch_keys.add((branch.node_id, False))
            self._branch_node_info[branch.node_id] = branch
        if self._cumulative_uncovered_branch_keys is None:
            self._cumulative_uncovered_branch_keys = test_uncovered_branch_keys
        else:
            self._cumulative_uncovered_branch_keys &= test_uncovered_branch_keys

    # ------------------------------------------------------------------
    # Public read-only helpers (still used by generator/ablation appraisal)
    # ------------------------------------------------------------------

    def unvisited_summary(self) -> list[dict]:
        result = []
        for condition_id, condition in self._condition_id_to_text.items():
            key_true = ConditionKey(condition_id, True)
            key_false = ConditionKey(condition_id, False)
            needs_true = key_true not in self._covered_keys
            needs_false = key_false not in self._covered_keys
            if needs_true or needs_false:
                result.append(
                    {
                        "condition_id": condition_id,
                        "condition": condition,
                        "line_in_function": self._condition_id_to_line.get(condition_id),
                        "needs_true": needs_true,
                        "needs_false": needs_false,
                    }
                )
        result.sort(
            key=lambda item: (
                item["line_in_function"] is None,
                item["line_in_function"]
                if item["line_in_function"] is not None
                else float("inf"),
                item["condition_id"],
            )
        )
        return result

    def _observed_polarity_counts(
        self, tests: list[TestResult]
    ) -> dict[ConditionKey, int]:
        counts: dict[ConditionKey, int] = {}
        for test in tests:
            if not test.condition_trace:
                continue
            for entry in test.condition_trace:
                condition_id = entry.identity()
                if entry.true_branch_visited:
                    key = ConditionKey(condition_id, True)
                    counts[key] = counts.get(key, 0) + 1
                if entry.false_branch_visited:
                    key = ConditionKey(condition_id, False)
                    counts[key] = counts.get(key, 0) + 1
        return counts
