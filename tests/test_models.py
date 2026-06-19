"""Tests for covxplore.models — pure data model, no network, no secrets."""

import pytest

from covxplore.types import (
    ConditionKey,
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TestSuite,
    UnvisitedBranch,
    UnvisitedMcdc,
    UnvisitedStatement,
)
from covxplore.status import TestStatus


LEGACY_TEST_SUITE_APIS = {
    "covered_keys",
    "total_mcdc_conditions",
    "all_conditions",
    "condition_id_to_text",
    "condition_id_to_line",
    "cumulative_uncovered_stmt_ids",
    "cumulative_uncovered_branch_keys",
    "_stmt_node_info",
    "_branch_node_info",
    "total_statements",
    "total_branches",
    "consecutive_redundant",
    "_coverage_state",
    "_sync_coverage_state",
    "unvisited_summary",
    "_observed_polarity_counts",
    "coverage_gap_prompt_fragment",
    "mcdc_coverage_pct",
    "statement_coverage_pct",
    "branch_coverage_pct",
    "covered_statements",
    "covered_branches",
    "cumulative_unvisited_statements",
    "cumulative_unvisited_branches",
}


# ------------------------------------------------------------------ #
# ConditionKey
# ------------------------------------------------------------------ #
class TestConditionKey:
    def test_construction(self):
        key = ConditionKey(42, True)
        assert key.condition_id == 42
        assert key.polarity is True

    def test_equality(self):
        a = ConditionKey(1, True)
        b = ConditionKey(1, True)
        c = ConditionKey(1, False)
        assert a == b
        assert a != c

    def test_hashable(self):
        s = {ConditionKey(1, True), ConditionKey(2, False)}
        assert ConditionKey(1, True) in s
        assert ConditionKey(3, True) not in s


# ------------------------------------------------------------------ #
# ConditionTraceEntry
# ------------------------------------------------------------------ #
class TestConditionTraceEntry:
    def test_identity_requires_node_id(self):
        entry = ConditionTraceEntry(condition="a && b", true_branch_visited=True, false_branch_visited=False)
        with pytest.raises(RuntimeError, match="nodeId"):
            entry.identity()

    def test_identity_with_node_id(self):
        entry = ConditionTraceEntry(node_id=7, condition="x", true_branch_visited=True, false_branch_visited=False)
        assert entry.identity() == 7

    def test_none_node_id_raises(self):
        entry = ConditionTraceEntry(node_id=None, condition="c", true_branch_visited=False, false_branch_visited=False)
        with pytest.raises(RuntimeError, match="nodeId"):
            entry.identity()


# ------------------------------------------------------------------ #
# UnvisitedMcdc
# ------------------------------------------------------------------ #
class TestUnvisitedMcdc:
    def test_identity_with_node_id(self):
        u = UnvisitedMcdc(node_id=10, condition="x > 0", true_branch_visited=False, false_branch_visited=True)
        assert u.identity() == 10

    def test_none_node_id_raises(self):
        u = UnvisitedMcdc(node_id=None, condition="x", true_branch_visited=False, false_branch_visited=False)
        with pytest.raises(RuntimeError, match="nodeId"):
            u.identity()


# ------------------------------------------------------------------ #
# TestResult.condition_keys
# ------------------------------------------------------------------ #
class TestResultConditionKeys:
    def test_empty_traces(self):
        tr = TestResult(test_name="t1", test_body="void test() {}", status="PASSED")
        assert tr.condition_keys() == set()

    def test_from_condition_trace(self):
        tr = TestResult(
            test_name="t2",
            test_body="void test() {}",
            status="PASSED",
            condition_trace=[
                ConditionTraceEntry(
                    node_id=1, condition="a", true_branch_visited=True, false_branch_visited=False
                ),
                ConditionTraceEntry(
                    node_id=2, condition="b", true_branch_visited=False, false_branch_visited=True
                ),
                ConditionTraceEntry(
                    node_id=3, condition="c", true_branch_visited=True, false_branch_visited=True
                ),
            ],
        )
        keys = tr.condition_keys()
        expected = {
            ConditionKey(1, True),
            ConditionKey(2, False),
            ConditionKey(3, True),
            ConditionKey(3, False),
        }
        assert keys == expected

    def test_from_unvisited_mcdc(self):
        tr = TestResult(
            test_name="t3",
            test_body="void test() {}",
            status="PASSED",
            unvisited_mcdc=[
                UnvisitedMcdc(node_id=5, condition="d", true_branch_visited=True, false_branch_visited=False),
                UnvisitedMcdc(node_id=6, condition="e", true_branch_visited=False, false_branch_visited=True),
            ],
        )
        keys = tr.condition_keys()
        expected = {ConditionKey(5, True), ConditionKey(6, False)}
        assert keys == expected

    def test_merged_from_both(self):
        """Keys from both condition_trace and unvisited_mcdc are merged."""
        tr = TestResult(
            test_name="t4",
            test_body="void test() {}",
            status="PASSED",
            condition_trace=[
                ConditionTraceEntry(
                    node_id=1, condition="a", true_branch_visited=True, false_branch_visited=False
                ),
            ],
            unvisited_mcdc=[
                UnvisitedMcdc(node_id=2, condition="b", true_branch_visited=False, false_branch_visited=True),
            ],
        )
        keys = tr.condition_keys()
        expected = {ConditionKey(1, True), ConditionKey(2, False)}
        assert keys == expected

    def test_no_duplicates(self):
        """Same condition from both sources should not duplicate."""
        tr = TestResult(
            test_name="t5",
            test_body="void test() {}",
            status="PASSED",
            condition_trace=[
                ConditionTraceEntry(
                    node_id=1, condition="a", true_branch_visited=True, false_branch_visited=False
                ),
            ],
            unvisited_mcdc=[
                UnvisitedMcdc(node_id=1, condition="a", true_branch_visited=True, false_branch_visited=False),
            ],
        )
        keys = tr.condition_keys()
        assert keys == {ConditionKey(1, True)}


# ------------------------------------------------------------------ #
# TestSuite.add_result
# ------------------------------------------------------------------ #
class TestSuiteAddResult:
    def _make_result(self, name, status="PASSED", node_id=1, polarity=True):
        return TestResult(
            test_name=name,
            test_body="void test() {}",
            status=status,
            condition_trace=[
                ConditionTraceEntry(
                    node_id=node_id,
                    condition="cond",
                    true_branch_visited=polarity,
                    false_branch_visited=not polarity,
                ),
            ],
            mcdc_coverage=CoverageDetail(visited=0, total=2, progress=0.0),
        )

    def test_first_result_added(self):
        suite = TestSuite(function_path="/f.cpp::foo()")
        r = self._make_result("t1", node_id=1, polarity=True)
        suite.add_result(r)

        assert len(suite.tests) == 1
        assert suite.iteration_count == 1
        assert ConditionKey(1, True) in suite.coverage._covered_keys

    def test_new_keys_accumulate(self):
        suite = TestSuite(function_path="/f.cpp::foo()")
        suite.coverage.total_mcdc_pairs = 2
        r1 = self._make_result("t1", node_id=1, polarity=True)
        r2 = self._make_result("t2", node_id=2, polarity=False)

        suite.add_result(r1)
        suite.add_result(r2)

        assert len(suite.coverage._covered_keys) == 2
        assert ConditionKey(1, True) in suite.coverage._covered_keys
        assert ConditionKey(2, False) in suite.coverage._covered_keys

    def test_new_mcdc_pairs_tracked(self):
        suite = TestSuite(function_path="/f.cpp::foo()")
        suite.coverage.total_mcdc_pairs = 2
        r1 = self._make_result("t1", node_id=1, polarity=True)
        r2 = self._make_result("t2", node_id=2, polarity=False)

        suite.add_result(r1)
        assert r1.new_mcdc_pairs_covered == 1

        suite.add_result(r2)
        assert r2.new_mcdc_pairs_covered == 1

    def test_redundant_flag(self):
        """add_result checks len(self.tests) >= min_suite_size BEFORE append,
        so the 4th test with same coverage at min_suite_size=3 is redundant."""
        suite = TestSuite(function_path="/f.cpp::foo()")
        suite.coverage.total_mcdc_pairs = 1
        r1 = self._make_result("t1", node_id=1, polarity=True)
        r2 = self._make_result("t2", node_id=1, polarity=True)  # same coverage
        r3 = self._make_result("t3", node_id=1, polarity=True)  # still same

        suite.add_result(r1)
        suite.add_result(r2)
        suite.add_result(r3)

        assert not r1.is_redundant
        assert not r2.is_redundant
        # r3: len(tests) before append = 2 < min_suite_size(3) → not redundant
        assert not r3.is_redundant

        # 4th test with same key: len(tests)=3 ≥ min_suite_size, new_keys=0 → redundant
        r4 = self._make_result("t4", node_id=1, polarity=True)
        suite.add_result(r4)
        assert r4.is_redundant

    def test_redundant_with_passed_runtime_error(self):
        """Tests with PASSED or RUNTIME_ERROR status are processed.
        Redundancy only kicks in when len(tests) ≥ min_suite_size (default 3)."""
        suite = TestSuite(function_path="/f.cpp::foo()")
        suite.coverage.total_mcdc_pairs = 1
        r1 = self._make_result("t1", status="PASSED", node_id=1, polarity=True)
        r2 = self._make_result("t2", status="RUNTIME_ERROR", node_id=1, polarity=True)
        r3 = self._make_result("t3", status="PASSED", node_id=1, polarity=True)

        suite.add_result(r1)
        suite.add_result(r2)
        suite.add_result(r3)

        assert not r1.is_redundant
        assert not r2.is_redundant
        # r3: len(tests)=2 < 3 → not redundant
        assert not r3.is_redundant

        # 4th: len(tests)=3 ≥ 3, new_keys=0 → redundant
        r4 = self._make_result("t4", status="PASSED", node_id=1, polarity=True)
        suite.add_result(r4)
        assert r4.is_redundant

    def test_failed_not_processed(self):
        """FAILED/COMPILE_ERROR tests don't update covered_keys."""
        suite = TestSuite(function_path="/f.cpp::foo()")
        suite.coverage.total_mcdc_pairs = 1
        r = self._make_result("t1", status="FAILED", node_id=1, polarity=True)
        suite.add_result(r)
        assert len(suite.coverage._covered_keys) == 0
        assert r.new_mcdc_pairs_covered == 0

    def test_condition_id_text_discovered(self):
        suite = TestSuite(function_path="/f.cpp::foo()")
        r = TestResult(
            test_name="t1",
            test_body="void test() {}",
            status="PASSED",
            condition_trace=[
                ConditionTraceEntry(
                    node_id=42, condition="a > b", true_branch_visited=True, false_branch_visited=False
                ),
            ],
        )
        suite.add_result(r)
        assert suite.coverage._condition_id_to_text[42] == "a > b"

    def test_iteration_counter(self):
        suite = TestSuite(function_path="/f.cpp::foo()")
        for i in range(5):
            suite.add_result(self._make_result(f"t{i}"))
        assert suite.iteration_count == 5
        # last test should have iteration = 5
        assert suite.tests[-1].iteration == 5

    def test_consecutive_redundant_counter(self):
        suite = TestSuite(function_path="/f.cpp::foo()")
        suite.coverage.total_mcdc_pairs = 1
        # Need 4 tests covering the same key for first redundant to appear (min_suite_size=3)
        results = [
            self._make_result(f"t{i}", node_id=1, polarity=True) for i in (1, 2, 3, 4)
        ]
        for r in results:
            suite.add_result(r)
        # t1,t2,t3: not redundant (len<3 before append). t4: redundant.
        assert not results[0].is_redundant
        assert not results[1].is_redundant
        assert not results[2].is_redundant
        assert results[3].is_redundant
        assert suite.coverage.consecutive_redundant == 1

        # 5th test, same key → also redundant
        r5 = self._make_result("t5", node_id=1, polarity=True)
        suite.add_result(r5)
        assert r5.is_redundant
        assert suite.coverage.consecutive_redundant == 2

    def test_test_suite_no_longer_exposes_legacy_coverage_api(self):
        suite = TestSuite(function_path="/f.cpp::foo()")

        for name in LEGACY_TEST_SUITE_APIS:
            assert name not in suite.__dict__
            assert not hasattr(TestSuite, name)


# ------------------------------------------------------------------ #
# CoverageDetail
# ------------------------------------------------------------------ #
class TestCoverageDetail:
    def test_pct_property(self):
        cd = CoverageDetail(visited=3, total=10, progress=0.3)
        assert cd.pct == 0.3

    def test_defaults(self):
        cd = CoverageDetail()
        assert cd.visited == 0
        assert cd.total == 0
        assert cd.pct == 0.0


# ------------------------------------------------------------------ #
# UnvisitedStatement / UnvisitedBranch
# ------------------------------------------------------------------ #
class TestUnvisitedStatementBranch:
    def test_unvisited_statement_defaults(self):
        us = UnvisitedStatement(statement="x = 1;")
        assert us.node_id is None
        assert us.statement == "x = 1;"
        assert us.line_in_function is None

    def test_unvisited_branch_partial_visit(self):
        ub = UnvisitedBranch(
            node_id=5, condition="flag", true_visited=True, false_visited=False, line_in_function=12,
        )
        assert ub.node_id == 5
        assert ub.true_visited is True
        assert ub.false_visited is False
        assert ub.line_in_function == 12

    def test_unvisited_branch_defaults(self):
        ub = UnvisitedBranch(condition="x", true_visited=False, false_visited=False)
        assert ub.node_id is None
        assert ub.start_offset is None
        assert ub.end_offset is None
