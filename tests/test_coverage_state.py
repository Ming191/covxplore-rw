import inspect

from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.coverage.state import CoverageState
from covxplore.types import CoverageDetail, TestResult, UnvisitedBranch, UnvisitedStatement


def _result(name: str, *, statements=(), branches=(), stmt=(0, 3), branch=(0, 4)):
    return TestResult(
        test_name=name,
        test_body=name,
        status="PASSED",
        statement_coverage=CoverageDetail(visited=stmt[0], total=stmt[1]),
        branch_coverage=CoverageDetail(visited=branch[0], total=branch[1]),
        unvisited_statements=list(statements),
        unvisited_branches=list(branches),
    )


def test_statement_gain_uses_cumulative_unvisited_ids():
    state = CoverageState()
    first = _result("first", stmt=(1, 3), statements=[
        UnvisitedStatement(node_id=2, statement="b;"),
        UnvisitedStatement(node_id=3, statement="c;"),
    ])
    second = _result("second", stmt=(2, 3), statements=[
        UnvisitedStatement(node_id=3, statement="c;"),
    ])
    state.add_result(first, prior_results=[])
    state.add_result(second, prior_results=[first])
    assert first.new_structural_coverage == 1
    assert second.new_structural_coverage == 1
    assert [item.node_id for item in state._cumulative_unvisited_stmts()] == [3]


def test_branch_side_gain_uses_cumulative_unvisited_ids():
    state = CoverageState()
    first = _result("first", branch=(0, 2), branches=[
        UnvisitedBranch(node_id=10, condition="x", true_visited=False, false_visited=False),
    ])
    second = _result("second", branch=(1, 2), branches=[
        UnvisitedBranch(node_id=10, condition="x", true_visited=True, false_visited=False),
    ])
    state.add_result(first, prior_results=[])
    state.add_result(second, prior_results=[first])
    assert second.new_structural_coverage == 1
    assert state._cumulative_unvisited_brs()[0].true_visited is True


def test_missing_ids_fall_back_to_monotonic_counts():
    state = CoverageState()
    first = _result("first", stmt=(1, 3), statements=[UnvisitedStatement(node_id=None, statement="x;")])
    second = _result("second", stmt=(2, 3), statements=[UnvisitedStatement(node_id=None, statement="x;")])
    state.add_result(first, prior_results=[])
    state.add_result(second, prior_results=[first])
    assert first.new_structural_coverage == 1
    assert second.new_structural_coverage == 1


def test_later_zero_gain_is_redundant():
    state = CoverageState()
    first = _result("first", stmt=(1, 3))
    state.add_result(first, prior_results=[])
    duplicate = _result("duplicate", stmt=(1, 3))
    state.add_result(duplicate, prior_results=[first], min_suite_size=1)
    assert duplicate.is_redundant is True
    assert state.consecutive_redundant == 1


def test_gap_analyzer_uses_structural_gap_input():
    state = CoverageState()
    result = _result("first", stmt=(1, 3), branch=(1, 2))
    state.add_result(result, prior_results=[])
    gap = GapAnalyzer().analyze(state.gap_input([result], batch_count=1))
    assert "MC/DC" not in gap.text
    assert "suite" not in str(inspect.signature(GapAnalyzer.analyze))
