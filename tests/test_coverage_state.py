import inspect
from pathlib import Path

from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.types import CoverageGapInput, CoverageMetrics
from covxplore.coverage.state import CoverageState
from covxplore.types import (
    ConditionKey,
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TestSuite,
    UnvisitedBranch,
    UnvisitedStatement,
)


def _result(
    name: str,
    *,
    node_id: int = 1,
    polarity: bool = True,
    stmt_unvisited: list[UnvisitedStatement] | None = None,
    branch_unvisited: list[UnvisitedBranch] | None = None,
) -> TestResult:
    return TestResult(
        test_name=name,
        test_body="void test() {}",
        status="PASSED",
        condition_trace=[
            ConditionTraceEntry(
                node_id=node_id,
                condition=f"c{node_id}",
                true_branch_visited=polarity,
                false_branch_visited=not polarity,
                line_in_function=node_id,
            )
        ],
        statement_coverage=CoverageDetail(visited=1, total=3, progress=1 / 3),
        branch_coverage=CoverageDetail(visited=1, total=4, progress=0.25),
        mcdc_coverage=CoverageDetail(visited=1, total=2, progress=0.5),
        unvisited_statements=stmt_unvisited or [],
        unvisited_branches=branch_unvisited or [],
    )


def test_coverage_state_accumulates_mcdc_and_redundancy() -> None:
    state = CoverageState(total_mcdc_pairs=1)
    prior: list[TestResult] = []

    first = _result("t1")
    state.add_result(first, prior_results=prior)
    prior.append(first)
    assert state._covered_keys == {ConditionKey(1, True)}
    assert first.new_mcdc_pairs_covered == 1
    assert first.is_redundant is False

    for index in range(2, 5):
        result = _result(f"t{index}")
        state.add_result(result, prior_results=prior)
        prior.append(result)

    assert prior[-1].is_redundant is True
    assert state.consecutive_redundant == 1


def test_coverage_state_tracks_statement_and_branch_intersections() -> None:
    state = CoverageState()
    first = _result(
        "t1",
        stmt_unvisited=[
            UnvisitedStatement(node_id=1, statement="a;", line_in_function=1),
            UnvisitedStatement(node_id=2, statement="b;", line_in_function=2),
        ],
        branch_unvisited=[
            UnvisitedBranch(node_id=10, condition="x", true_visited=False, false_visited=False),
        ],
    )
    second = _result(
        "t2",
        node_id=2,
        stmt_unvisited=[UnvisitedStatement(node_id=2, statement="b;", line_in_function=2)],
        branch_unvisited=[
            UnvisitedBranch(node_id=10, condition="x", true_visited=True, false_visited=False),
        ],
    )

    state.add_result(first, prior_results=[])
    state.add_result(second, prior_results=[first])

    assert state._covered_statements([first, second]) == 2
    assert [s.node_id for s in state._cumulative_unvisited_stmts()] == [2]
    branches = state._cumulative_unvisited_brs()
    assert len(branches) == 1
    assert branches[0].true_visited is True
    assert branches[0].false_visited is False


def test_coverage_state_metrics_and_gap_input_are_explicit_dtos() -> None:
    state = CoverageState(total_mcdc_pairs=2)
    first = _result("t1")
    state.add_result(first, prior_results=[])

    metrics = state.metrics([first])
    gap_input = state.gap_input([first], iteration_count=1)

    assert isinstance(metrics, CoverageMetrics)
    assert metrics.mcdc_pct == 0.5
    assert metrics.covered_mcdc_pairs == 1
    assert metrics.total_mcdc_pairs == 2
    assert isinstance(gap_input, CoverageGapInput)
    assert gap_input.tests == [first]
    assert gap_input.metrics == metrics
    assert gap_input.iteration_count == 1


def test_coverage_state_stmt_branch_redundancy_without_mcdc_feedback() -> None:
    state = CoverageState(mcdc_execution_feedback=False)
    first = _result(
        "t1",
        stmt_unvisited=[
            UnvisitedStatement(node_id=1, statement="a;", line_in_function=1),
            UnvisitedStatement(node_id=2, statement="b;", line_in_function=2),
        ],
    )
    second = _result(
        "t2",
        stmt_unvisited=[
            UnvisitedStatement(node_id=1, statement="a;", line_in_function=1),
            UnvisitedStatement(node_id=2, statement="b;", line_in_function=2),
        ],
    )

    state.add_result(first, prior_results=[])
    state.add_result(second, prior_results=[first], min_suite_size=1)

    assert first.is_redundant is False
    assert second.is_redundant is True
    assert second.new_mcdc_pairs_covered == 0


def test_infer_gtest_config_from_function_path(tmp_path: Path) -> None:
    from covxplore.driver.gtest_executor import infer_gtest_config

    root = tmp_path / "proj"
    root.mkdir()
    (root / "main.cpp").write_text("int main() {}", encoding="utf-8")
    (root / "helper.cpp").write_text("void help() {}", encoding="utf-8")
    (root / "util.c").write_text("void util() {}", encoding="utf-8")

    cfg = infer_gtest_config(str(root / "main.cpp") + "/Ns::foo()")
    assert cfg["gtest_source_root"] == str(root)
    assert str(root / "helper.cpp") in cfg["gtest_project_sources"]
    assert str(root / "util.c") in cfg["gtest_project_sources"]
    assert str(root / "main.cpp") not in cfg["gtest_project_sources"]
    assert cfg["gtest_extra_compile_flags"] == ["-std=c++14"]


def test_gap_analyzer_accepts_gap_input_not_suite_object() -> None:
    suite = TestSuite(function_path="/f.cpp::foo()")
    suite.coverage.total_mcdc_pairs = 2
    suite.add_result(_result("t1"))

    direct = GapAnalyzer().analyze(suite.coverage.gap_input(suite.tests, suite.iteration_count))

    assert "The following MC/DC condition polarities are NOT yet covered" in direct.text
    assert "suite" not in str(inspect.signature(GapAnalyzer.analyze))


def test_prompt_text_constants_module_exists_and_is_used() -> None:
    from covxplore.coverage import prompt_text

    assert prompt_text.NO_TESTS_WITH_MCDC.startswith("No tests executed yet.")
    source = inspect.getsource(GapAnalyzer.analyze)
    assert "No tests executed yet. Target is" not in source
    assert "SUCCESS! All coverage targets" not in source
