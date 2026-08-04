import asyncio

from covxplore.api_client import ExecuteResult
from covxplore.agents.schemas import GenerateTestAction
from covxplore.generation.batch import (
    TestcaseCandidate,
    async_execute_candidates,
    filter_preflight_candidates,
    merge_batch_results,
)
from covxplore.types import CoverageDetail, TestResult, TestSuite, UnvisitedBranch, UnvisitedStatement


def _result(name: str, *, stmt: int = 0, branch: int = 0, body: str | None = None) -> TestResult:
    return TestResult(
        test_name=name, test_body=body or name, status="PASSED",
        statement_coverage=CoverageDetail(visited=stmt, total=3),
        branch_coverage=CoverageDetail(visited=branch, total=2),
    )


def _action(name: str, *, node_id: int, polarity: str, path=None) -> GenerateTestAction:
    return GenerateTestAction.model_validate(
        {
            "test_name": name,
            "test_body": f"int AKA_AI_x = 0; // {name}",
            "target_node_id": node_id,
            "target_polarity": polarity,
            "expected_path": path
            or [{"node_id": node_id, "polarity": polarity}],
        }
    )


def test_merge_accepts_first_passing_zero_gain_seed_result():
    suite = TestSuite(function_path="/f.cpp::f()")
    result = _result("seed")
    summary = merge_batch_results(suite, [result])
    assert summary.accepted == [result]
    assert suite.tests == [result]


def test_merge_rejects_later_zero_gain_result():
    suite = TestSuite(function_path="/f.cpp::f()")
    seed = _result("seed")
    merge_batch_results(suite, [seed])
    duplicate = _result("duplicate")
    summary = merge_batch_results(suite, [duplicate])
    assert summary.rejected_redundant == [duplicate]


def test_merge_orders_by_structural_gain():
    suite = TestSuite(function_path="/f.cpp::f()")
    small = _result("small", stmt=1, body="x();")
    large = _result("large", stmt=2, branch=1, body="long();")
    summary = merge_batch_results(suite, [small, large])
    assert summary.accepted == [large]
    assert summary.rejected_redundant == [small]


def test_async_execute_candidates_runs_sync_executor_sequentially():
    class Executor:
        def __init__(self):
            self.calls = []
            self.active = 0
            self.max_active = 0

        def execute(self, absolute_path, test_body, test_name=None):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((absolute_path, test_body, test_name))
            self.active -= 1
            return ExecuteResult(raw={"testName": test_name, "status": "PASSED"})

    executor = Executor()
    candidates = [TestcaseCandidate("a();", "A"), TestcaseCandidate("b();", "B")]
    results = asyncio.run(async_execute_candidates("/f.cpp::f()", candidates, executor))
    assert [result.test_name for result in results] == ["A", "B"]
    assert executor.calls == [
        ("/f.cpp::f()", "a();", "A"),
        ("/f.cpp::f()", "b();", "B"),
    ]
    assert executor.max_active == 1


def test_preflight_dedupes_identical_expected_paths_in_batch():
    suite = TestSuite(function_path="/f.cpp::f()")
    a = _action("a", node_id=3, polarity="TRUE")
    b = _action("b", node_id=3, polarity="TRUE")
    filtered = filter_preflight_candidates(suite, [a, b])
    assert [c.test_name for c in filtered.executable] == ["a"]
    assert len(filtered.notes) == 1
    assert "duplicate expected_path" in filtered.notes[0]


def test_preflight_drops_prefix_path_when_longer_path_present():
    suite = TestSuite(function_path="/f.cpp::f()")
    short = _action(
        "short",
        node_id=1,
        polarity="TRUE",
        path=[{"node_id": 1, "polarity": "TRUE"}],
    )
    long = _action(
        "long",
        node_id=2,
        polarity="FALSE",
        path=[
            {"node_id": 1, "polarity": "TRUE"},
            {"node_id": 2, "polarity": "FALSE"},
        ],
    )
    sibling = _action(
        "sibling",
        node_id=1,
        polarity="FALSE",
        path=[{"node_id": 1, "polarity": "FALSE"}],
    )
    # short is a proper prefix of long → drop short; sibling is not dominated.
    filtered = filter_preflight_candidates(suite, [short, long, sibling])
    assert [c.test_name for c in filtered.executable] == ["long", "sibling"]
    assert any("proper prefix" in note and "short" in note for note in filtered.notes)


def test_preflight_prefix_drop_is_order_independent():
    suite = TestSuite(function_path="/f.cpp::f()")
    short = _action(
        "short",
        node_id=1,
        polarity="TRUE",
        path=[{"node_id": 1, "polarity": "TRUE"}],
    )
    long = _action(
        "long",
        node_id=2,
        polarity="FALSE",
        path=[
            {"node_id": 1, "polarity": "TRUE"},
            {"node_id": 2, "polarity": "FALSE"},
        ],
    )
    filtered = filter_preflight_candidates(suite, [long, short])
    assert [c.test_name for c in filtered.executable] == ["long"]
    assert any("proper prefix" in note for note in filtered.notes)


def test_preflight_skips_already_covered_target():
    suite = TestSuite(function_path="/f.cpp::f()")
    # Seed suite coverage so node 3 TRUE is no longer uncovered.
    seed = TestResult(
        test_name="seed",
        test_body="seed();",
        status="PASSED",
        statement_coverage=CoverageDetail(visited=1, total=2),
        branch_coverage=CoverageDetail(visited=1, total=2),
        unvisited_branches=[
            UnvisitedBranch(
                node_id=3,
                condition="x > 0",
                true_visited=True,
                false_visited=False,
            ),
            UnvisitedBranch(
                node_id=4,
                condition="y > 0",
                true_visited=False,
                false_visited=False,
            ),
        ],
    )
    suite.add_result(seed, min_suite_size=0)

    already = _action("already", node_id=3, polarity="TRUE")
    still_needed = _action("still", node_id=3, polarity="FALSE")
    filtered = filter_preflight_candidates(suite, [already, still_needed])
    assert [c.test_name for c in filtered.executable] == ["still"]
    assert any("already covered" in note for note in filtered.notes)


def test_preflight_keeps_covered_branch_when_statement_gaps_remain():
    suite = TestSuite(function_path="/f.cpp::f()")
    seed = TestResult(
        test_name="seed",
        test_body="seed();",
        status="PASSED",
        statement_coverage=CoverageDetail(visited=1, total=2),
        branch_coverage=CoverageDetail(visited=1, total=2),
        unvisited_statements=[
            UnvisitedStatement(node_id=60, statement="return false;"),
        ],
        unvisited_branches=[
            UnvisitedBranch(
                node_id=3,
                condition="x > 0",
                true_visited=True,
                false_visited=False,
            ),
            UnvisitedBranch(
                node_id=4,
                condition="y > 0",
                true_visited=False,
                false_visited=False,
            ),
        ],
    )
    suite.add_result(seed, min_suite_size=0)
    assert suite.coverage._cumulative_uncovered_stmt_ids == {60}

    already = _action("already", node_id=3, polarity="TRUE")
    filtered = filter_preflight_candidates(suite, [already])
    assert [c.test_name for c in filtered.executable] == ["already"]
    assert not any("already covered" in note for note in filtered.notes)


def test_preflight_does_not_skip_unknown_node_id_as_covered():
    """line-as-nodeId phantoms must still be executable (not 'already covered')."""
    suite = TestSuite(function_path="/f.cpp::f()")
    seed = TestResult(
        test_name="seed",
        test_body="seed();",
        status="PASSED",
        statement_coverage=CoverageDetail(visited=1, total=2),
        branch_coverage=CoverageDetail(visited=1, total=2),
        unvisited_branches=[
            UnvisitedBranch(
                node_id=65,
                condition="c == 0",
                true_visited=True,
                false_visited=False,
                line_in_function=11,
            ),
        ],
    )
    suite.add_result(seed, min_suite_size=0)

    phantom_line_as_id = _action("phantom", node_id=11, polarity="FALSE")
    real_gap = _action("real", node_id=65, polarity="FALSE")
    filtered = filter_preflight_candidates(suite, [phantom_line_as_id, real_gap])
    assert [c.test_name for c in filtered.executable] == ["phantom", "real"]
    assert not any("already covered" in note for note in filtered.notes)


def test_merge_preserves_preflight_notes():
    suite = TestSuite(function_path="/f.cpp::f()")
    summary = merge_batch_results(
        suite, [_result("seed")], preflight_notes=["t2: skipped preflight — duplicate"]
    )
    assert summary.preflight_notes == ["t2: skipped preflight — duplicate"]
