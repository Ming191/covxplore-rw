from covxplore.coverage.path_understanding import (
    compute_path_understanding_metrics,
    target_hit,
)
from covxplore.types import ExpectedPathStep, TestResult, UnvisitedBranch


def _result(*, name="t", path=None, branches=None, status="PASSED"):
    return TestResult(
        test_name=name,
        test_body="x();",
        status=status,
        expected_path=path or [],
        unvisited_branches=branches or [],
    )


def test_target_hit_when_node_absent_from_unvisited():
    result = _result(path=[ExpectedPathStep(node_id=5, polarity="TRUE")])
    assert target_hit(result) is True


def test_target_miss_when_opposite_polarity_only():
    result = _result(
        path=[ExpectedPathStep(node_id=5, polarity="TRUE")],
        branches=[
            UnvisitedBranch(
                node_id=5,
                condition="x",
                true_visited=False,
                false_visited=True,
            )
        ],
    )
    assert target_hit(result) is False


def test_understanding_metrics_match_and_catalog_rates():
    matched = _result(
        name="ok",
        path=[ExpectedPathStep(node_id=1, polarity="TRUE")],
        branches=[],
    )
    diverged = _result(
        name="bad",
        path=[ExpectedPathStep(node_id=99, polarity="FALSE")],
        branches=[],
    )
    metrics = compute_path_understanding_metrics(
        accepted=[matched],
        rejected=[diverged],
        known_node_ids={1, 2},
    )
    assert metrics["path_candidates"] == 2
    assert metrics["matched"] == 1
    assert metrics["diverged"] == 1
    assert metrics["path_match_rate"] == 0.5
    assert metrics["catalog_id_rate"] == 0.5
    # matched: target hit; diverged unknown id: still counts as target_hit via absence
    assert metrics["target_hits"] >= 1


def test_repair_success_after_divergence():
    first = _result(
        name="miss",
        path=[ExpectedPathStep(node_id=7, polarity="TRUE")],
        branches=[
            UnvisitedBranch(
                node_id=7,
                condition="c",
                true_visited=False,
                false_visited=True,
            )
        ],
    )
    repair = _result(
        name="fix",
        path=[ExpectedPathStep(node_id=7, polarity="TRUE")],
        branches=[],  # both sides observed ⇒ hit
    )
    metrics = compute_path_understanding_metrics(
        accepted=[first, repair],
        known_node_ids={7},
    )
    assert metrics["repairs_attempted"] == 1
    assert metrics["repairs_succeeded"] == 1
    assert metrics["repair_success_rate"] == 1.0
