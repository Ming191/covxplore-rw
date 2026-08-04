from covxplore.coverage.path_understanding import (
    compute_path_understanding_metrics,
    target_hit,
)
from covxplore.types import ExpectedPathStep, TestResult, TraceSummary


def _result(*, name="t", path=None, steps=None, status="PASSED"):
    return TestResult(
        test_name=name,
        test_body="x();",
        status=status,
        expected_path=path or [],
        trace_summary=TraceSummary.model_validate(
            {"targetFunctionConditionSteps": steps or []}
        ),
    )


def test_target_hit_from_ordered_trace():
    result = _result(
        path=[ExpectedPathStep(node_id=5, polarity="TRUE")],
        steps=[{"nodeId": 5, "conditionValue": True}],
    )
    assert target_hit(result) is True


def test_target_hit_when_prior_expected_step_diverges():
    result = _result(
        path=[
            ExpectedPathStep(node_id=1, polarity="FALSE"),
            ExpectedPathStep(node_id=5, polarity="TRUE"),
        ],
        steps=[
            {"nodeId": 1, "conditionValue": True},
            {"nodeId": 5, "conditionValue": True},
        ],
    )
    assert target_hit(result) is True


def test_target_miss_when_trace_has_opposite_polarity():
    result = _result(
        path=[ExpectedPathStep(node_id=5, polarity="TRUE")],
        steps=[{"nodeId": 5, "conditionValue": False}],
    )
    assert target_hit(result) is False


def test_target_miss_without_trace_no_fallback():
    result = _result(path=[ExpectedPathStep(node_id=5, polarity="TRUE")])
    assert target_hit(result) is False


def test_understanding_metrics_match_and_catalog_rates():
    matched = _result(
        name="ok",
        path=[ExpectedPathStep(node_id=1, polarity="TRUE")],
        steps=[{"nodeId": 1, "branch": "TRUE"}],
    )
    diverged = _result(
        name="bad",
        path=[ExpectedPathStep(node_id=99, polarity="FALSE")],
        steps=[{"nodeId": 99, "branch": "FALSE"}],
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
    assert metrics["target_hits"] == 2
    assert metrics["target_misses"] == 0


def test_failed_candidate_is_not_path_bearing():
    failed = _result(
        name="compile-fail",
        status="FAILED",
        path=[ExpectedPathStep(node_id=7, polarity="TRUE")],
    )
    passed = _result(
        name="passed",
        path=[ExpectedPathStep(node_id=7, polarity="TRUE")],
        steps=[{"nodeId": 7, "branch": "TRUE"}],
    )

    metrics = compute_path_understanding_metrics(
        accepted=[failed, passed],
        known_node_ids={7},
    )

    assert metrics["path_candidates"] == 1
    assert metrics["matched"] == 1
    assert metrics["diverged"] == 0
    assert metrics["path_match_rate"] == 1.0
    assert metrics["target_hits"] == 1
    assert metrics["target_misses"] == 0


def test_repair_success_after_divergence():
    first = _result(
        name="miss",
        path=[ExpectedPathStep(node_id=7, polarity="TRUE")],
        steps=[{"nodeId": 7, "branch": "FALSE"}],
    )
    repair = _result(
        name="fix",
        path=[ExpectedPathStep(node_id=7, polarity="TRUE")],
        steps=[{"nodeId": 7, "branch": "TRUE"}],
    )
    metrics = compute_path_understanding_metrics(
        accepted=[first, repair],
        known_node_ids={7},
    )
    assert metrics["repairs_attempted"] == 1
    assert metrics["repairs_succeeded"] == 1
    assert metrics["repair_success_rate"] == 1.0
