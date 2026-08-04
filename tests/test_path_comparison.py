from covxplore.coverage.path_comparison import (
    compare_expected_path,
    effective_expected_path,
    format_divergence,
    is_proper_path_prefix,
    path_signature,
)
from covxplore.types import ExpectedPathStep, TestResult, TraceSummary


def _result(*, expected_path=None, target_node_id=None, target_polarity=None, steps=None):
    return TestResult(
        test_name="t",
        test_body="x();",
        status="PASSED",
        target_node_id=target_node_id,
        target_polarity=target_polarity,
        expected_path=expected_path or [],
        trace_summary=TraceSummary.model_validate(
            {"targetFunctionConditionSteps": steps or []}
        ),
    )


def test_effective_path_falls_back_to_legacy_target():
    result = _result(target_node_id=7, target_polarity="FALSE")
    assert effective_expected_path(result) == [ExpectedPathStep(node_id=7, polarity="FALSE")]


def test_compare_matches_exact_ordered_trace():
    result = _result(
        expected_path=[
            ExpectedPathStep(node_id=1, polarity="TRUE"),
            ExpectedPathStep(node_id=2, polarity="FALSE"),
        ],
        steps=[
            {"nodeId": 1, "conditionValue": True},
            {"nodeId": 2, "branch": "FALSE"},
        ],
    )
    assert compare_expected_path(result).matched is True


def test_compare_diverges_on_wrong_order():
    result = _result(
        expected_path=[
            ExpectedPathStep(node_id=1, polarity="TRUE"),
            ExpectedPathStep(node_id=2, polarity="FALSE"),
        ],
        steps=[
            {"nodeId": 2, "branch": "FALSE"},
            {"nodeId": 1, "branch": "TRUE"},
        ],
    )
    comparison = compare_expected_path(result)
    assert comparison.matched is False
    assert comparison.divergence_index == 0
    assert comparison.observed_step is not None
    assert comparison.observed_step.node_id == 2
    assert "runtime evaluated node 2=FALSE instead" in format_divergence(result)


def test_compare_diverges_when_trace_ends_before_expected_path():
    result = _result(
        expected_path=[
            ExpectedPathStep(node_id=1, polarity="TRUE"),
            ExpectedPathStep(node_id=9, polarity="TRUE"),
        ],
        steps=[{"nodeId": 1, "branch": "TRUE"}],
    )
    text = format_divergence(result)
    assert text is not None
    assert "ordered runtime trace ended" in text


def test_compare_does_not_fallback_without_trace():
    result = TestResult(
        test_name="t",
        test_body="x();",
        status="PASSED",
        expected_path=[ExpectedPathStep(node_id=1, polarity="TRUE")],
    )
    assert compare_expected_path(result).matched is False


def test_compare_diverges_on_unknown_catalog_node():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=999, polarity="TRUE")],
        steps=[{"nodeId": 999, "branch": "TRUE"}],
    )
    comparison = compare_expected_path(result, known_node_ids={1, 2, 3})
    assert comparison.matched is False
    assert "BRANCH NODE CATALOG" in format_divergence(result, known_node_ids={1, 2, 3})


def test_condition_value_is_used_when_branch_marker_is_absent():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=60, polarity="FALSE")],
        steps=[
            {
                "nodeId": 60,
                "conditionValue": False,
                "runtimeValues": [{"expression": "valLen", "value": "5", "type": "size_t"}],
            }
        ],
    )
    assert compare_expected_path(result).matched is True


def test_repeated_loop_steps_are_compared_in_order():
    result = _result(
        expected_path=[
            ExpectedPathStep(node_id=74, polarity="FALSE"),
            ExpectedPathStep(node_id=75, polarity="FALSE"),
            ExpectedPathStep(node_id=74, polarity="FALSE"),
            ExpectedPathStep(node_id=75, polarity="TRUE"),
        ],
        steps=[
            {"nodeId": 74, "conditionValue": False},
            {"nodeId": 75, "conditionValue": False},
            {"nodeId": 74, "conditionValue": False},
            {"nodeId": 75, "conditionValue": True},
        ],
    )
    assert compare_expected_path(result).matched is True


def test_path_signature_is_order_sensitive():
    a = [ExpectedPathStep(node_id=1, polarity="TRUE"), ExpectedPathStep(node_id=2, polarity="FALSE")]
    b = [ExpectedPathStep(node_id=2, polarity="FALSE"), ExpectedPathStep(node_id=1, polarity="TRUE")]
    assert path_signature(a) != path_signature(b)


def test_is_proper_path_prefix():
    short = [ExpectedPathStep(node_id=1, polarity="TRUE")]
    long = [short[0], ExpectedPathStep(node_id=2, polarity="FALSE")]
    other = [ExpectedPathStep(node_id=1, polarity="FALSE")]
    assert is_proper_path_prefix(short, long) is True
    assert is_proper_path_prefix(long, short) is False
    assert is_proper_path_prefix(short, other) is False
    assert is_proper_path_prefix(path_signature(short), path_signature(long)) is True


def test_format_divergence_includes_runtime_operands():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=2, polarity="FALSE")],
        steps=[
            {
                "nodeId": 2,
                "branch": "TRUE",
                "runtimeValues": [
                    {"expression": "p->ch", "value": '","', "type": "unsigned char"},
                    {"expression": "exitCh", "value": '"}"', "type": "char"},
                ],
            }
        ],
    )
    text = format_divergence(result)
    assert text is not None
    assert 'runtime operands: p->ch="," (unsigned char), exitCh="}" (char)' in text


def test_format_divergence_caps_and_truncates_runtime_operands():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=2, polarity="FALSE")],
        steps=[
            {
                "nodeId": 2,
                "branch": "TRUE",
                "runtimeValues": [
                    {"expression": "a", "value": "x" * 100, "type": "char[]"},
                    {"expression": "b", "value": "2", "type": "int"},
                    {"expression": "c", "value": "3", "type": "int"},
                    {"expression": "d", "value": "4", "type": "int"},
                    {"expression": "e", "value": "5", "type": "int"},
                ],
            }
        ],
    )
    text = format_divergence(result)
    assert text is not None
    assert "a=" + "x" * 79 + "…" in text
    assert "d=4 (int)" in text
    assert "e=5 (int)" not in text
