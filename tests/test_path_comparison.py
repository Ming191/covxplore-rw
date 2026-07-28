from covxplore.coverage.path_comparison import (
    compare_expected_path,
    effective_expected_path,
    format_divergence,
    is_proper_path_prefix,
    path_signature,
)
from covxplore.types import ExpectedPathStep, TestResult, TraceSummary, UnvisitedBranch
from covxplore.types.trace_summary import TargetFunctionConditionStep


def _result(*, expected_path=None, target_node_id=None, target_polarity=None, branches=None):
    return TestResult(
        test_name="t",
        test_body="x();",
        status="PASSED",
        target_node_id=target_node_id,
        target_polarity=target_polarity,
        expected_path=expected_path or [],
        unvisited_branches=branches or [],
    )


def test_effective_path_falls_back_to_legacy_target():
    result = _result(target_node_id=7, target_polarity="FALSE")
    path = effective_expected_path(result)
    assert path == [ExpectedPathStep(node_id=7, polarity="FALSE")]


def test_compare_matched_when_predicted_side_not_listed_as_unvisited():
    # Node absent from unvisited_branches ⇒ both sides observed this run.
    result = _result(
        expected_path=[
            ExpectedPathStep(node_id=1, polarity="TRUE"),
            ExpectedPathStep(node_id=2, polarity="FALSE"),
        ],
        branches=[],
    )
    assert compare_expected_path(result).matched is True


def test_compare_diverges_when_predicted_polarity_missing():
    result = _result(
        expected_path=[
            ExpectedPathStep(node_id=1, polarity="TRUE"),
            ExpectedPathStep(node_id=2, polarity="FALSE"),
        ],
        branches=[
            UnvisitedBranch(
                node_id=2,
                condition="x > 0",
                true_visited=True,
                false_visited=False,
            )
        ],
    )
    comparison = compare_expected_path(result)
    assert comparison.matched is False
    assert comparison.divergence_index == 1
    assert comparison.divergence_step is not None
    assert comparison.divergence_step.node_id == 2

    text = format_divergence(result)
    assert text is not None
    assert "node 2" in text
    assert "predicted FALSE" in text
    assert "only evaluated TRUE" in text


def test_compare_diverges_when_condition_never_reached():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=9, polarity="TRUE")],
        branches=[
            UnvisitedBranch(
                node_id=9,
                condition="ptr != nullptr",
                true_visited=False,
                false_visited=False,
            )
        ],
    )
    text = format_divergence(result)
    assert text is not None
    assert "never evaluated" in text


def test_path_signature_is_order_sensitive():
    a = [ExpectedPathStep(node_id=1, polarity="TRUE"), ExpectedPathStep(node_id=2, polarity="FALSE")]
    b = [ExpectedPathStep(node_id=2, polarity="FALSE"), ExpectedPathStep(node_id=1, polarity="TRUE")]
    assert path_signature(a) != path_signature(b)


def test_compare_diverges_on_unknown_catalog_node():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=999, polarity="TRUE")],
        branches=[],
    )
    comparison = compare_expected_path(result, known_node_ids={1, 2, 3})
    assert comparison.matched is False
    assert comparison.divergence_step is not None
    assert comparison.divergence_step.node_id == 999
    text = format_divergence(result, known_node_ids={1, 2, 3})
    assert text is not None
    assert "BRANCH NODE CATALOG" in text


def test_compare_matched_when_node_in_catalog():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=2, polarity="FALSE")],
        branches=[],
    )
    assert compare_expected_path(result, known_node_ids={1, 2, 3}).matched is True


def test_is_proper_path_prefix():
    short = [ExpectedPathStep(node_id=1, polarity="TRUE")]
    long = [
        ExpectedPathStep(node_id=1, polarity="TRUE"),
        ExpectedPathStep(node_id=2, polarity="FALSE"),
    ]
    other = [ExpectedPathStep(node_id=1, polarity="FALSE")]
    assert is_proper_path_prefix(short, long) is True
    assert is_proper_path_prefix(long, short) is False
    assert is_proper_path_prefix(short, other) is False
    assert is_proper_path_prefix(path_signature(short), path_signature(long)) is True


def test_format_divergence_includes_latest_runtime_operands_by_node_id():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=2, polarity="FALSE")],
        branches=[UnvisitedBranch(node_id=2, condition="p->ch == exitCh", true_visited=True, false_visited=False)],
    )
    result.trace_summary = TraceSummary.model_validate(
        {"targetFunctionConditionSteps": [
            {
                "line": 10,
                "start": 5,
                "end": 20,
                "nodeId": 2,
                "branch": "TRUE",
                "runtimeValues": [
                    {"expression": "p->ch", "value": "x", "type": "unsigned char"},
                ],
            },
            {
                "line": 10,
                "start": 5,
                "end": 20,
                "nodeId": 2,
                "branch": "TRUE",
                "runtimeValues": [
                    {"expression": "p->ch", "value": '","', "type": "unsigned char"},
                    {"expression": "exitCh", "value": '"}"', "type": "char"},
                ],
            },
        ]}
    )

    text = format_divergence(result)

    assert text is not None
    assert 'runtime operands: p->ch="," (unsigned char), exitCh="}" (char)' in text


def test_format_divergence_omits_runtime_operands_without_values():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=2, polarity="FALSE")],
        branches=[UnvisitedBranch(node_id=2, condition="x", true_visited=True, false_visited=False)],
    )
    result.trace_summary = TraceSummary.model_validate(
        {"targetFunctionConditionSteps": [{"line": 10, "start": 5, "end": 20, "nodeId": 2}]}
    )

    text = format_divergence(result)

    assert text is not None
    assert "runtime operands:" not in text


def test_format_divergence_matches_runtime_operands_by_offsets_without_node_id():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=7, polarity="TRUE")],
        branches=[
            UnvisitedBranch(
                node_id=7,
                condition="a == b",
                true_visited=False,
                false_visited=True,
                line_in_function=3,
                start_offset=11,
                end_offset=17,
            )
        ],
    )
    result.trace_summary = TraceSummary.model_validate(
        {"targetFunctionConditionSteps": [
            {
                "line": 3,
                "start": 11,
                "end": 17,
                "runtimeValues": [{"expression": "a", "value": "1", "type": "int"}],
            }
        ]}
    )

    text = format_divergence(result)

    assert text is not None
    assert "runtime operands: a=1 (int)" in text


def test_format_divergence_uses_subcondition_operands_for_adjacent_decision_node():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=18, polarity="TRUE")],
        branches=[
            UnvisitedBranch(
                node_id=18,
                condition="ch == delimiter",
                true_visited=False,
                false_visited=True,
                line_in_function=92,
                start_offset=2780,
                end_offset=2795,
            )
        ],
    )
    result.trace_summary = TraceSummary.model_validate(
        {"targetFunctionConditionSteps": [
            {
                "line": 92,
                "start": 2780,
                "end": 2795,
                "nodeId": 17,
                "runtimeValues": [
                    {"expression": "ch", "value": "\\n", "type": "char"},
                    {"expression": "delimiter", "value": '\"', "type": "char"},
                ],
            },
            {"line": 92, "start": 2780, "end": 2795, "nodeId": 18, "branch": "FALSE"},
        ]}
    )

    text = format_divergence(result)

    assert text is not None
    assert "node 18" in text
    assert 'runtime operands: ch=\\n (char), delimiter=" (char)' in text


def test_format_divergence_caps_and_truncates_runtime_operands():
    result = _result(
        expected_path=[ExpectedPathStep(node_id=2, polarity="FALSE")],
        branches=[UnvisitedBranch(node_id=2, condition="x", true_visited=True, false_visited=False)],
    )
    result.trace_summary = TraceSummary.model_validate(
        {
            "targetFunctionConditionSteps": [
                TargetFunctionConditionStep.model_validate(
                    {
                        "nodeId": 2,
                        "runtimeValues": [
                            {"expression": "a", "value": "x" * 100, "type": "char[]"},
                            {"expression": "b", "value": "2", "type": "int"},
                            {"expression": "c", "value": "3", "type": "int"},
                            {"expression": "d", "value": "4", "type": "int"},
                            {"expression": "e", "value": "5", "type": "int"},
                        ],
                    }
                )
            ]
        }
    )

    text = format_divergence(result)

    assert text is not None
    assert "a=" + "x" * 80 not in text
    assert "a=" + "x" * 79 + "…" in text
    assert "… (char[])" in text
    assert "d=4 (int)" in text
    assert "e=5 (int)" not in text
