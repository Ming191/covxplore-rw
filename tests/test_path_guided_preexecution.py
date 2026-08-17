from covxplore.agents.schemas import PathGuidedTestAction
from covxplore.generation.path_guidance import PathGuidance
from covxplore.types import TraceSummary


def _candidate(name, path):
    return PathGuidedTestAction(
        test_name=name,
        test_body="f();",
        path_id=name,
        expected_path=[{"node_id": node, "outcome": outcome} for node, outcome in path],
    )


def test_prunes_duplicate_and_proper_prefix_before_execution():
    short = _candidate("short", [(6, "FALSE"), (55, "FALSE")])
    duplicate = _candidate("duplicate", [(6, "FALSE"), (55, "FALSE")])
    long = _candidate("long", [(6, "FALSE"), (55, "FALSE"), (61, "TRUE")])

    result = PathGuidance.prune([short, duplicate, long])

    assert result.kept == [long]
    assert {item.diagnostic.split(":")[0] for item in result.rejected} == {
        "EXPECTED_PATH_PREFIX",
        "DUPLICATE_EXPECTED_PATH",
    }


def test_scores_branch_support_without_prefix_penalty():
    candidate = _candidate("p1", [(6, "FALSE"), (55, "FALSE"), (61, "TRUE")])
    trace = TraceSummary(
        targetFunctionConditionSteps=[
            {"nodeId": 6, "branch": "FALSE"},
            {"nodeId": 55, "branch": "TRUE"},
            {"nodeId": 61, "branch": "TRUE"},
        ]
    )

    result = PathGuidance.score(candidate, trace)

    assert result.hypotheses == 3
    assert result.supported_hypotheses == 2
    assert result.branch_support_pct == 66.6667
    assert result.full_path_match is False
    assert result.first_unsupported_index == 1


def test_scores_compound_decision_leaves_from_condition_values():
    candidate = _candidate("p1", [(29, "TRUE"), (30, "FALSE")])
    trace = TraceSummary(
        targetFunctionConditionSteps=[
            {"nodeId": 29, "conditionValue": True},
            {"nodeId": 30, "conditionValue": False},
        ]
    )

    result = PathGuidance.score(candidate, trace)

    assert result.branch_support_pct == 100.0
    assert result.full_path_match is True


def test_full_path_match_allows_extra_loop_events_but_preserves_order():
    candidate = _candidate("p1", [(6, "TRUE"), (6, "FALSE"), (9, "TRUE")])
    trace = TraceSummary(
        targetFunctionConditionSteps=[
            {"nodeId": 6, "branch": "TRUE"},
            {"nodeId": 3, "branch": "FALSE"},
            {"nodeId": 6, "branch": "TRUE"},
            {"nodeId": 6, "branch": "FALSE"},
            {"nodeId": 9, "branch": "TRUE"},
        ]
    )

    result = PathGuidance.score(candidate, trace)

    assert result.supported_hypotheses == 3
    assert result.branch_support_pct == 100.0
    assert result.full_path_match is True
    assert result.first_unsupported_index is None


def test_mismatch_feedback_includes_first_node_runtime_values():
    candidate = _candidate("p1", [(33, "FALSE"), (34, "FALSE")])
    trace = TraceSummary(
        targetFunctionConditionSteps=[
            {"nodeId": 33, "conditionValue": False},
            {
                "nodeId": 34,
                "conditionValue": True,
                "runtimeValues": [
                    {"expression": "ch", "value": " ", "type": "char"}
                ],
            },
        ]
    )
    result = type("Result", (), {})()
    result.test_name = "space_eof"
    result.trace_summary = trace
    result.path_prediction = PathGuidance.score(candidate, trace)

    feedback = PathGuidance.mismatch_feedback([result])

    assert "space_eof [p1]: support=50.0%, full_match=false" in feedback
    assert "first_unsupported[1]: nodeId=34 expected=FALSE, observed=TRUE" in feedback
    assert 'runtime: ch=" "' in feedback


def test_mismatch_feedback_is_limited_and_omits_full_matches():
    mismatch = _candidate("bad", [(3, "TRUE")])
    matched = _candidate("good", [(3, "FALSE")])

    def result(name, candidate, branch):
        trace = TraceSummary(
            targetFunctionConditionSteps=[{"nodeId": 3, "branch": branch}]
        )
        value = type("Result", (), {})()
        value.test_name = name
        value.trace_summary = trace
        value.path_prediction = PathGuidance.score(candidate, trace)
        return value

    feedback = PathGuidance.mismatch_feedback(
        [result("matched", matched, "FALSE")]
        + [result(f"bad-{index}", mismatch, "FALSE") for index in range(4)]
    )

    assert "matched" not in feedback
    assert feedback.count("\n- ") == 3
    assert "bad-3" not in feedback
