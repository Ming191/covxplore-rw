import pytest
from pydantic import ValidationError

from covxplore.agents import GenerateTestAction, validate_tool_input
from covxplore.agents.schemas import GenerateTestBatchAction, GenerateTestBatchAnyAction


def _valid_payload(**overrides):
    payload = {
        "test_body": "int x = 1;\nAKA_ACTUAL_OUTPUT = x;",
        "test_name": "t1",
    }
    payload.update(overrides)
    return payload


def test_generate_test_action_accepts_valid_payload():
    action = GenerateTestAction.model_validate(
        _valid_payload(target_node_id=2, target_polarity="TRUE", target_reason="cover true")
    )

    assert action.target_node_id == 2
    assert action.target_polarity == "TRUE"


@pytest.mark.parametrize("field", ["test_body"])
def test_generate_test_action_rejects_blank_required_fields(field):
    with pytest.raises(ValidationError):
        GenerateTestAction.model_validate(_valid_payload(**{field: "  "}))


def test_generate_test_action_rejects_invalid_polarity():
    with pytest.raises(ValidationError):
        GenerateTestAction.model_validate(
            _valid_payload(target_node_id=2, target_polarity="MAYBE")
        )


def test_generate_test_action_requires_node_for_polarity():
    with pytest.raises(ValidationError):
        GenerateTestAction.model_validate(_valid_payload(target_polarity="TRUE"))


def test_generate_test_action_requires_polarity_for_node():
    with pytest.raises(ValidationError):
        GenerateTestAction.model_validate(_valid_payload(target_node_id=2))


def test_generate_test_action_rejects_non_positive_node_id():
    with pytest.raises(ValidationError):
        GenerateTestAction.model_validate(_valid_payload(target_node_id=0, target_polarity="TRUE"))


@pytest.mark.parametrize("field", ["test_name", "target_reason"])
def test_generate_test_action_rejects_blank_optional_strings(field):
    with pytest.raises(ValidationError):
        GenerateTestAction.model_validate(_valid_payload(**{field: "  "}))


def test_validate_action_returns_errors_without_raising():
    result = validate_tool_input(_valid_payload(test_body=""))

    assert result.ok is False
    assert result.errors
    assert "test_body" in result.errors[0]


def test_validate_action_rejects_partial_target_metadata():
    result = validate_tool_input(_valid_payload(target_node_id=2))

    assert result.ok is False
    assert any("target_node_id requires target_polarity" in error for error in result.errors)


def test_validate_action_warns_when_target_reason_missing():
    result = validate_tool_input(_valid_payload(target_node_id=2, target_polarity="FALSE"))

    assert result.ok is True
    assert "target_polarity provided without target_reason" in result.warnings


def test_generate_test_batch_action_rejects_more_than_five_candidates():
    GenerateTestBatchAction.model_validate(
        {"candidates": [_valid_payload(test_name=f"t{i}") for i in range(5)]}
    )

    with pytest.raises(ValidationError):
        GenerateTestBatchAction.model_validate(
            {"candidates": [_valid_payload(test_name=f"t{i}") for i in range(6)]}
        )


def test_generate_test_batch_any_action_accepts_more_than_five_candidates():
    action = GenerateTestBatchAnyAction.model_validate(
        {"candidates": [_valid_payload(test_name=f"t{i}") for i in range(8)]}
    )

    assert len(action.candidates) == 8
