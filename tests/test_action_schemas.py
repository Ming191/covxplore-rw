import pytest
from pydantic import ValidationError

from covxplore.agents.schemas import GenerateTestAction, GenerateTestBatchAction


def test_generate_test_action_accepts_strict_no_path_payload():
    action = GenerateTestAction.model_validate(
        {"test_name": "t1", "test_body": "int AKA_AI_x = 1;\nf(AKA_AI_x);"}
    )
    assert action.test_name == "t1"


@pytest.mark.parametrize("field", ["test_body", "test_name"])
def test_generate_test_action_rejects_blank_strings(field):
    payload = {"test_name": "t1", "test_body": "f();", field: "  "}
    with pytest.raises(ValidationError):
        GenerateTestAction.model_validate(payload)


def test_generate_test_action_rejects_path_metadata():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GenerateTestAction.model_validate(
            {
                "test_name": "t1",
                "test_body": "f();",
                "expected_path": [{"node_id": 1, "polarity": "TRUE"}],
            }
        )


def test_generate_test_batch_accepts_at_most_five_candidates():
    GenerateTestBatchAction.model_validate(
        {"candidates": [{"test_name": f"t{i}", "test_body": "f();"} for i in range(5)]}
    )
    with pytest.raises(ValidationError):
        GenerateTestBatchAction.model_validate(
            {"candidates": [{"test_name": f"t{i}", "test_body": "f();"} for i in range(6)]}
        )
