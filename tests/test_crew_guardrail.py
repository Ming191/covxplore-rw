from covxplore.agents.schemas import (
    GenerateTestBatchAction,
    GenerateTestBatchAnyAction,
    GenerateTestBatchOptionalPathAction,
    GenerateTestBatchSingleAction,
)
from covxplore.crews.test_generation.crew import _validate_batch_json, batch_types
from covxplore.prompts.registry import get_variant


def test_default_variant_uses_bounded_batch_schema():
    assert batch_types(get_variant("ours")) is GenerateTestBatchAction


def test_ablation_variants_select_matching_batch_schema():
    assert batch_types(get_variant("wo_batch")) is GenerateTestBatchSingleAction
    assert batch_types(get_variant("no_search_unlimited")) is GenerateTestBatchAnyAction
    assert batch_types(get_variant("wo_path")) is GenerateTestBatchOptionalPathAction


def test_batch_guard_strips_prose_and_returns_validated_json():
    ok, output = _validate_batch_json(GenerateTestBatchAction)(
        'analysis before\n```json\n{"candidates":[{"test_body":"f();",'
        '"expected_path":[{"node_id":1,"polarity":"TRUE"}]}]}\n```'
    )

    assert ok is True
    assert output.startswith('{"candidates"')
    assert "analysis before" not in output


def test_batch_guard_rejects_invalid_payload_without_retry():
    ok, message = _validate_batch_json(GenerateTestBatchAction)("no json")

    assert ok is False
    assert message == "Return one JSON object with a candidates array."
