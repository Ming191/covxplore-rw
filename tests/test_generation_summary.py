import pytest

from covxplore.generator import GenerationConfig, GenerationResult, StopReason, generate
from covxplore.types import TestResult, TestSuite, TraceSummary


def test_generator_public_api_importable():
    assert GenerationConfig
    assert GenerationResult
    assert StopReason
    assert generate


def test_generation_config_validates_treatment_and_limits():
    with pytest.raises(ValueError, match="Unknown reasoning technique"):
        GenerationConfig(function_path="foo", prompt_variant="default")
    with pytest.raises(ValueError, match="max_batches must be > 0"):
        GenerationConfig(function_path="foo", prompt_variant="none", max_batches=0)


def test_generated_run_ids_are_unique():
    first = GenerationConfig(function_path="foo", prompt_variant="none")
    second = GenerationConfig(function_path="foo", prompt_variant="none")
    assert first.run_id != second.run_id


def test_summary_contract_and_frozen_experiment_metadata(monkeypatch):
    config = GenerationConfig(function_path="foo", prompt_variant="none", run_id="run")
    result = GenerationResult(
        config=config,
        suite=TestSuite(function_path="foo"),
        stop_reason="agent_done",
        crew_prompt_tokens=11,
        crew_completion_tokens=13,
    )
    captured = result.experiment
    monkeypatch.setattr("covxplore.generator.get_settings", lambda: object())

    summary = result.to_summary_dict()
    assert summary["metrics"]["total_tokens"] == 24
    assert summary["experiment"]["model_id"] == captured.model_id


def test_test_summary_preserves_trace_summary_runtime_values():
    test = TestResult(test_name="t", test_body="", status="PASSED")
    test.trace_summary = TraceSummary.model_validate(
        {
            "targetFunctionConditionSteps": [
                {
                    "nodeId": 4,
                    "runtimeValues": [{"expression": "x", "value": "1", "type": "int"}],
                }
            ]
        }
    )
    step = GenerationResult._test_summary(test)["trace_summary"][
        "targetFunctionConditionSteps"
    ][0]
    assert step["nodeId"] == 4
    assert step["runtimeValues"] == [{"expression": "x", "value": "1", "type": "int"}]
