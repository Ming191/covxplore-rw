from covxplore.generator import GenerationConfig, GenerationResult, StopReason, generate
from covxplore.types import TestResult, TestSuite


def test_generator_public_api_importable():
    assert GenerationConfig
    assert GenerationResult
    assert StopReason
    assert generate


def test_summary_contract_minimal_keys_and_token_total_sum():
    config = GenerationConfig(function_path="foo", prompt_variant="default", run_id="run")
    suite = TestSuite(function_path="foo")
    suite.tests = [TestResult(test_name="t", test_body="", status="PASSED")]
    result = GenerationResult(
        config=config,
        suite=suite,
        stop_reason="agent_done",
        crew_prompt_tokens=11,
        crew_completion_tokens=13,
    )

    summary = result.to_summary_dict()

    assert {"run_id", "function_path", "prompt_variant", "stop_reason", "metrics", "test_suite"} <= set(summary)
    assert summary["metrics"]["total_input_tokens"] == 11
    assert summary["metrics"]["total_output_tokens"] == 13
    assert summary["metrics"]["total_tokens"] == 24
    assert summary["tracing_url"] is None
    assert summary["llm_interactions"] == []
