from types import SimpleNamespace

from covxplore.crews.test_generation.crew import TestGenerationCrew, _validate_batch


def test_task_validates_json_without_native_response_format():
    crew = TestGenerationCrew.__new__(TestGenerationCrew)
    crew._llm = "fake"
    crew.tasks_config = {"generate_tests": {"description": "task", "expected_output": "batch"}}
    task = crew.generate_tests()
    assert task.output_pydantic is None
    assert task.output_json is None
    assert task.guardrail is _validate_batch


def test_guardrail_returns_validated_canonical_json():
    ok, result = _validate_batch(
        SimpleNamespace(raw='{"candidates":[{"test_name":"t","test_body":"f();"}]}')
    )
    assert ok is True
    assert '"test_body":"f();"' in result


def test_guardrail_rejects_path_metadata():
    ok, message = _validate_batch(
        SimpleNamespace(
            raw='{"candidates":[{"test_name":"t","test_body":"f();","expected_path":[]}]}'
        )
    )
    assert ok is False
    assert "Extra inputs are not permitted" in message
