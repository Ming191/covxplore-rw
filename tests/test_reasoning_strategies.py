import json

from covxplore.api_client import ExecutionPath, PathStep
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.strategies import (
    ReasoningInput,
    _uncovered_targets,
    get_reasoning_strategy,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def call(self, messages, response_model=None):
        self.prompts.append(messages[-1]["content"])
        return next(self.responses)


def _input():
    return ReasoningInput(system_prompt="system", task_prompt="task")


def _batch(name="t"):
    return json.dumps({"candidates": [{"test_name": name, "test_body": "f();"}]})


def test_direct_uses_one_call():
    llm = FakeLLM([_batch()])
    result = get_reasoning_strategy(ReasoningTechnique.NONE, llm).generate_batch(_input())
    assert result.candidates[0].test_name == "t"
    assert len(llm.prompts) == 1


def test_native_structured_output_accepts_model_instance():
    class NativeLLM(FakeLLM):
        native_structured_output = True

        def call(self, messages, response_model=None):
            self.prompts.append(messages[-1]["content"])
            return response_model.model_validate_json(next(self.responses))

    response = json.dumps(
        {
            "candidates": [
                {
                    "test_name": "native",
                    "test_body_encoded": "f(__DQ__x__BS__n__DQ__);",
                }
            ]
        }
    )
    result = get_reasoning_strategy(
        ReasoningTechnique.NONE, NativeLLM([response])
    ).generate_batch(_input())

    assert result.candidates[0].test_name == "native"
    assert result.candidates[0].test_body == 'f("x\\n");'


def test_native_structured_output_decodes_quote_free_raw_literal():
    class NativeLLM(FakeLLM):
        native_structured_output = True

        def call(self, messages, response_model=None):
            return response_model.model_validate(
                {
                    "candidates": [
                        {
                            "test_body_encoded": (
                                "std::istringstream input(__RAW_BEGIN____DQ____BS__u0041__DQ____RAW_END__);"
                            )
                        }
                    ]
                }
            )

    result = get_reasoning_strategy(
        ReasoningTechnique.NONE, NativeLLM([])
    ).generate_batch(_input())

    assert result.candidates[0].test_body == (
        'std::istringstream input(R"AKA("\\u0041")AKA");'
    )


def test_cot_requires_explicit_rationale_and_batch():
    llm = FakeLLM([
        json.dumps({"reasoning": "step 1; step 2", "candidates": [{"test_name": "cot", "test_body": "f();"}]})
    ])
    strategy = get_reasoning_strategy(ReasoningTechnique.COT, llm)
    result = strategy.generate_batch(_input())
    assert result.candidates[0].test_name == "cot"
    assert strategy.reasoning_trace == [
        {"stage": "chain_of_thought", "content": "step 1; step 2"}
    ]
    assert "no marginal coverage" in llm.prompts[0]


def test_path_guided_uses_cfg_paths_without_cot(monkeypatch):
    path = ExecutionPath(
        execution_sequence=[PathStep(3, "x > 0", "TRUE")],
        target_node_id=3,
        target_condition="x > 0",
        target_outcome="TRUE",
    )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def get_node_conditions(self, *_, **__):
            return type("Result", (), {"execution_paths": [path]})()

    monkeypatch.setattr("covxplore.api_client.AkaUTClient", FakeClient)
    llm = FakeLLM([
        json.dumps({
            "candidates": [{
                "test_name": "path",
                "test_body": "f();",
                "path_id": "p1",
            }]
        })
    ])
    strategy = get_reasoning_strategy(ReasoningTechnique.PATH_GUIDED, llm)
    result = strategy.generate_batch(_input())
    assert result.candidates[0].test_name == "path"
    assert result.candidates[0].path_id == "p1"
    assert result.candidates[0].expected_path[0].node_id == 3
    assert "CFG-DERIVED EXECUTION PATHS (T=TRUE, F=FALSE)" in llm.prompts[0]
    assert '"path":"3T"' in llm.prompts[0]
    assert '"expected_path"' not in llm.prompts[0]
    assert "Generate one candidate for every listed path_id" in llm.prompts[0]
    assert "Do not add reasoning" in llm.prompts[0]


def test_path_guided_filters_from_current_coverage_guidance(monkeypatch):
    paths = [
        ExecutionPath(
            execution_sequence=[PathStep(3, "x > 0", outcome)],
            target_node_id=3,
            target_condition="x > 0",
            target_outcome=outcome,
        )
        for outcome in ("TRUE", "FALSE")
    ]

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def get_node_conditions(self, *_, **__):
            return type("Result", (), {"execution_paths": paths})()

    monkeypatch.setattr("covxplore.api_client.AkaUTClient", FakeClient)
    llm = FakeLLM([
        json.dumps({
            "candidates": [{
                "test_name": "false_path",
                "test_body": "f();",
                "path_id": "p1",
            }]
        })
    ])
    strategy = get_reasoning_strategy(ReasoningTechnique.PATH_GUIDED, llm)
    result = strategy.generate_batch(
        ReasoningInput(
            system_prompt="system",
            task_prompt="[nodeId=3] 'x > 0' — missing: FALSE",
            execution_feedback_text="Accepted: previous(PASSED, +1)",
        )
    )

    assert result.candidates[0].expected_path[0].outcome == "FALSE"
    assert '"target":"3F"' in llm.prompts[0]
    assert '"target":"3T"' not in llm.prompts[0]


def test_uncovered_targets_parses_rendered_branch_feedback():
    feedback = """Branch coverage: 50%\n  • [nodeId=3] 'x > 0' — missing: TRUE, FALSE\n  • [nodeId=7] 'done' — missing: FALSE"""

    assert _uncovered_targets(feedback) == {
        (3, "TRUE"),
        (3, "FALSE"),
        (7, "FALSE"),
    }


def test_uncovered_targets_accepts_legacy_node_tag():
    assert _uncovered_targets("[node:3] 'x > 0' — missing: TRUE") == {(3, "TRUE")}


def test_cached_context_evidence_skips_agent_call():
    llm = FakeLLM([_batch("cached")])
    strategy = get_reasoning_strategy(ReasoningTechnique.NONE, llm)
    inp = ReasoningInput(
        system_prompt="sys",
        task_prompt="task",
        function_path="test_func",
        cached_context_evidence="PRECACHED EVIDENCE",
    )
    result = strategy.generate_batch(inp)
    assert result.candidates[0].test_name == "cached"
    assert "PRECACHED EVIDENCE" in llm.prompts[0]


def test_disable_agentic_context_returns_empty():
    llm = FakeLLM([_batch("disabled")])
    strategy = get_reasoning_strategy(ReasoningTechnique.NONE, llm)
    inp = ReasoningInput(
        system_prompt="sys",
        task_prompt="task",
        function_path="test_func",
        disable_agentic_context=True,
    )
    result = strategy.generate_batch(inp)
    assert result.candidates[0].test_name == "disabled"
    assert "AGENTIC CONTEXT EVIDENCE" not in llm.prompts[0]
