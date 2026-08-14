import json

from covxplore.api_client import ExecutionPath, PathStep
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.strategies import ReasoningInput, get_reasoning_strategy


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def call(self, messages):
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
    assert "compute expected marginal coverage" in llm.prompts[0]


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
                "target_node_id": 3,
                "target_outcome": "TRUE",
            }]
        })
    ])
    strategy = get_reasoning_strategy(ReasoningTechnique.PATH_GUIDED, llm)
    result = strategy.generate_batch(_input())
    assert result.candidates[0].test_name == "path"
    assert "CFG-DERIVED EXECUTION PATHS" in llm.prompts[0]
    assert "Chain-of-Thought or a reasoning field" in llm.prompts[0]


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
