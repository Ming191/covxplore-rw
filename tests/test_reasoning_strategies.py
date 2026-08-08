import json

import pytest

from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.strategies import (
    ReasoningInput,
    execute_program_of_thoughts,
    get_reasoning_strategy,
)


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
    result = get_reasoning_strategy(ReasoningTechnique.COT, llm).generate_batch(_input())
    assert result.candidates[0].test_name == "cot"
    assert "step by step" in llm.prompts[0]


def test_least_to_most_decomposes_then_solves_in_order():
    llm = FakeLLM([
        '{"subproblems":["setup","hard branch"]}',
        '{"analysis":"setup solved","scenarios":["s1"]}',
        '{"analysis":"hard solved","scenarios":["s2"]}',
        _batch("ltm"),
    ])
    result = get_reasoning_strategy(
        ReasoningTechnique.LEAST_TO_MOST, llm
    ).generate_batch(_input())
    assert result.candidates[0].test_name == "ltm"
    assert len(llm.prompts) == 4
    assert "setup solved" in llm.prompts[2]
    assert "hard solved" in llm.prompts[3]


def test_tot_generates_evaluates_expands_and_selects_before_batch():
    plans = {
        "plans": [
            {"plan_id": "a", "approach": "A", "target_gaps": ["a"], "setup_strategy": "A"},
            {"plan_id": "b", "approach": "B", "target_gaps": ["b"], "setup_strategy": "B"},
            {"plan_id": "c", "approach": "C", "target_gaps": ["c"], "setup_strategy": "C"},
        ]
    }
    scores = {"scores": [
        {"plan_id": "a", "score": 4, "rationale": "ok"},
        {"plan_id": "b", "score": 9, "rationale": "best"},
        {"plan_id": "c", "score": 2, "rationale": "weak"},
    ]}
    final_scores = {"scores": [
        {"plan_id": "b2", "score": 10, "rationale": "best"},
        {"plan_id": "a2", "score": 5, "rationale": "ok"},
        {"plan_id": "b", "score": 8, "rationale": "base"},
    ]}
    llm = FakeLLM([
        json.dumps(plans), json.dumps(scores),
        '{"plan_id":"b2","approach":"B2","target_gaps":["b"],"setup_strategy":"B2"}',
        '{"plan_id":"a2","approach":"A2","target_gaps":["a"],"setup_strategy":"A2"}',
        json.dumps(final_scores), _batch("tot"),
    ])
    result = get_reasoning_strategy(
        ReasoningTechnique.TREE_OF_THOUGHTS, llm
    ).generate_batch(_input())
    assert result.candidates[0].test_name == "tot"
    assert len(llm.prompts) == 6
    assert '"plan_id":"b2"' in llm.prompts[-1]


def test_program_of_thoughts_executes_program_before_synthesis():
    llm = FakeLLM([
        json.dumps({"program": "scenarios = []\nfor x in range(3):\n    scenarios.append({'x': x})"}),
        _batch("pot"),
    ])
    result = get_reasoning_strategy(
        ReasoningTechnique.PROGRAM_OF_THOUGHTS, llm
    ).generate_batch(_input())
    assert result.candidates[0].test_name == "pot"
    assert '"x":0' in llm.prompts[-1]


def test_program_of_thoughts_rejects_imports_and_file_access():
    with pytest.raises(ValueError, match="forbids Import"):
        execute_program_of_thoughts("import os\nscenarios = []")
    with pytest.raises(ValueError, match="call is not permitted"):
        execute_program_of_thoughts("scenarios = open('/tmp/x').read()")
