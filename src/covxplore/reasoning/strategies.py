from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from crewai import LLM
from pydantic import BaseModel

from covxplore.agents.schemas import GenerateTestBatchAction
from covxplore.llm import build_llm
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.context_agent import ContextAgent, ContextRequest
from covxplore.reasoning.context_tools import AkaUTContextTools
from covxplore.reasoning.schemas import ChainOfThoughtResult, PathGuidedResult

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningInput:
    system_prompt: str
    task_prompt: str
    function_path: str = ""
    execution_feedback_text: str | None = None
    cached_context_evidence: str | None = None
    disable_agentic_context: bool = False


class ReasoningStrategy(ABC):
    def __init__(
        self,
        llm: LLM | None = None,
        context_tools: AkaUTContextTools | None = None,
    ):
        self.llm = llm or build_llm()
        self.reasoning_trace: list[dict[str, str]] = []
        self._context_agent = ContextAgent(self._json_call, context_tools)

    @abstractmethod
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        """Generate one test batch without executing it."""

    def _json_call(self, prompt: str, schema: type[T]) -> T:
        messages = [
            {"role": "system", "content": "Return only the requested JSON object."},
            {"role": "user", "content": prompt},
        ]
        error = ""
        raw_response = ""
        for _ in range(2):
            response = self.llm.call(messages)
            raw_response = str(response)
            try:
                return schema.model_validate_json(_json_object(raw_response))
            except Exception as exc:
                error = str(exc)
                messages.extend(
                    [
                        {"role": "assistant", "content": raw_response},
                        {
                            "role": "user",
                            "content": f"Invalid output: {error}. Return a corrected JSON object only.",
                        },
                    ]
                )
        diagnostic = raw_response[:4096]
        logger.error("Invalid %s response after retry: %r", schema.__name__, diagnostic)
        raise ValueError(
            f"LLM did not return valid {schema.__name__}: {error}; raw response: {diagnostic!r}"
        )

    def _context_evidence(self, reasoning_input: ReasoningInput) -> str:
        if reasoning_input.disable_agentic_context:
            return ""
        if reasoning_input.cached_context_evidence is not None:
            return reasoning_input.cached_context_evidence
        if not reasoning_input.function_path:
            return ""
        evidence = self._context_agent.resolve(
            ContextRequest(
                function_path=reasoning_input.function_path,
                system_prompt=reasoning_input.system_prompt,
                task_prompt=reasoning_input.task_prompt,
                execution_feedback_text=reasoning_input.execution_feedback_text,
            )
        )
        reasoning_input.cached_context_evidence = evidence
        return evidence

    @staticmethod
    def _evidence_text(*parts: str) -> str:
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _prompt_with_context(reasoning_input: ReasoningInput, evidence: str) -> str:
        suffix = f"\n\nAGENTIC CONTEXT EVIDENCE:\n{evidence}" if evidence else ""
        return f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}{suffix}"

    @classmethod
    def _batch_prompt(
        cls,
        reasoning_input: ReasoningInput,
        context_evidence: str = "",
        reasoning_evidence: str = "",
    ) -> str:
        suffix = (
            f"\n\nREASONING EVIDENCE:\n{reasoning_evidence}"
            if reasoning_evidence
            else ""
        )
        return (
            f"{cls._prompt_with_context(reasoning_input, context_evidence)}{suffix}\n\n"
            "Return JSON matching: "
            '{"candidates":[{"test_name":"...","test_body":"..."}]}.'
        )


class DirectStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        context = self._context_evidence(reasoning_input)
        return self._json_call(
            self._batch_prompt(reasoning_input, context), GenerateTestBatchAction
        )


class ChainOfThoughtStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        context = self._context_evidence(reasoning_input)
        prompt = (
            f"{self._prompt_with_context(reasoning_input, context)}\n\n"
            "Before writing tests, build a coverage plan in this exact order inside reasoning:\n"
            "1. Resolve relevant constants and helper predicates. Enumerate every focal condition outcome "
            "and mark it reachable or unreachable with a concrete reason.\n"
            "2. For each reachable outcome, derive concrete input/state constraints, including loop "
            "iterations and nested/helper calls that may re-enter the focal function.\n"
            "3. Propose candidates and map each candidate to the condition outcomes it should cover.\n"
            "4. Order candidates, compute expected marginal coverage against the union of earlier "
            "candidates, and remove or merge any candidate whose outcomes are a subset. Different input "
            "syntax alone is not diversity.\n"
            "5. Check that each remaining test body is compilable under the supplied harness contract. "
            "Do not claim complete reachable coverage unless every reachable outcome is mapped.\n"
            "Then provide the final batch. Return JSON matching "
            '{"reasoning":"numbered coverage plan with reachability, constraints, candidate mapping, '
            'overlap check, and compile check",'
            '"candidates":[{"test_name":"...","test_body":"..."}]}. '
        )
        result = self._json_call(prompt, ChainOfThoughtResult)
        self.reasoning_trace.append({"stage": "chain_of_thought", "content": result.reasoning})
        return GenerateTestBatchAction(candidates=result.candidates)


class PathGuidedStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        context = self._context_evidence(reasoning_input)
        from covxplore.api_client import AkaUTClient

        with AkaUTClient() as client:
            paths = client.get_node_conditions(
                reasoning_input.function_path, coverage_type="BRANCH"
            ).execution_paths
        if not paths:
            raise ValueError("AkaUT returned no CFG execution paths")
        uncovered_targets = _uncovered_targets(reasoning_input.execution_feedback_text)
        selected = [
            path
            for path in paths
            if not uncovered_targets
            or (path.target_node_id, path.target_outcome) in uncovered_targets
        ]
        selected.sort(
            key=lambda path: (
                -len(path.execution_sequence),
                path.target_node_id,
                path.target_outcome,
            )
        )
        payload = [
            {
                "target_node_id": path.target_node_id,
                "target_condition": path.target_condition,
                "target_outcome": path.target_outcome,
                "execution_sequence": [
                    {
                        "node_id": step.node_id,
                        "condition": step.condition,
                        "required_outcome": step.required_outcome,
                    }
                    for step in path.execution_sequence
                ],
            }
            for path in selected[:5]
        ]
        result = self._json_call(
            f"{self._prompt_with_context(reasoning_input, context)}\n\n"
            "CFG-DERIVED EXECUTION PATHS:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Generate one test per listed target path. Satisfy the ordered outcomes exactly; earlier "
            "conditions are mandatory reachability constraints. Do not add Chain-of-Thought or a "
            "reasoning field. Avoid a candidate if another listed path's test necessarily covers its "
            "target. Return one JSON object, never a JSON array. For JSON-like C++ input, use raw "
            'literals such as R"AKA({...})AKA", never escaped quotes. Return JSON matching '
            '{"candidates":[{"test_name":"...","test_body":"...",'
            '"target_node_id":1,"target_outcome":"TRUE"}]}.',
            PathGuidedResult,
        )
        self.reasoning_trace.append(
            {
                "stage": "cfg_execution_paths",
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )
        return GenerateTestBatchAction(
            candidates=[
                {"test_name": candidate.test_name, "test_body": candidate.test_body}
                for candidate in result.candidates
            ]
        )


_STRATEGIES: dict[ReasoningTechnique, type[ReasoningStrategy]] = {
    ReasoningTechnique.NONE: DirectStrategy,
    ReasoningTechnique.COT: ChainOfThoughtStrategy,
    ReasoningTechnique.PATH_GUIDED: PathGuidedStrategy,
}


def get_reasoning_strategy(
    technique: ReasoningTechnique,
    llm: LLM | None = None,
    context_tools: AkaUTContextTools | None = None,
) -> ReasoningStrategy:
    return _STRATEGIES[technique](llm, context_tools)


def _uncovered_targets(feedback: str | None) -> set[tuple[int, str]]:
    if not feedback:
        return set()
    import re

    targets: set[tuple[int, str]] = set()
    for node_id, outcomes in re.findall(
        r"node:(\d+).*?missing=([A-Z,]+)", feedback
    ):
        targets.update((int(node_id), outcome) for outcome in outcomes.split(","))
    return targets


def _json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl >= 0:
            stripped = stripped[first_nl + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    return stripped[start : end + 1]
