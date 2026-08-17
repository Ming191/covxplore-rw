from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from crewai import LLM
from pydantic import BaseModel

from covxplore.agents.schemas import GenerateTestBatchAction, PathGuidedTestBatchAction
from covxplore.llm import build_llm
from covxplore.prompts.catalog import catalog_text
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.context_agent import ContextAgent, ContextRequest
from covxplore.reasoning.context_tools import AkaUTContextTools
from covxplore.reasoning.schemas import (
    ChainOfThoughtResult,
    EncodedChainOfThoughtResult,
    EncodedPathGuidedResult,
    EncodedTestBatchResult,
    PathGuidedCandidate,
    PathGuidedResult,
)
from covxplore.api_client import ExecutionPath

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

    def _json_call(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "Return only the requested JSON object.",
    ) -> T:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        error = ""
        raw_response = ""
        for _ in range(2):
            response = self.llm.call(
                messages,
                response_model=(
                    schema if getattr(self.llm, "native_structured_output", False) else None
                ),
            )
            if isinstance(response, schema):
                return response
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
            )
        )
        reasoning_input.cached_context_evidence = evidence
        return evidence

    @staticmethod
    def _prompt_with_context(reasoning_input: ReasoningInput, evidence: str) -> str:
        if not evidence:
            return reasoning_input.task_prompt
        return f"{reasoning_input.task_prompt}\n\n{catalog_text('reasoning', 'context_evidence').format(evidence=evidence)}"

    def _encoded_output(self) -> str:
        return (
            catalog_text("reasoning", "encoded_output")
            if getattr(self.llm, "native_structured_output", False)
            else ""
        )


class DirectStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        encoded = getattr(self.llm, "native_structured_output", False)
        prompt = "\n\n".join(
            filter(
                None,
                (
                    self._prompt_with_context(
                        reasoning_input, self._context_evidence(reasoning_input)
                    ),
                    catalog_text("reasoning", "direct"),
                    self._encoded_output(),
                ),
            )
        )
        result = self._json_call(
            prompt,
            EncodedTestBatchResult if encoded else GenerateTestBatchAction,
            reasoning_input.system_prompt,
        )
        return result.decode() if encoded else result


class ChainOfThoughtStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        context = self._context_evidence(reasoning_input)
        encoded = getattr(self.llm, "native_structured_output", False)
        prompt = "\n\n".join(
            filter(
                None,
                (
                    self._prompt_with_context(reasoning_input, context),
                    catalog_text("reasoning", "cot"),
                    self._encoded_output(),
                ),
            )
        )
        result = self._json_call(
            prompt,
            EncodedChainOfThoughtResult if encoded else ChainOfThoughtResult,
            reasoning_input.system_prompt,
        )
        self.reasoning_trace.append({"stage": "chain_of_thought", "content": result.reasoning})
        return GenerateTestBatchAction(
            candidates=[candidate.decode() for candidate in result.candidates]
            if encoded
            else result.candidates
        )


class PathGuidedStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> PathGuidedTestBatchAction:
        context = self._context_evidence(reasoning_input)
        from covxplore.api_client import AkaUTClient

        with AkaUTClient() as client:
            paths = client.get_node_conditions(
                reasoning_input.function_path, coverage_type="BRANCH"
            ).execution_paths
        if not paths:
            raise ValueError("AkaUT returned no CFG execution paths")
        uncovered_targets = _uncovered_targets(reasoning_input.task_prompt)
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
        path_catalog = {
            f"p{index + 1}": path
            for index, path in enumerate(selected[:5])
        }
        payload = [
            {
                "id": path_id,
                "target": f"{path.target_node_id}{path.target_outcome[0]}",
                "condition": path.target_condition,
                "path": ">".join(
                    f"{step.node_id}{step.required_outcome[0]}"
                    for step in path.execution_sequence
                ),
            }
            for path_id, path in path_catalog.items()
        ]
        encoded = getattr(self.llm, "native_structured_output", False)
        output_contract = (
            self._encoded_output()
            if encoded
            else catalog_text("reasoning", "path_output")
        )
        result = self._json_call(
            "\n\n".join(
                (
                    self._prompt_with_context(reasoning_input, context),
                    catalog_text("reasoning", "path_guided").format(
                        paths=json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        ),
                        output_contract=output_contract,
                    ),
                )
            ),
            EncodedPathGuidedResult if encoded else PathGuidedResult,
            reasoning_input.system_prompt,
        )
        if encoded:
            result = result.decode()
        self.reasoning_trace.append(
            {
                "stage": "cfg_execution_paths",
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )
        return PathGuidedTestBatchAction(
            candidates=[
                self._resolve_candidate(candidate, path_catalog)
                for candidate in result.candidates
            ]
        )

    @staticmethod
    def _resolve_candidate(
        candidate: PathGuidedCandidate,
        path_catalog: dict[str, ExecutionPath],
    ) -> dict:
        path = path_catalog.get(candidate.path_id)
        if path is None:
            raise ValueError(f"Unknown path_id {candidate.path_id!r}")
        return {
            "test_name": candidate.test_name,
            "test_body": candidate.test_body,
            "path_id": candidate.path_id,
            "expected_path": [
                {"node_id": step.node_id, "outcome": step.required_outcome}
                for step in path.execution_sequence
            ],
        }


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
        r"node(?:Id=|:)(\d+).*?missing:\s*([A-Z]+(?:\s*,\s*[A-Z]+)*)", feedback
    ):
        targets.update(
            (int(node_id), outcome.strip()) for outcome in outcomes.split(",")
        )
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
