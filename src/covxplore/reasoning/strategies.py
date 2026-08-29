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
from covxplore.prompts.catalog import catalog_text
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.context_agent import ContextAgent, ContextRequest
from covxplore.reasoning.context_tools import AkaUTContextTools
from covxplore.reasoning.schemas import (
    ChainOfThoughtResult,
    GeneratedTestBatchResult,
    PathGuidedPlanningResult,
)
from covxplore.api_client import AkaUTClient

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


class DirectStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        prompt = "\n\n".join(
            (
                self._prompt_with_context(
                    reasoning_input, self._context_evidence(reasoning_input)
                ),
                catalog_text("reasoning", "direct"),
            )
        )
        result = self._json_call(
            prompt, GeneratedTestBatchResult, reasoning_input.system_prompt
        )
        return GenerateTestBatchAction(candidates=result.to_actions())


class ChainOfThoughtStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        context = self._context_evidence(reasoning_input)
        prompt = "\n\n".join(
            (
                self._prompt_with_context(reasoning_input, context),
                catalog_text("reasoning", "cot"),
            )
        )
        result = self._json_call(
            prompt, ChainOfThoughtResult, reasoning_input.system_prompt
        )
        self.reasoning_trace.append({"stage": "chain_of_thought", "content": result.reasoning})
        return GenerateTestBatchAction(candidates=result.to_actions())


class PathGuidedStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        context = self._context_evidence(reasoning_input)
        with AkaUTClient() as client:
            conditions = client.get_node_conditions(
                reasoning_input.function_path, coverage_type="BRANCH"
            ).conditions

        if not conditions:
            logger.warning("AkaUT returned no conditions, falling back to ChainOfThought")
            return ChainOfThoughtStrategy(self.llm).generate_batch(reasoning_input)

        uncovered_targets = _uncovered_targets(reasoning_input.task_prompt)
        catalog_lines: list[str] = []
        for i, cond in enumerate(conditions):
            node_id = cond.node_id
            if node_id is None:
                continue
            t_status = (
                "UNCOVERED"
                if not uncovered_targets or (node_id, "TRUE") in uncovered_targets
                else "COVERED"
            )
            f_status = (
                "UNCOVERED"
                if not uncovered_targets or (node_id, "FALSE") in uncovered_targets
                else "COVERED"
            )
            catalog_lines.append(
                f"- [N{i + 1}] Node {node_id} (Line ~{cond.line_in_function}): "
                f"`{cond.condition}` -> TRUE [{t_status}], FALSE [{f_status}]"
            )

        if not catalog_lines:
            logger.warning("AkaUT conditions lacked node ids, falling back to ChainOfThought")
            return ChainOfThoughtStrategy(self.llm).generate_batch(reasoning_input)

        prompt = "\n\n".join(
            (
                self._prompt_with_context(reasoning_input, context),
                catalog_text("reasoning", "path_guided").format(
                    node_catalog="\n".join(catalog_lines),
                ),
            )
        )

        result = self._json_call(
            prompt,
            PathGuidedPlanningResult,
            reasoning_input.system_prompt,
        )
        self.reasoning_trace.append(
            {"stage": "self_proposed_path_planning", "content": result.reasoning}
        )
        return GenerateTestBatchAction(candidates=result.to_actions())



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
