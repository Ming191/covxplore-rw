from __future__ import annotations

from dataclasses import dataclass

from covxplore.llm import build_llm
from covxplore.prompts.builder import PromptBuilder
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.strategies import ReasoningInput, ReasoningStrategy, get_reasoning_strategy


@dataclass(frozen=True)
class CrewBundle:
    """Pairs a reasoning runner with the prompt builder for one technique."""

    runner: StrategyRunner
    builder: PromptBuilder


class StrategyRunner:
    """Execute one reasoning strategy and expose its LLM usage metrics."""

    def __init__(self, strategy: ReasoningStrategy):
        self.strategy = strategy

    @property
    def usage_metrics(self):
        return getattr(self.strategy.llm, "usage_metrics", None)

    def generate(self, inputs: ReasoningInput):
        return self.strategy.generate_batch(inputs)


def build_crew(technique: ReasoningTechnique) -> CrewBundle:
    llm = build_llm()
    return CrewBundle(
        runner=StrategyRunner(get_reasoning_strategy(technique, llm)),
        builder=PromptBuilder(technique),
    )
