from covxplore.crews.test_generation.crew import TestGenerationRunner
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.strategies import DirectStrategy, get_reasoning_strategy


class FakeLLM:
    usage_metrics = object()


def test_generation_runner_exposes_strategy_and_usage():
    strategy = get_reasoning_strategy(ReasoningTechnique.NONE, FakeLLM())
    runner = TestGenerationRunner(strategy)
    assert isinstance(runner.strategy, DirectStrategy)
    assert runner.usage_metrics is FakeLLM.usage_metrics
