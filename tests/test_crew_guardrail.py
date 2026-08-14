from covxplore.crews.test_generation.crew import TestGenerationRunner
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.strategies import DirectStrategy, get_reasoning_strategy


class FakeLLM:
    metrics = object()

    def get_token_usage_summary(self):
        return self.metrics


def test_generation_runner_exposes_strategy_and_usage():
    strategy = get_reasoning_strategy(ReasoningTechnique.NONE, FakeLLM())
    runner = TestGenerationRunner(strategy)
    assert isinstance(runner.strategy, DirectStrategy)
    assert runner.usage_metrics is FakeLLM.metrics
