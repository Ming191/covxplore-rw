"""Backward-compatibility shim — real implementation moved to generation.runner."""
from covxplore.generation.runner import CrewBundle, StrategyRunner, build_crew  # noqa: F401

# Legacy alias
TestGenerationRunner = StrategyRunner
