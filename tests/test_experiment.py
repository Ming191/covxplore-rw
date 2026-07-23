from covxplore.experiment import flat_row
from covxplore.generator import GenerationConfig, GenerationResult
from covxplore.types import TestSuite


def test_flat_row_projects_structural_run():
    config = GenerationConfig(function_path="/a.cpp::f()", prompt_variant="full")
    result = GenerationResult(config=config, suite=TestSuite(function_path=config.function_path), stop_reason="agent_done")
    row = flat_row(result)
    assert row["function_path"] == config.function_path
    assert row["accepted_test_count"] == 0
