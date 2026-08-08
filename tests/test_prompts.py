from covxplore.prompts.builder import PromptBuilder
from covxplore.prompts.catalog import load_catalog
from covxplore.prompts.config import ReasoningTechnique
from covxplore.prompts.registry import VARIANTS, get_reasoning_variants, get_variant
from covxplore.types import TestSuite


def test_registry_contains_only_reasoning_techniques():
    assert get_reasoning_variants() == [technique.value for technique in ReasoningTechnique]
    assert set(VARIANTS) == set(get_reasoning_variants())


def test_reasoning_variants_share_the_same_non_reasoning_prompt():
    prompts = {name: PromptBuilder(get_variant(name)).system_prompt() for name in VARIANTS}
    assert all("omit expected_path" in prompt.lower() for prompt in prompts.values())
    # Reasoning-specific guidance lives in strategies, not in the system prompt.
    # System prompts are role + output_format only; techniques are applied at call time.
    assert all(
        "without an explicit reasoning" not in prompt.lower()
        for prompt in prompts.values()
    )


def test_task_contains_context_source_gap_and_prior_feedback():
    text = PromptBuilder(get_variant("none")).task_description(
        function_path="/x.cpp::f()",
        suite=TestSuite(function_path="/x.cpp::f()"),
        remaining_batches=3,
        static_context_text="context",
        static_source_text="source",
        execution_feedback_text="compile failed",
    )
    assert "context" in text
    assert "source" in text
    assert "compile failed" in text
    assert "COVERAGE GUIDANCE" in text


def test_catalog_has_one_reasoning_template_per_technique():
    # Reasoning techniques live in strategies, not in the YAML catalog.
    catalog = load_catalog()
    assert "reasoning_techniques" not in catalog
