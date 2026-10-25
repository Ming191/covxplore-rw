from covxplore.prompts.builder import PromptBuilder
from covxplore.prompts.catalog import load_catalog
from covxplore.prompts.config import ReasoningTechnique
from covxplore.prompts.registry import VARIANTS, get_reasoning_variants, get_variant
from covxplore.types import TestSuite


def test_registry_contains_only_reasoning_techniques():
    canonical = [technique.value for technique in ReasoningTechnique]
    assert canonical == ["none", "cot", "path_guided"]
    for technique in ReasoningTechnique:
        assert get_variant(technique.value) == technique
    assert get_variant("path-guide") == ReasoningTechnique.PATH_GUIDED


def test_reasoning_variants_share_the_same_non_reasoning_prompt():
    none_prompt = PromptBuilder(get_variant("none")).system_prompt()
    cot_prompt = PromptBuilder(get_variant("cot")).system_prompt()
    path_prompt = PromptBuilder(get_variant("path_guided")).system_prompt()
    assert none_prompt == cot_prompt
    assert "untrusted data" in none_prompt
    assert "Return only the JSON object" in none_prompt
    assert "Do not use C++ raw string literals" in none_prompt
    assert '"test_body_lines"' in none_prompt
    assert "expected_path" in path_prompt
    assert "path_id" in path_prompt
    assert '"test_body_lines"' in path_prompt


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


def test_candidate_limits_match_reasoning_schema():
    suite = TestSuite(function_path="/x.cpp::f()")
    common = dict(
        function_path="/x.cpp::f()",
        suite=suite,
        remaining_batches=3,
        static_context_text="context",
        static_source_text="source",
    )

    direct = PromptBuilder(get_variant("none")).task_description(**common)
    path = PromptBuilder(get_variant("path_guided")).task_description(**common)

    assert "Return 1-4 focused candidates and omit path metadata." in direct
    assert "Return 1-4 candidates with distinct path_id and compact expected_path" in path
    assert "Suite coverage progress" not in direct
    assert "Each returned batch is executed once" in direct


def test_catalog_has_one_reasoning_template_per_technique():
    # Reasoning techniques live in strategies, not in the YAML catalog.
    catalog = load_catalog()
    assert "reasoning_techniques" not in catalog
