"""Tests for covxplore.prompts — pure, no network, no secrets."""

from pathlib import Path

import pytest

from covxplore.prompts.config import PromptConfig
from covxplore.prompts.builder import PromptBuilder
from covxplore.prompts.catalog import load_catalog
from covxplore.prompts.registry import (
    VARIANTS,
    get_leave_one_out_variants,
    get_variant,
)
from covxplore.types import TestSuite


def test_prompt_catalog_contains_all_sections_and_templates():
    catalog = load_catalog()

    assert set(catalog["sections"]) == {
        "role_persona",
        "role_persona_no_path",
        "cot_reasoning",
        "cot_reasoning_no_path",
        "coverage_guidance",
        "self_reflection",
        "few_shot_examples",
        "output_format",
        "output_format_no_path",
    }
    assert set(catalog["path_feedback"]) == {"repair"}
    assert set(catalog["workflows"]) == {"unlimited_batch", "no_search", "no_path", "search"}
    assert set(catalog["coverage"]) == {
        "no_tests",
        "compile_error",
        "failed",
        "unknown",
        "success",
        "detail_unavailable",
        "detail_header",
        "max_statement_lines",
        "max_branch_lines",
    }
    assert set(catalog["task"]) == {"description", "context", "source", "branch_catalog"}


def test_legacy_prompt_text_files_are_removed():
    sections_dir = Path(__file__).parents[1] / "src" / "covxplore" / "prompts" / "sections"

    assert not list(sections_dir.glob("*.txt"))


class TestPromptConfigEnabledSections:
    def test_full_variant(self):
        cfg = PromptConfig(
            "full",
            role_persona=True,
            cot_reasoning=True,
            coverage_guidance=True,
            self_reflection=True,
            few_shot_examples=False,
            output_format=True,
        )
        sections = cfg.enabled_sections()
        assert "role_persona" in sections
        assert "cot_reasoning" in sections
        assert "coverage_guidance" in sections
        assert "self_reflection" in sections
        assert "few_shot_examples" not in sections
        assert "output_format" in sections
        assert len(sections) == 5

    def test_baseline_all_off(self):
        cfg = PromptConfig(
            "baseline",
            role_persona=False,
            cot_reasoning=False,
            coverage_guidance=False,
            self_reflection=False,
            few_shot_examples=False,
            output_format=False,
        )
        assert cfg.enabled_sections() == []

    def test_single_section(self):
        cfg = PromptConfig(
            "s1_role",
            role_persona=True,
            cot_reasoning=False,
            coverage_guidance=False,
            self_reflection=False,
            few_shot_examples=False,
            output_format=False,
        )
        assert cfg.enabled_sections() == ["role_persona"]

    def test_few_shot_only(self):
        cfg = PromptConfig(
            "s5_fewshot",
            role_persona=False,
            cot_reasoning=False,
            coverage_guidance=False,
            self_reflection=False,
            few_shot_examples=True,
            output_format=False,
        )
        assert cfg.enabled_sections() == ["few_shot_examples"]

    def test_order_preserved(self):
        # Defaults for omitted booleans are True (except few_shot_examples=False).
        # To test order, explicitly set what we care about and set others False.
        cfg = PromptConfig(
            "test",
            output_format=True,
            role_persona=True,
            cot_reasoning=False,
            coverage_guidance=False,
            self_reflection=False,
            few_shot_examples=False,
        )
        # enabled_sections returns in the canonical order
        assert cfg.enabled_sections() == ["role_persona", "output_format"]

    def test_str_representation(self):
        cfg = PromptConfig(
            "test",
            role_persona=True,
            cot_reasoning=False,
            coverage_guidance=False,
            self_reflection=False,
            few_shot_examples=False,
            output_format=False,
        )
        s = str(cfg)
        assert "'test'" in s
        assert "role_persona" in s
        assert "output_format" not in s

    def test_str_no_sections(self):
        cfg = PromptConfig(
            "baseline",
            role_persona=False,
            cot_reasoning=False,
            coverage_guidance=False,
            self_reflection=False,
            few_shot_examples=False,
            output_format=False,
        )
        s = str(cfg)
        assert "none" in s.lower()


class TestGetVariant:
    def test_known_variant_returns_prompt_config(self):
        cfg = get_variant("full")
        assert isinstance(cfg, PromptConfig)
        assert cfg.name == "full"
        assert cfg.role_persona is True

    def test_no_search_variant_disables_search_tools(self):
        cfg = get_variant("no_search")

        assert cfg.search_tools is False

    def test_minimal_context_search_keeps_search_without_preloaded_context(self):
        cfg = get_variant("minimal_context_search")

        assert cfg.search_tools is True
        assert cfg.preload_context is False
        assert cfg.self_reflection is False

    def test_no_search_unlimited_disables_search_and_batch_cap(self):
        cfg = get_variant("no_search_unlimited")

        assert cfg.search_tools is False
        assert cfg.unlimited_batch is True
        assert cfg.self_reflection is False

    def test_all_registered_variants_resolve(self):
        for name in VARIANTS:
            cfg = get_variant(name)
            assert cfg.name == name

    def test_unknown_variant_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown prompt variant"):
            get_variant("nonexistent_variant_xyz")


class TestGetLeaveOneOutVariants:
    def test_includes_full_refs(self):
        variants = get_leave_one_out_variants()
        assert "full" in variants
        assert "full_shots" in variants

    def test_includes_no_prefixes(self):
        variants = get_leave_one_out_variants()
        for name in VARIANTS:
            if name.startswith("no_") and name != "no_search":
                assert name in variants, f"{name} missing from leave-one-out"
        assert "no_search" not in variants

    def test_excludes_single_section_variants(self):
        variants = get_leave_one_out_variants()
        for name in ("s1_role", "s2_cot", "s3_coverage", "s4_reflection",
                     "s5_fewshot", "s6_format", "baseline"):
            assert name not in variants

    def test_first_are_refs_then_loo(self):
        variants = get_leave_one_out_variants()
        ref_count = sum(1 for v in variants if v in ("full", "full_shots"))
        assert ref_count == 2
        # refs come first
        assert variants[0] == "full"
        assert variants[1] == "full_shots"


class TestPromptBuilderTaskDescription:
    def test_task_description_includes_run_id_instruction(self):
        builder = PromptBuilder(
            PromptConfig(
                "minimal",
                role_persona=False,
                cot_reasoning=False,
                coverage_guidance=False,
                self_reflection=False,
                few_shot_examples=False,
                output_format=False,
            )
        )

        text = builder.task_description(function_path="/x.cpp::f()")

        assert "Run id" not in text
        assert "execute_testcase call MUST include" not in text

    def test_no_search_task_description_removes_search_workflow(self):
        builder = PromptBuilder(PromptConfig("no_search", search_tools=False))

        text = builder.task_description(function_path="/x.cpp::f()")

        assert "No search/source tools are available" in text
        assert "preloaded source" in text
        assert "search_nodes then get_node_source" not in text

    def test_path_prompt_requires_complete_exact_runtime_sequence(self):
        builder = PromptBuilder(PromptConfig("path"))

        system = builder.system_prompt()
        task = builder.task_description(function_path="/x.cpp::f()")
        text = system + "\n" + task

        assert "Predict every branch evaluation" in text
        assert "repeated loop evaluations" in text
        assert "never shorten the path" in text
        assert "Prefer a SHORT path" not in text
        assert "Shorten the path" not in text

    def test_no_path_prompt_has_no_path_instructions_or_metadata(self):
        builder = PromptBuilder(PromptConfig("no-path", require_expected_path=False))

        text = builder.system_prompt()

        assert "Do not output expected_path or target metadata" in text
        assert "omit expected_path and target metadata" in text
        assert "Predict every branch evaluation" not in text
        assert "path-aware candidate" not in text
        assert "Copy that N into expected_path" not in text
        assert "#include directives, main()" in text
        assert "prefix input variable names with AKA_AI_" in text
        assert "call the exact function signature" in text
        assert "assertions" in text
        assert '{"candidates":[{"test_name":"tc_f_001"' in text

    def test_path_feedback_repair_is_only_in_feedback_variant(self):
        with_feedback = PromptBuilder(PromptConfig("ours", path_feedback=True))
        without_feedback = PromptBuilder(PromptConfig("without", path_feedback=False))

        feedback_task = with_feedback.task_description(function_path="/x.cpp::f()")
        no_feedback_task = without_feedback.task_description(function_path="/x.cpp::f()")

        assert "preserve the confirmed runtime prefix" in feedback_task
        assert "preserve the confirmed runtime prefix" not in no_feedback_task

    def test_coverage_guidance_allows_covered_prefixes(self):
        builder = PromptBuilder(PromptConfig("coverage"))

        text = builder.task_description(
            function_path="/x.cpp::f()",
            suite=TestSuite(function_path="/x.cpp::f()"),
            remaining_batches=3,
        )

        assert "Covered prefix branches are allowed" in text
        assert "repeats same reached statement or branch path" not in text
