"""Named ablation variants — every combination of interest for the study.

Naming convention:
  baseline        — no sections at all (pure zero-shot)
  s{N}            — only section N enabled
  full            — all sections except few-shot (default)
  full_shots      — all sections including few-shot
  no_{section}    — full minus one section (leave-one-out ablation)
"""
from covxplore.prompts.config import PromptConfig

VARIANTS: dict[str, PromptConfig] = {
    # ------------------------------------------------------------------ #
    # Baseline                                                            #
    # ------------------------------------------------------------------ #
    "baseline": PromptConfig(
        "baseline",
        role_persona=False,
        cot_reasoning=False,
        coverage_guidance=False,
        self_reflection=False,
        few_shot_examples=False,
        output_format=False,
    ),

    # ------------------------------------------------------------------ #
    # Single-section ablations (S1–S6 in isolation)                      #
    # ------------------------------------------------------------------ #
    "s1_role": PromptConfig(
        "s1_role",
        role_persona=True,
        cot_reasoning=False,
        coverage_guidance=False,
        self_reflection=False,
        few_shot_examples=False,
        output_format=False,
    ),
    "s2_cot": PromptConfig(
        "s2_cot",
        role_persona=False,
        cot_reasoning=True,
        coverage_guidance=False,
        self_reflection=False,
        few_shot_examples=False,
        output_format=False,
    ),
    "s3_coverage": PromptConfig(
        "s3_coverage",
        role_persona=False,
        cot_reasoning=False,
        coverage_guidance=True,
        self_reflection=False,
        few_shot_examples=False,
        output_format=False,
    ),
    "s4_reflection": PromptConfig(
        "s4_reflection",
        role_persona=False,
        cot_reasoning=False,
        coverage_guidance=False,
        self_reflection=True,
        few_shot_examples=False,
        output_format=False,
    ),
    "s5_fewshot": PromptConfig(
        "s5_fewshot",
        role_persona=False,
        cot_reasoning=False,
        coverage_guidance=False,
        self_reflection=False,
        few_shot_examples=True,
        output_format=False,
    ),
    "s6_format": PromptConfig(
        "s6_format",
        role_persona=False,
        cot_reasoning=False,
        coverage_guidance=False,
        self_reflection=False,
        few_shot_examples=False,
        output_format=True,
    ),

    # ------------------------------------------------------------------ #
    # Leave-one-out ablations (full minus one section)                   #
    # ------------------------------------------------------------------ #
    "no_role": PromptConfig(
        "no_role",
        role_persona=False,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
    ),
    "no_cot": PromptConfig(
        "no_cot",
        role_persona=True,
        cot_reasoning=False,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
    ),
    "no_coverage": PromptConfig(
        "no_coverage",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=False,
        self_reflection=True,
        output_format=True,
    ),
    "no_reflection": PromptConfig(
        "no_reflection",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=False,
        output_format=True,
    ),
    "no_search": PromptConfig(
        "no_search",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=False,
        output_format=True,
        search_tools=False,
    ),
    "minimal_context_search": PromptConfig(
        "minimal_context_search",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=False,
        output_format=True,
        search_tools=True,
        preload_context=False,
    ),
    "no_search_unlimited": PromptConfig(
        "no_search_unlimited",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=False,
        output_format=True,
        search_tools=False,
        unlimited_batch=True,
    ),
    "no_search_nopath": PromptConfig(
        "no_search_nopath",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=False,
        require_expected_path=False,
    ),
    "no_format": PromptConfig(
        "no_format",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=False,
    ),

    # ------------------------------------------------------------------ #
    # Full variants                                                       #
    # ------------------------------------------------------------------ #
    "full": PromptConfig(
        "full",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        few_shot_examples=False,
        output_format=True,
    ),
    "full_shots": PromptConfig(
        "full_shots",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        few_shot_examples=True,
        output_format=True,
    ),

    # ------------------------------------------------------------------ #
    # Q1 mechanism ablations (Ours = preload + search + path + feedback) #
    # ------------------------------------------------------------------ #
    "ours": PromptConfig(
        "ours",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=True,
        preload_context=True,
        require_expected_path=True,
        preload_branch_catalog=True,
        path_feedback=True,
        include_exec_detail=True,
        max_batch_candidates=5,
    ),
    "wo_gaps": PromptConfig(
        "wo_gaps",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=False,
        self_reflection=True,
        output_format=True,
        search_tools=True,
        preload_context=True,
        require_expected_path=True,
        preload_branch_catalog=True,
        path_feedback=True,
        include_exec_detail=True,
        max_batch_candidates=5,
    ),
    "wo_search": PromptConfig(
        "wo_search",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=False,
        preload_context=True,
        require_expected_path=True,
        preload_branch_catalog=True,
        path_feedback=True,
        include_exec_detail=True,
        max_batch_candidates=5,
    ),
    "gap_ids_only": PromptConfig(
        "gap_ids_only",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=True,
        preload_context=True,
        require_expected_path=True,
        preload_branch_catalog=False,
        path_feedback=True,
        include_exec_detail=True,
        max_batch_candidates=5,
    ),
    "wo_path": PromptConfig(
        "wo_path",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=True,
        preload_context=True,
        require_expected_path=False,
        preload_branch_catalog=True,
        path_feedback=False,
        include_exec_detail=True,
        max_batch_candidates=5,
    ),
    "wo_path_feedback": PromptConfig(
        "wo_path_feedback",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=True,
        preload_context=True,
        require_expected_path=True,
        preload_branch_catalog=True,
        path_feedback=False,
        include_exec_detail=True,
        max_batch_candidates=5,
    ),
    "wo_batch": PromptConfig(
        "wo_batch",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=True,
        preload_context=True,
        require_expected_path=True,
        preload_branch_catalog=True,
        path_feedback=True,
        include_exec_detail=True,
        max_batch_candidates=1,
    ),
    "wo_exec_detail": PromptConfig(
        "wo_exec_detail",
        role_persona=True,
        cot_reasoning=True,
        coverage_guidance=True,
        self_reflection=True,
        output_format=True,
        search_tools=True,
        preload_context=True,
        require_expected_path=True,
        preload_branch_catalog=True,
        path_feedback=True,
        include_exec_detail=False,
        max_batch_candidates=5,
    ),
}


def get_variant(name: str) -> PromptConfig:
    if name not in VARIANTS:
        available = ", ".join(sorted(VARIANTS))
        raise KeyError(f"Unknown prompt variant {name!r}. Available: {available}")
    return VARIANTS[name]


def get_leave_one_out_variants() -> list[str]:
    """Return leave-one-out preset: full refs + full-minus-one variants."""
    refs = [name for name in ("full", "full_shots") if name in VARIANTS]
    loo = [name for name in VARIANTS if name.startswith("no_") and name != "no_search"]
    return refs + loo


def get_q1_ablation_variants() -> list[str]:
    """Paper Q1 mechanism ablation order: Ours then leave-one-mechanism-out."""
    return [
        "ours",
        "wo_gaps",
        "wo_search",
        "gap_ids_only",
        "wo_path",
        "wo_path_feedback",
        "wo_batch",
        "wo_exec_detail",
    ]
