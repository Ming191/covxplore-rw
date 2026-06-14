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
    # Quick-win optimized variants                                        #
    # ------------------------------------------------------------------ #
    "batch_fast": PromptConfig(
        "batch_fast",
        role_persona=True,
        cot_reasoning=False,
        coverage_guidance=True,
        self_reflection=False,
        few_shot_examples=False,
        output_format=True,
        batch_generation=True,
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
    loo = [name for name in VARIANTS if name.startswith("no_")]
    return refs + loo
