"""Prompt variants available to the app.

The app ships the production prompt configuration. The full ablation matrix
(baseline, single-section, and leave-one-out variants) lives on the
experimental ``playground`` branch.
"""
from covxplore.prompts.config import PromptConfig

VARIANTS: dict[str, PromptConfig] = {
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
}


def get_variant(name: str) -> PromptConfig:
    if name not in VARIANTS:
        available = ", ".join(sorted(VARIANTS))
        raise KeyError(f"Unknown prompt variant {name!r}. Available: {available}")
    return VARIANTS[name]
