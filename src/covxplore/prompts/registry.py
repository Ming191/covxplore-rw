"""Reasoning techniques available to generation experiments."""

from covxplore.prompts.config import ReasoningTechnique


VARIANTS: dict[str, ReasoningTechnique] = {
    technique.value: technique for technique in ReasoningTechnique
}


def get_variant(name: str) -> ReasoningTechnique:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(VARIANTS)
        raise KeyError(f"Unknown reasoning technique {name!r}. Available: {available}") from exc


def get_reasoning_variants() -> list[str]:
    return list(VARIANTS)
