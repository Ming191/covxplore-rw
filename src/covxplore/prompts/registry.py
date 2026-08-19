"""Reasoning techniques available to generation experiments."""

from covxplore.prompts.config import ReasoningTechnique


VARIANTS: dict[str, ReasoningTechnique] = {
    technique.value: technique for technique in ReasoningTechnique
}
VARIANTS["path-guide"] = ReasoningTechnique.PATH_GUIDED
VARIANTS["path-guided"] = ReasoningTechnique.PATH_GUIDED
VARIANTS["path_guide"] = ReasoningTechnique.PATH_GUIDED


def get_variant(name: str) -> ReasoningTechnique:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in VARIANTS:
        return VARIANTS[normalized]
    try:
        return VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(VARIANTS)
        raise KeyError(f"Unknown reasoning technique {name!r}. Available: {available}") from exc


def get_reasoning_variants() -> list[str]:
    return list(VARIANTS)
