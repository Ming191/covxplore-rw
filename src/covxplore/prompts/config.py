"""Reasoning treatment selected for one generation run."""

from enum import StrEnum


class ReasoningTechnique(StrEnum):
    NONE = "none"
    COT = "cot"
    PATH_GUIDED = "path_guided"
