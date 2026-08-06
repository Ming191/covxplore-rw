"""Reasoning treatment selected for one generation run."""

from enum import StrEnum


class ReasoningTechnique(StrEnum):
    NONE = "none"
    COT = "cot"
    LEAST_TO_MOST = "least_to_most"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    PROGRAM_OF_THOUGHTS = "program_of_thoughts"
