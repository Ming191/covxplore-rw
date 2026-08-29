from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from covxplore.agents.schemas import GenerateTestAction, PathGuidedTestAction
from covxplore.types.path_prediction import ExpectedPath


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratedTestAction(_StrictModel):
    """One LLM-produced driver body represented as source lines."""

    test_body_lines: list[str] = Field(..., min_length=1)
    test_name: str | None = None

    @field_validator("test_body_lines")
    @classmethod
    def _lines_not_blank(cls, value: list[str]) -> list[str]:
        if not any(line.strip() for line in value):
            raise ValueError("must contain a non-blank line")
        return value

    def to_action(self) -> GenerateTestAction:
        return GenerateTestAction(
            test_name=self.test_name,
            test_body="\n".join(self.test_body_lines),
        )


class GeneratedTestBatchResult(_StrictModel):
    candidates: list[GeneratedTestAction] = Field(..., min_length=1, max_length=8)

    def to_actions(self) -> list[GenerateTestAction]:
        return [candidate.to_action() for candidate in self.candidates]


class ChainOfThoughtResult(GeneratedTestBatchResult):
    reasoning: str = Field(..., min_length=1)


class PathGuidedGeneratedAction(_StrictModel):
    """Path-guided candidate with source lines plus path metadata."""

    test_body_lines: list[str] = Field(..., min_length=1)
    test_name: str | None = None
    path_id: str = Field(..., min_length=1)
    expected_path: ExpectedPath

    @field_validator("test_body_lines")
    @classmethod
    def _lines_not_blank(cls, value: list[str]) -> list[str]:
        if not any(line.strip() for line in value):
            raise ValueError("must contain a non-blank line")
        return value

    def to_action(self) -> PathGuidedTestAction:
        return PathGuidedTestAction(
            test_name=self.test_name,
            test_body="\n".join(self.test_body_lines),
            path_id=self.path_id,
            expected_path=self.expected_path,
        )


class PathGuidedPlanningResult(_StrictModel):
    reasoning: str = Field(..., min_length=1)
    candidates: list[PathGuidedGeneratedAction] = Field(..., min_length=2, max_length=5)

    def to_actions(self) -> list[PathGuidedTestAction]:
        return [candidate.to_action() for candidate in self.candidates]


class ContextAction(_StrictModel):
    action: Literal["search_nodes", "get_node_source", "done"]
    query: str | None = None
    types: list[str] = Field(default_factory=list, max_length=8)
    absolute_path: str | None = None
