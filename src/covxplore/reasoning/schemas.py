from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from covxplore.agents.schemas import GenerateTestAction


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChainOfThoughtResult(_StrictModel):
    reasoning: str = Field(..., min_length=1)
    candidates: list[GenerateTestAction] = Field(..., min_length=1, max_length=8)


class PathTargetedTest(GenerateTestAction):
    target_node_id: int
    target_outcome: Literal["TRUE", "FALSE"]


class PathGuidedResult(_StrictModel):
    candidates: list[PathTargetedTest] = Field(..., min_length=1, max_length=5)


class ContextAction(_StrictModel):
    action: Literal["search_nodes", "get_node_source", "done"]
    query: str | None = None
    types: list[str] = Field(default_factory=list, max_length=8)
    absolute_path: str | None = None
