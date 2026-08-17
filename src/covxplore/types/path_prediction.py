from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExpectedPathStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: int
    outcome: Literal["TRUE", "FALSE"]

    @property
    def key(self) -> tuple[int, str]:
        return self.node_id, self.outcome


class PathPrediction(BaseModel):
    path_id: str
    expected_path: list[ExpectedPathStep] = Field(min_length=1)
    hypotheses: int
    supported_hypotheses: int = 0
    branch_support_pct: float | None = None
    full_path_match: bool | None = None
    first_unsupported_index: int | None = None

    @classmethod
    def pending(cls, path_id: str, expected_path: list[ExpectedPathStep]) -> "PathPrediction":
        return cls(
            path_id=path_id,
            expected_path=expected_path,
            hypotheses=len(expected_path),
        )
