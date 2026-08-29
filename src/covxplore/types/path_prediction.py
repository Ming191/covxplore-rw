from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_serializer


class ExpectedPathStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: int
    outcome: Literal["TRUE", "FALSE"]

    @property
    def key(self) -> tuple[int, str]:
        return self.node_id, self.outcome

    @property
    def compact(self) -> str:
        return f"{self.node_id}{'T' if self.outcome == 'TRUE' else 'F'}"


_COMPACT_STEP = re.compile(r"^(\d+)([TF])$", re.IGNORECASE)


def parse_compact_expected_path(value: str) -> list[ExpectedPathStep]:
    """Parse ``6F-55F-61T`` into structured path steps."""
    text = value.strip()
    if not text:
        raise ValueError("expected_path must not be blank")
    steps: list[ExpectedPathStep] = []
    for token in text.split("-"):
        match = _COMPACT_STEP.fullmatch(token.strip())
        if match is None:
            raise ValueError(
                f"invalid expected_path token {token!r}; use nodeId+T/F segments like '6F-55T'"
            )
        node_id, outcome = match.groups()
        steps.append(
            ExpectedPathStep(
                node_id=int(node_id),
                outcome="TRUE" if outcome.upper() == "T" else "FALSE",
            )
        )
    return steps


def format_compact_expected_path(steps: list[ExpectedPathStep]) -> str:
    return "-".join(step.compact for step in steps)


def coerce_expected_path(value: Any) -> list[ExpectedPathStep]:
    if isinstance(value, str):
        return parse_compact_expected_path(value)
    if isinstance(value, list):
        if not value:
            raise ValueError("expected_path must not be empty")
        return [
            item
            if isinstance(item, ExpectedPathStep)
            else ExpectedPathStep.model_validate(item)
            for item in value
        ]
    raise TypeError("expected_path must be a compact string or step list")


ExpectedPath = Annotated[
    list[ExpectedPathStep],
    BeforeValidator(coerce_expected_path),
    Field(min_length=1),
]


class PathPrediction(BaseModel):
    path_id: str
    expected_path: ExpectedPath
    hypotheses: int
    supported_hypotheses: int = 0
    branch_support_pct: float | None = None
    full_path_match: bool | None = None
    first_unsupported_index: int | None = None

    @field_serializer("expected_path")
    def _serialize_expected_path(self, steps: list[ExpectedPathStep]) -> str:
        return format_compact_expected_path(steps)

    @classmethod
    def pending(cls, path_id: str, expected_path: list[ExpectedPathStep]) -> "PathPrediction":
        return cls(
            path_id=path_id,
            expected_path=expected_path,
            hypotheses=len(expected_path),
        )
