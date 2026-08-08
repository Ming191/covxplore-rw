from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from covxplore.agents.schemas import GenerateTestAction


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChainOfThoughtResult(_StrictModel):
    reasoning: str = Field(..., min_length=1)
    candidates: list[GenerateTestAction] = Field(..., min_length=1, max_length=5)


class Decomposition(_StrictModel):
    subproblems: list[str] = Field(..., min_length=1, max_length=6)


class SubproblemSolution(_StrictModel):
    analysis: str = Field(..., min_length=1)
    scenarios: list[str] = Field(..., min_length=1, max_length=8)


class ThoughtPlan(_StrictModel):
    plan_id: str = Field(..., min_length=1)
    approach: str = Field(..., min_length=1)
    target_gaps: list[str] = Field(..., min_length=1)
    setup_strategy: str = Field(..., min_length=1)


class ThoughtPlans(_StrictModel):
    plans: list[ThoughtPlan] = Field(..., min_length=3, max_length=5)


class ThoughtScore(_StrictModel):
    plan_id: str = Field(..., min_length=1)
    score: int = Field(..., ge=0, le=10)
    rationale: str = Field(..., min_length=1)


class ThoughtEvaluation(_StrictModel):
    scores: list[ThoughtScore] = Field(..., min_length=1)


class ProgramOfThoughts(_StrictModel):
    program: str = Field(..., min_length=1)


class ProgramScenarios(_StrictModel):
    scenarios: list[dict[str, Any]] = Field(..., min_length=1, max_length=100)
