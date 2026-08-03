from __future__ import annotations
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuntimeValue(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    expression: str
    value: Any = None
    type: str | None = None


class TargetFunctionConditionStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    line: int | None = None
    start: int | None = None
    end: int | None = None
    full: str | None = None
    sub: str | None = None
    node_id: int | None = Field(default=None, alias="nodeId")
    branch: str | bool | None = None
    decision_role: str | None = Field(default=None, alias="decisionRole")
    decision_id: str | None = Field(default=None, alias="decisionId")
    condition_index: int | None = Field(default=None, alias="conditionIndex")
    condition_value: bool | None = Field(default=None, alias="conditionValue")
    decision_value: bool | None = Field(default=None, alias="decisionValue")
    runtime_values: list[RuntimeValue] = Field(default_factory=list, alias="runtimeValues")


class TraceSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    raw_step_count: int = Field(default=0, alias="rawStepCount")
    target_function_step_count: int = Field(default=0, alias="targetFunctionStepCount")
    condition_step_count: int = Field(default=0, alias="conditionStepCount")
    unique_condition_offsets: int = Field(default=0, alias="uniqueConditionOffsets")
    visited_functions: list[str] = Field(default_factory=list, alias="visitedFunctions")
    target_function_condition_steps: list[TargetFunctionConditionStep] = Field(
        default_factory=list,
        alias="targetFunctionConditionSteps",
    )
