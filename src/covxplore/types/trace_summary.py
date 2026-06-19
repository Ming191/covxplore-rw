from __future__ import annotations
from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    raw_step_count: int = 0
    target_function_step_count: int = 0
    condition_step_count: int = 0
    unique_condition_offsets: int = 0
    visited_functions: list[str] = Field(default_factory=list)
