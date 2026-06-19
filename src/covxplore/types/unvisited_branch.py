from __future__ import annotations
from pydantic import BaseModel


class UnvisitedBranch(BaseModel):
    node_id: int | None = None
    condition: str
    true_visited: bool
    false_visited: bool
    line_in_function: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
