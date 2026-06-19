from __future__ import annotations
from pydantic import BaseModel


class UnvisitedStatement(BaseModel):
    node_id: int | None = None
    statement: str
    line_in_function: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
