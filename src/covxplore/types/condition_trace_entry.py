from __future__ import annotations
from pydantic import BaseModel


class ConditionTraceEntry(BaseModel):
    node_id: int | None = None
    condition: str
    true_branch_visited: bool
    false_branch_visited: bool
    line_in_function: int | None = None
    start_offset_in_function: int | None = None
    end_offset_in_function: int | None = None

    def identity(self) -> int:
        if self.node_id is None:
            raise RuntimeError(
                "Backend payload missing nodeId in conditionTrace. "
                "nodeId is required for MC/DC identity."
            )
        return self.node_id
