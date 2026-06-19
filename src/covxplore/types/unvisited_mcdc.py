from __future__ import annotations
from pydantic import BaseModel


class UnvisitedMcdc(BaseModel):
    node_id: int | None = None
    condition: str
    true_branch_visited: bool
    false_branch_visited: bool

    def identity(self) -> int:
        if self.node_id is None:
            raise RuntimeError(
                "Backend payload missing nodeId in unvisitedMcdcConditions. "
                "nodeId is required for MC/DC identity."
            )
        return self.node_id
