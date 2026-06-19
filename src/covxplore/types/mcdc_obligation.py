from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class McdcObligation:
    condition_id: int
    condition: str
    line_in_function: int | None
    needs_true: bool
    needs_false: bool

    @property
    def true_visited(self) -> bool:
        return not self.needs_true

    @property
    def false_visited(self) -> bool:
        return not self.needs_false
