from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class McdcObligation:
    condition_id: int
    condition: str
    line_in_function: int | None
    needs_true: bool
    needs_false: bool
