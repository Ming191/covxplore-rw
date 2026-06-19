from __future__ import annotations
from typing import NamedTuple


class ConditionKey(NamedTuple):
    condition_id: int
    polarity: bool
