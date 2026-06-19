from __future__ import annotations
from pydantic import BaseModel


class CoverageDetail(BaseModel):
    visited: int = 0
    total: int = 0
    progress: float = 0.0

    @property
    def pct(self) -> float:
        return self.progress
