from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageMetrics:
    statement_pct: float = 0.0
    branch_pct: float = 0.0
    covered_statements: int = 0
    total_statements: int = 0
    covered_branches: int = 0
    total_branches: int = 0
