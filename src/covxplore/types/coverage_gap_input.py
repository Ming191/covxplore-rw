from __future__ import annotations

from dataclasses import dataclass, field

from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.coverage_metrics import CoverageMetrics
from covxplore.types.test_result import TestResult


@dataclass(frozen=True)
class CoverageGapInput:
    tests: list[TestResult] = field(default_factory=list)
    metrics: CoverageMetrics = field(default_factory=CoverageMetrics)
    cumulative_unvisited_statements: list[UnvisitedStatement] = field(default_factory=list)
    cumulative_unvisited_branches: list[UnvisitedBranch] = field(default_factory=list)
    consecutive_redundant: int = 0
    batch_count: int = 0
