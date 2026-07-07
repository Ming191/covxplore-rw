from __future__ import annotations

from dataclasses import dataclass, field

from covxplore.types.condition_key import ConditionKey
from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.coverage_metrics import CoverageMetrics
from covxplore.types.mcdc_obligation import McdcObligation
from covxplore.types.test_result import TestResult


@dataclass(frozen=True)
class CoverageGapInput:
    tests: list[TestResult] = field(default_factory=list)
    metrics: CoverageMetrics = field(default_factory=CoverageMetrics)
    obligations: list[McdcObligation] = field(default_factory=list)
    observed_polarity_counts: dict[ConditionKey, int] = field(default_factory=dict)
    cumulative_unvisited_statements: list[UnvisitedStatement] = field(default_factory=list)
    cumulative_unvisited_branches: list[UnvisitedBranch] = field(default_factory=list)
    all_conditions_count: int = 0
    unique_condition_ids: int = 0
    consecutive_redundant: int = 0
    batch_count: int = 0
