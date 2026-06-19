"""Shared types — no circular dependency on coverage or models."""

from covxplore.types.condition_key import ConditionKey
from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.coverage_detail import CoverageDetail
from covxplore.types.unvisited_mcdc import UnvisitedMcdc
from covxplore.types.condition_trace_entry import ConditionTraceEntry
from covxplore.types.trace_summary import TraceSummary
from covxplore.types.test_result import TestResult
from covxplore.types.coverage_metrics import CoverageMetrics
from covxplore.types.mcdc_obligation import McdcObligation
from covxplore.types.coverage_gap_input import CoverageGapInput
from covxplore.types.coverage_gap import CoverageGap
from covxplore.types.coverage_readable import CoverageReadable
from covxplore.types.test_suite import TestSuite

__all__ = [
    "ConditionKey",
    "CoverageDetail",
    "CoverageGap",
    "CoverageGapInput",
    "CoverageMetrics",
    "CoverageReadable",
    "ConditionTraceEntry",
    "McdcObligation",
    "TestResult",
    "TestSuite",
    "TraceSummary",
    "UnvisitedBranch",
    "UnvisitedMcdc",
    "UnvisitedStatement",
]
