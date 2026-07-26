"""Shared types — no circular dependency on coverage or models."""

from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.coverage_detail import CoverageDetail
from covxplore.types.trace_summary import TraceSummary
from covxplore.types.expected_path_step import ExpectedPathStep
from covxplore.types.test_result import TestResult
from covxplore.types.coverage_metrics import CoverageMetrics
from covxplore.types.coverage_gap_input import CoverageGapInput
from covxplore.types.coverage_gap import CoverageGap
from covxplore.types.coverage_readable import CoverageReadable
from covxplore.types.test_suite import TestSuite

__all__ = [
    "CoverageDetail",
    "CoverageGap",
    "CoverageGapInput",
    "CoverageMetrics",
    "CoverageReadable",
    "ExpectedPathStep",
    "TestResult",
    "TestSuite",
    "TraceSummary",
    "UnvisitedBranch",
    "UnvisitedStatement",
]
