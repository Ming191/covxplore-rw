from __future__ import annotations

from pydantic import BaseModel, Field

from covxplore.types.coverage_detail import CoverageDetail
from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.trace_summary import TraceSummary
from covxplore.types.expected_path_step import ExpectedPathStep


class TestResult(BaseModel):
    test_name: str
    test_body: str
    status: str  # PASSED | FAILED | RUNTIME_ERROR | COMPILE_ERROR | UNKNOWN
    execute_log: str | None = None

    statement_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    branch_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    unvisited_statements: list[UnvisitedStatement] = Field(default_factory=list)
    unvisited_branches: list[UnvisitedBranch] = Field(default_factory=list)
    trace_summary: TraceSummary | None = None

    target_node_id: int | None = None
    target_polarity: str | None = None
    target_reason: str | None = None
    # Predicted branch-outcome trace from function entry to the target (see
    # ExpectedPathStep); compared against trace_summary's ordered runtime path.
    expected_path: list[ExpectedPathStep] = Field(default_factory=list)

    new_structural_coverage: int = 0
    is_redundant: bool = False

    token_input: int = 0
    token_output: int = 0
    elapsed_ms: float = 0.0
    accepted_order: int = 0
