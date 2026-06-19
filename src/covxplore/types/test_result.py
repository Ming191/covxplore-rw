from __future__ import annotations

from pydantic import BaseModel, Field

from covxplore.types.condition_key import ConditionKey
from covxplore.types.coverage_detail import CoverageDetail
from covxplore.types.unvisited_mcdc import UnvisitedMcdc
from covxplore.types.unvisited_statement import UnvisitedStatement
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.condition_trace_entry import ConditionTraceEntry
from covxplore.types.trace_summary import TraceSummary


class TestResult(BaseModel):
    test_name: str
    test_body: str
    status: str  # PASSED | FAILED | RUNTIME_ERROR | COMPILE_ERROR | UNKNOWN
    execute_log: str | None = None

    statement_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    branch_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    mcdc_coverage: CoverageDetail = Field(default_factory=CoverageDetail)

    unvisited_mcdc: list[UnvisitedMcdc] = Field(default_factory=list)
    unvisited_statements: list[UnvisitedStatement] = Field(default_factory=list)
    unvisited_branches: list[UnvisitedBranch] = Field(default_factory=list)
    condition_trace: list[ConditionTraceEntry] = Field(default_factory=list)
    trace_summary: TraceSummary | None = None

    new_mcdc_pairs_covered: int = 0
    is_redundant: bool = False

    token_input: int = 0
    token_output: int = 0
    elapsed_ms: float = 0.0
    iteration: int = 0

    def condition_keys(self) -> set[ConditionKey]:
        """Return (condition, polarity) keys visited by this test."""
        keys: set[ConditionKey] = set()
        for entry in self.condition_trace:
            cond_id = entry.identity()
            if entry.true_branch_visited:
                keys.add(ConditionKey(cond_id, True))
            if entry.false_branch_visited:
                keys.add(ConditionKey(cond_id, False))

        for entry in self.unvisited_mcdc:
            cond_id = entry.identity()
            if entry.true_branch_visited:
                keys.add(ConditionKey(cond_id, True))
            if entry.false_branch_visited:
                keys.add(ConditionKey(cond_id, False))

        return keys
