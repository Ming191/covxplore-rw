from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, Field


class CoverageDetail(BaseModel):
    visited: int = 0
    total: int = 0
    progress: float = 0.0

    @property
    def pct(self) -> float:
        return self.progress


class UnvisitedMcdc(BaseModel):
    node_id: int | None = None
    condition: str
    true_branch_visited: bool
    false_branch_visited: bool

    def identity(self) -> int:
        if self.node_id is None:
            raise RuntimeError(
                "Backend payload missing nodeId in unvisitedMcdcConditions. "
                "nodeId is required for MC/DC identity."
            )
        return self.node_id


class UnvisitedStatement(BaseModel):
    node_id: int | None = None
    statement: str
    line_in_function: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class UnvisitedBranch(BaseModel):
    node_id: int | None = None
    condition: str
    true_visited: bool
    false_visited: bool
    line_in_function: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class ConditionTraceEntry(BaseModel):
    node_id: int | None = None
    condition: str
    true_branch_visited: bool
    false_branch_visited: bool
    line_in_function: int | None = None
    start_offset_in_function: int | None = None
    end_offset_in_function: int | None = None

    def identity(self) -> int:
        if self.node_id is None:
            raise RuntimeError(
                "Backend payload missing nodeId in conditionTrace. "
                "nodeId is required for MC/DC identity."
            )
        return self.node_id


class TraceSummary(BaseModel):
    raw_step_count: int = 0
    target_function_step_count: int = 0
    condition_step_count: int = 0
    unique_condition_offsets: int = 0
    visited_functions: list[str] = Field(default_factory=list)


class ConditionKey(NamedTuple):
    condition_id: int
    polarity: bool


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
    new_statements_covered: int = 0
    new_branches_covered: int = 0
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
