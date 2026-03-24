"""
Core data structures for covxplore.

TestResult  — one executed test case + its coverage impact on the suite.
TestSuite   — aggregate state of all generated tests, MC/DC bookkeeping.

Design notes
------------
* A ConditionKey uniquely identifies one (condition-expression, branch-polarity) pair
  that must be independently exercised for full MC/DC coverage.
* TestSuite.add_result() is the only place ConditionKeys are added to the global
  covered set, so delta calculations are always consistent.
* Redundancy is decided at add-time: a test is redundant iff it covers 0 new keys.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitives mirroring the AkaUT REST response shape
# ---------------------------------------------------------------------------

class CoverageDetail(BaseModel):
    visited: int = 0
    total: int = 0
    progress: float = 0.0

    @property
    def pct(self) -> float:
        return self.progress


class UnvisitedMcdc(BaseModel):
    condition: str
    true_branch_visited: bool
    false_branch_visited: bool


class ConditionTraceEntry(BaseModel):
    condition: str
    true_branch_visited: bool
    false_branch_visited: bool
    line_in_function: int | None = None
    start_offset_in_function: int | None = None
    end_offset_in_function: int | None = None


class TraceSummary(BaseModel):
    raw_step_count: int = 0
    target_function_step_count: int = 0
    condition_step_count: int = 0
    unique_condition_offsets: int = 0
    visited_functions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# A unique key for one half of an MC/DC independence pair
# (condition text, polarity that was exercised)
# ---------------------------------------------------------------------------

class ConditionKey(NamedTuple):
    """Identifies one (condition, polarity) pair within the target function."""
    condition: str
    polarity: bool   # True → true-branch exercised; False → false-branch


# ---------------------------------------------------------------------------
# TestResult — one executed test case
# ---------------------------------------------------------------------------

class TestResult(BaseModel):
    test_name: str
    test_body: str
    status: str                                  # PASSED | FAILED | COMPILE_ERROR | UNKNOWN
    execute_log: str | None = None

    statement_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    branch_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    mcdc_coverage: CoverageDetail = Field(default_factory=CoverageDetail)

    unvisited_mcdc: list[UnvisitedMcdc] = Field(default_factory=list)
    condition_trace: list[ConditionTraceEntry] = Field(default_factory=list)
    trace_summary: TraceSummary | None = None

    # Metrics set by TestSuite.add_result()
    new_mcdc_pairs_covered: int = 0
    is_redundant: bool = False

    # Token / timing bookkeeping (set by ExecuteTestcaseTool)
    token_input: int = 0
    token_output: int = 0
    elapsed_ms: float = 0.0
    iteration: int = 0

    def condition_keys(self) -> set[ConditionKey]:
        """Return all (condition, polarity) keys covered by this test."""
        keys: set[ConditionKey] = set()
        for entry in self.unvisited_mcdc:
            if entry.true_branch_visited:
                keys.add(ConditionKey(entry.condition, True))
            if entry.false_branch_visited:
                keys.add(ConditionKey(entry.condition, False))
        return keys


# ---------------------------------------------------------------------------
# TestSuite — aggregate state across all generated tests
# ---------------------------------------------------------------------------

@dataclass
class TestSuite:
    """Tracks the live state of all generated tests during a generation run.

    Only TestResult objects whose ``status`` is not COMPILE_ERROR are
    considered for coverage bookkeeping (they still go into ``tests``).
    """

    function_path: str

    tests: list[TestResult] = field(default_factory=list)
    covered_keys: set[ConditionKey] = field(default_factory=set)
    total_mcdc_conditions: int = 0   # total unique ConditionKeys possible (set after first exec)
    iteration_count: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------ #
    # Core mutation                                                        #
    # ------------------------------------------------------------------ #

    def add_result(self, result: TestResult, min_suite_size: int = 3) -> None:
        """Register a new TestResult and update suite-level coverage tracking."""
        self.iteration_count += 1
        result.iteration = self.iteration_count

        if result.status not in ("COMPILE_ERROR",):
            new_keys = result.condition_keys() - self.covered_keys
            result.new_mcdc_pairs_covered = len(new_keys)
            result.is_redundant = (
                len(new_keys) == 0
                and len(self.tests) >= min_suite_size
            )
            self.covered_keys |= new_keys

            # Update total from the first passing result that has trace data
            if self.total_mcdc_conditions == 0 and result.condition_trace:
                # Total possible keys = 2 per condition (true + false)
                unique_conditions = {e.condition for e in result.condition_trace}
                self.total_mcdc_conditions = len(unique_conditions) * 2

        self.tests.append(result)

    # ------------------------------------------------------------------ #
    # Computed metrics                                                     #
    # ------------------------------------------------------------------ #

    @property
    def mcdc_coverage_pct(self) -> float:
        if self.total_mcdc_conditions == 0:
            return 0.0
        return len(self.covered_keys) / self.total_mcdc_conditions

    @property
    def redundancy_rate(self) -> float:
        if not self.tests:
            return 0.0
        redundant = sum(1 for t in self.tests if t.is_redundant)
        return redundant / len(self.tests)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.token_input for t in self.tests)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.token_output for t in self.tests)

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def non_redundant_tests(self) -> list[TestResult]:
        return [t for t in self.tests if not t.is_redundant]

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def unvisited_summary(self) -> list[dict]:
        """Return the real unvisited MC/DC conditions by combining the 
        function's full condition list with the suite's covered_keys."""
        if not self.tests:
            return []
        last = self.tests[-1]
        
        result = []
        for u in last.unvisited_mcdc:
            globally_covered_true = ConditionKey(u.condition, True) in self.covered_keys
            globally_covered_false = ConditionKey(u.condition, False) in self.covered_keys
            
            needs_true = not globally_covered_true
            needs_false = not globally_covered_false
            
            if needs_true or needs_false:
                result.append({
                    "condition": u.condition,
                    "needs_true": needs_true,
                    "needs_false": needs_false,
                })
        return result

    def coverage_gap_prompt_fragment(self) -> str:
        """Return a terse, prompt-ready description of what still needs covering.

        This is the CoverAgent-style local gap injection (arxiv:2402.09171):
        *specific condition at specific location* rather than a global goal.
        """
        unvisited = self.unvisited_summary()
        if not unvisited:
            return "All MC/DC conditions appear covered — try to verify with edge-case inputs."

        lines = ["The following MC/DC condition polarities are NOT yet covered:"]
        for item in unvisited[:8]:   # cap at 8 to stay within token budget
            missing = []
            if item["needs_true"]:
                missing.append("TRUE branch")
            if item["needs_false"]:
                missing.append("FALSE branch")
            lines.append(f"  • {item['condition']!r} — missing: {', '.join(missing)}")
        lines.append(
            "\nTarget the FIRST uncovered condition. Generate a test where ONLY that "
            "condition's value changes relative to an existing passing test (MC/DC independence rule)."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "function_path": self.function_path,
            "iteration_count": self.iteration_count,
            "mcdc_coverage_pct": round(self.mcdc_coverage_pct, 4),
            "covered_mcdc_pairs": len(self.covered_keys),
            "total_mcdc_pairs": self.total_mcdc_conditions,
            "redundancy_rate": round(self.redundancy_rate, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "tests": [t.model_dump() for t in self.tests],
        }
