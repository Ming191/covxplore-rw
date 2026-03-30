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

from covxplore.status import TestStatus, normalize_test_status


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


# ---------------------------------------------------------------------------
# A unique key for one half of an MC/DC independence pair
# (condition text, polarity that was exercised)
# ---------------------------------------------------------------------------


class ConditionKey(NamedTuple):
    """Identifies one (condition-identity, polarity) pair within target function."""

    condition_id: int
    polarity: bool  # True → true-branch exercised; False → false-branch


# ---------------------------------------------------------------------------
# TestResult — one executed test case
# ---------------------------------------------------------------------------


class TestResult(BaseModel):
    test_name: str
    test_body: str
    status: str  # PASSED | FAILED | RUNTIME_ERROR | COMPILE_ERROR | UNKNOWN
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
        """Return (condition, polarity) keys visited by this test."""
        keys: set[ConditionKey] = set()
        if self.unvisited_mcdc:
            for entry in self.unvisited_mcdc:
                cond_id = entry.identity()
                if entry.true_branch_visited:
                    keys.add(ConditionKey(cond_id, True))
                if entry.false_branch_visited:
                    keys.add(ConditionKey(cond_id, False))
            return keys

        for entry in self.condition_trace:
            cond_id = entry.identity()
            if entry.true_branch_visited:
                keys.add(ConditionKey(cond_id, True))
            if entry.false_branch_visited:
                keys.add(ConditionKey(cond_id, False))
        return keys


# ---------------------------------------------------------------------------
# TestSuite — aggregate state across all generated tests
# ---------------------------------------------------------------------------


@dataclass
class TestSuite:
    """Tracks the live state of all generated tests during a generation run.

    Only TestResult objects whose ``status`` is PASSED or RUNTIME_ERROR are
    considered for coverage bookkeeping (they still go into ``tests``).
    FAILED is tracked for diagnostics but intentionally excluded from coverage deltas.
    """

    function_path: str

    tests: list[TestResult] = field(default_factory=list)
    covered_keys: set[ConditionKey] = field(default_factory=set)
    total_mcdc_conditions: int = 0
    all_conditions: list[str] = field(default_factory=list)
    condition_id_to_text: dict[int, str] = field(default_factory=dict)
    condition_id_to_line: dict[int, int | None] = field(default_factory=dict)
    iteration_count: int = 0
    consecutive_redundant: int = 0  # reset to 0 after any non-redundant result
    started_at: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------ #
    # Core mutation                                                        #
    # ------------------------------------------------------------------ #

    def add_result(self, result: TestResult, min_suite_size: int = 3) -> None:
        """Register a new TestResult and update suite-level coverage tracking."""

        self.iteration_count += 1
        result.iteration = self.iteration_count

        normalized = normalize_test_status(result.status)
        if normalized in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}:
            raw_keys = result.condition_keys()

            new_keys = raw_keys - self.covered_keys
            result.new_mcdc_pairs_covered = len(new_keys)
            result.is_redundant = (
                len(new_keys) == 0 and len(self.tests) >= min_suite_size
            )
            self.covered_keys |= new_keys

            if result.is_redundant:
                self.consecutive_redundant += 1
            else:
                self.consecutive_redundant = 0

            # Discover condition identities from execution response.
            discovered_ids: dict[int, str] = {}
            if result.condition_trace:
                for e in result.condition_trace:
                    discovered_ids[e.identity()] = e.condition.strip()
                    self.condition_id_to_line[e.identity()] = e.line_in_function
            if result.unvisited_mcdc:
                for u in result.unvisited_mcdc:
                    discovered_ids[u.identity()] = u.condition.strip()

            for cid, ctext in discovered_ids.items():
                if cid not in self.condition_id_to_text:
                    self.condition_id_to_text[cid] = ctext

            self.all_conditions = [
                self.condition_id_to_text[cid] for cid in self.condition_id_to_text
            ]

            # Update total_mcdc_conditions from execution response if not already set
            if self.total_mcdc_conditions == 0 and result.mcdc_coverage.total > 0:
                self.total_mcdc_conditions = result.mcdc_coverage.total

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
        """Return unvisited MC/DC conditions against suite's covered_keys.
        Requires all_conditions to be pre-populated by _prefetch_conditions().
        """
        result = []
        for cond_id, cond in self.condition_id_to_text.items():
            key_true = ConditionKey(cond_id, True)
            key_false = ConditionKey(cond_id, False)
            needs_true = key_true not in self.covered_keys
            needs_false = key_false not in self.covered_keys
            if needs_true or needs_false:
                result.append(
                    {
                        "condition_id": cond_id,
                        "condition": cond,
                        "line_in_function": self.condition_id_to_line.get(cond_id),
                        "needs_true": needs_true,
                        "needs_false": needs_false,
                    }
                )
        return result

    def coverage_gap_prompt_fragment(self) -> str:
        """Return a terse, prompt-ready description of what still needs covering.

        This is the CoverAgent-style local gap injection (arxiv:2402.09171):
        *specific condition at specific location* rather than a global goal.
        """
        if not self.tests:
            if self.total_mcdc_conditions > 0:
                return (
                    f"No tests executed yet. Target is {self.total_mcdc_conditions} MC/DC pairs. "
                    "Do NOT stop. Generate and execute a first compilable test case."
                )
            return (
                "No tests executed yet and total MC/DC target is unknown. "
                "Do NOT stop. Call get_conditions_static, then execute a compilable test."
            )

        last = self.tests[-1]
        if last.status == TestStatus.COMPILE_ERROR.value:
            return (
                "Last execution status is COMPILE_ERROR. Coverage is not complete. "
                "Do NOT stop. Fix compilation issues and re-run execute_testcase."
            )
        if last.status == TestStatus.FAILED.value:
            return (
                "Last execution status is FAILED. Coverage is not complete. "
                "Inspect execute_log and continue with the next test."
            )
        if last.status == TestStatus.UNKNOWN.value:
            return (
                "Last execution status is UNKNOWN. Coverage state may be incomplete. "
                "Do NOT stop. Re-run with a valid test and continue."
            )

        if (
            self.total_mcdc_conditions > 0
            and len(self.covered_keys) >= self.total_mcdc_conditions
        ):
            return "SUCCESS! All MC/DC conditions are now covered (100%). DO NOT call any more tools. Please output your final 'DONE: ...' message immediately to finish the task."

        unvisited = self.unvisited_summary()
        if not unvisited:
            remaining = max(self.total_mcdc_conditions - len(self.covered_keys), 0)
            raise RuntimeError(
                "Unable to compute uncovered conditions while MC/DC pairs remain. "
                f"remaining={remaining}, all_conditions={len(self.all_conditions)}, "
                f"unique_condition_ids={len(self.condition_id_to_text)}, "
                f"covered_keys={len(self.covered_keys)}, total_mcdc={self.total_mcdc_conditions}. "
                "This indicates inconsistent condition identity between static and execution data."
            )

        lines = ["The following MC/DC condition polarities are NOT yet covered:"]
        for item in unvisited:
            missing = []
            if item["needs_true"]:
                missing.append("TRUE branch")
            if item["needs_false"]:
                missing.append("FALSE branch")
            line_tag = (
                f"line+{item['line_in_function']}"
                if item["line_in_function"] is not None
                else "line+?"
            )
            lines.append(
                f"  • [{line_tag}] {item['condition']!r} — missing: {', '.join(missing)}"
            )

        # Warn the LLM when it's producing a run of redundant tests
        if self.consecutive_redundant >= 3:
            lines.append(
                "\nWARNING: You have produced 3 consecutive REDUNDANT tests (0 new MC/DC pairs). "
                "You are stuck. DO NOT call any more tools. "
                "Output your final 'DONE: <summary>' message immediately to finish the task."
            )
        elif self.consecutive_redundant == 2:
            lines.append(
                "\nCAUTION: 2 consecutive redundant tests. You are likely stuck on the same path. "
                "Change your approach completely: try different variable values, operator boundaries, "
                "or a different code path entirely. If you cannot cover the remaining conditions, "
                "declare DONE on the next step."
            )

        lines.append(
            "\nTarget the test path that satisfies the highest number of conditions. "
            "Prioritize paths that cover multiple uncovered conditions simultaneously. "
            "If multiple paths are possible, choose the one that increases overall condition coverage the most. "
            "Prefer modifying an existing passing test when possible, but allow generating a new test if needed."
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
