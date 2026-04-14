from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple

from pydantic import BaseModel, Field

from covxplore.status import TestStatus, normalize_test_status


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


@dataclass
class TestSuite:
    function_path: str

    tests: list[TestResult] = field(default_factory=list)
    covered_keys: set[ConditionKey] = field(default_factory=set)
    total_mcdc_conditions: int = 0
    all_conditions: list[str] = field(default_factory=list)
    condition_id_to_text: dict[int, str] = field(default_factory=dict)
    condition_id_to_line: dict[int, int | None] = field(default_factory=dict)
    # Statement/branch cumulative (intersection) tracking
    # None = no successful test run yet; set = intersection of unvisited node ids across all tests
    cumulative_uncovered_stmt_ids: set[int] | None = field(default=None)
    cumulative_uncovered_branch_keys: set[tuple[int, bool]] | None = field(default=None)
    _stmt_node_info: dict[int, UnvisitedStatement] = field(default_factory=dict)
    _branch_node_info: dict[int, UnvisitedBranch] = field(default_factory=dict)
    total_statements: int = 0
    total_branches: int = 0
    iteration_count: int = 0
    consecutive_redundant: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def add_result(self, result: TestResult, min_suite_size: int = 3) -> None:
        self.iteration_count += 1
        result.iteration = self.iteration_count

        normalized = normalize_test_status(result.status)
        if normalized in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}:
            raw_keys = result.condition_keys()

            new_keys = raw_keys - self.covered_keys
            result.new_mcdc_pairs_covered = len(new_keys)
            # Only flag as redundant for MCDC functions; for stmt/branch-only
            # functions every test may add 0 MCDC pairs by design.
            result.is_redundant = (
                self.total_mcdc_conditions > 0
                and len(new_keys) == 0
                and len(self.tests) >= min_suite_size
            )
            self.covered_keys |= new_keys

            if result.is_redundant:
                self.consecutive_redundant += 1
            else:
                self.consecutive_redundant = 0

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

            if self.total_mcdc_conditions == 0 and result.mcdc_coverage.total > 0:
                self.total_mcdc_conditions = result.mcdc_coverage.total

            # Statement cumulative (intersection) tracking
            if result.statement_coverage.total > 0:
                self.total_statements = result.statement_coverage.total
            test_uncovered_stmt_ids: set[int] = set()
            for s in result.unvisited_statements:
                if s.node_id is not None:
                    test_uncovered_stmt_ids.add(s.node_id)
                    self._stmt_node_info[s.node_id] = s
            if self.cumulative_uncovered_stmt_ids is None:
                self.cumulative_uncovered_stmt_ids = test_uncovered_stmt_ids
            else:
                self.cumulative_uncovered_stmt_ids &= test_uncovered_stmt_ids

            # Branch cumulative (intersection) tracking
            if result.branch_coverage.total > 0:
                self.total_branches = result.branch_coverage.total
            test_uncovered_branch_keys: set[tuple[int, bool]] = set()
            for b in result.unvisited_branches:
                if b.node_id is not None:
                    if not b.true_visited:
                        test_uncovered_branch_keys.add((b.node_id, True))
                    if not b.false_visited:
                        test_uncovered_branch_keys.add((b.node_id, False))
                    self._branch_node_info[b.node_id] = b
            if self.cumulative_uncovered_branch_keys is None:
                self.cumulative_uncovered_branch_keys = test_uncovered_branch_keys
            else:
                self.cumulative_uncovered_branch_keys &= test_uncovered_branch_keys

        self.tests.append(result)

    @property
    def mcdc_coverage_pct(self) -> float:
        if self.total_mcdc_conditions == 0:
            return 0.0
        return len(self.covered_keys) / self.total_mcdc_conditions

    @property
    def statement_coverage_pct(self) -> float:
        if self.total_statements == 0 or self.cumulative_uncovered_stmt_ids is None:
            return 0.0
        covered = self.total_statements - len(self.cumulative_uncovered_stmt_ids)
        return max(0.0, covered) / self.total_statements

    @property
    def branch_coverage_pct(self) -> float:
        if self.total_branches == 0 or self.cumulative_uncovered_branch_keys is None:
            return 0.0
        covered = self.total_branches - len(self.cumulative_uncovered_branch_keys)
        return max(0.0, covered) / self.total_branches

    @property
    def cumulative_unvisited_statements(self) -> list[UnvisitedStatement]:
        """Statements unvisited by every successful test run so far (intersection)."""
        if self.cumulative_uncovered_stmt_ids is None:
            return []
        return [
            self._stmt_node_info[nid]
            for nid in self.cumulative_uncovered_stmt_ids
            if nid in self._stmt_node_info
        ]

    @property
    def cumulative_unvisited_branches(self) -> list[UnvisitedBranch]:
        """Synthetic UnvisitedBranch objects for branch sides unvisited by every test (intersection)."""
        if self.cumulative_uncovered_branch_keys is None:
            return []
        node_missing: dict[int, tuple[bool, bool]] = {}
        for node_id, is_true_side in self.cumulative_uncovered_branch_keys:
            t_miss, f_miss = node_missing.get(node_id, (False, False))
            if is_true_side:
                node_missing[node_id] = (True, f_miss)
            else:
                node_missing[node_id] = (t_miss, True)
        result: list[UnvisitedBranch] = []
        for node_id, (true_missing, false_missing) in node_missing.items():
            if node_id in self._branch_node_info:
                info = self._branch_node_info[node_id]
                result.append(UnvisitedBranch(
                    node_id=node_id,
                    condition=info.condition,
                    true_visited=not true_missing,
                    false_visited=not false_missing,
                    line_in_function=info.line_in_function,
                    start_offset=info.start_offset,
                    end_offset=info.end_offset,
                ))
        return result

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

    def unvisited_summary(self) -> list[dict]:
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
        result.sort(
            key=lambda item: (
                item["line_in_function"] is None,
                item["line_in_function"]
                if item["line_in_function"] is not None
                else float("inf"),
                item["condition_id"],
            )
        )
        return result

    def _observed_polarity_counts(self) -> dict[ConditionKey, int]:
        """Count observed condition polarities across executed tests.

        Uses per-test condition_trace when available. This is used only for
        target-priority heuristics in prompt guidance.
        """
        counts: dict[ConditionKey, int] = {}
        for test in self.tests:
            if not test.condition_trace:
                continue
            for entry in test.condition_trace:
                cond_id = entry.identity()
                if entry.true_branch_visited:
                    key = ConditionKey(cond_id, True)
                    counts[key] = counts.get(key, 0) + 1
                if entry.false_branch_visited:
                    key = ConditionKey(cond_id, False)
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def coverage_gap_prompt_fragment(self) -> str:  # noqa: C901
        has_mcdc = self.total_mcdc_conditions > 0

        # --- No tests yet ---
        if not self.tests:
            if has_mcdc:
                return (
                    f"No tests executed yet. Target is {self.total_mcdc_conditions} MC/DC pairs. "
                    "Do NOT stop. Generate and execute a first compilable test case."
                )
            return (
                "No tests executed yet. Execute a first compilable test case to assess "
                "statement and branch coverage."
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

        # --- Check what is still missing ---
        mcdc_done = not has_mcdc or len(self.covered_keys) >= self.total_mcdc_conditions
        cum_stmts = self.cumulative_unvisited_statements
        cum_branches = self.cumulative_unvisited_branches
        stmt_done = not cum_stmts
        branch_done = not cum_branches

        if mcdc_done and stmt_done and branch_done:
            return (
                "SUCCESS! All coverage targets are now met (statement, branch"
                + (", MC/DC" if has_mcdc else "")
                + "). DO NOT call any more tools. "
                "Please output your final 'DONE: ...' message immediately to finish the task."
            )

        sections: list[str] = []

        # --- MC/DC section (only when function has conditions) ---
        if has_mcdc and not mcdc_done:
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

            observed = self._observed_polarity_counts()

            obligations: list[tuple[dict, bool]] = []
            for item in unvisited:
                if item["needs_true"]:
                    obligations.append((item, True))
                if item["needs_false"]:
                    obligations.append((item, False))

            def _is_suspected_stuck(item: dict, polarity: bool) -> bool:
                if self.iteration_count < 6:
                    return False
                target_seen = observed.get(ConditionKey(item["condition_id"], polarity), 0)
                opposite_seen = observed.get(
                    ConditionKey(item["condition_id"], not polarity), 0
                )
                return target_seen == 0 and opposite_seen >= 3

            obligations.sort(
                key=lambda pair: (
                    _is_suspected_stuck(pair[0], pair[1]),
                    pair[0]["line_in_function"] is None,
                    pair[0]["line_in_function"]
                    if pair[0]["line_in_function"] is not None
                    else float("inf"),
                    pair[0]["condition_id"],
                    0 if pair[1] else 1,
                )
            )

            suspected = [
                (item, pol) for item, pol in obligations if _is_suspected_stuck(item, pol)
            ]
            non_stuck = [
                (item, pol)
                for item, pol in obligations
                if not _is_suspected_stuck(item, pol)
            ]

            if not non_stuck and suspected:
                mcdc_lines = [
                    "Condition identity note: MC/DC condition identity is nodeId; condition text may repeat across different nodeIds.",
                    "",
                    "All remaining MC/DC obligations are likely stuck/unobservable from backend feedback.",
                    "Do NOT continue issuing near-duplicate tests for these.",
                    "Report likely instrumentation gap for the remaining nodeId/polarities:",
                ]
                for item, pol in suspected[:8]:
                    line_tag = (
                        f"line+{item['line_in_function']}"
                        if item["line_in_function"] is not None
                        else "line+?"
                    )
                    need = "TRUE" if pol else "FALSE"
                    mcdc_lines.append(
                        f"  • [node:{item['condition_id']} {line_tag}] need {need} for {item['condition']!r}"
                    )
                mcdc_lines.append(
                    "Rationale: opposite polarity was observed repeatedly, but this polarity never appears."
                )
                sections.append("\n".join(mcdc_lines))
            else:
                mcdc_lines = [
                    "Condition identity note: MC/DC condition identity is nodeId; condition text may repeat across different nodeIds.",
                    "",
                    "The following MC/DC condition polarities are NOT yet covered:",
                ]
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
                    mcdc_lines.append(
                        f"  • [node:{item['condition_id']} {line_tag}] {item['condition']!r} — missing: {', '.join(missing)}"
                    )

                if suspected:
                    mcdc_lines.append(
                        "\nPotentially stuck obligations (opposite polarity seen repeatedly):"
                    )
                    for item, pol in suspected[:5]:
                        line_tag = (
                            f"line+{item['line_in_function']}"
                            if item["line_in_function"] is not None
                            else "line+?"
                        )
                        need = "TRUE" if pol else "FALSE"
                        mcdc_lines.append(
                            f"  • [node:{item['condition_id']} {line_tag}] need {need} for {item['condition']!r}"
                        )
                    mcdc_lines.append(
                        "  Do not fixate on these first; cover other obligations, then retry with a different baseline/path."
                    )

                if self.consecutive_redundant >= 3:
                    mcdc_lines.append(
                        "\nWARNING: 3+ consecutive redundant tests (0 new MC/DC pairs). "
                        "You are likely stuck at a local optimum. "
                        "Do NOT keep issuing near-duplicate tests. "
                        "Switch to a different nodeId family/branch structure. "
                        "If the next attempt is still redundant, output final DONE summary immediately to stop token waste."
                    )
                elif self.consecutive_redundant == 2:
                    mcdc_lines.append(
                        "\nCAUTION: 2 consecutive redundant tests. Switch to a different baseline passing test and "
                        "drive a different code path for the targeted nodeId/polarity."
                    )

                mcdc_lines.append(
                    "\nTarget the test path that satisfies the highest number of conditions. "
                    "Prioritize paths that cover multiple uncovered conditions simultaneously. "
                    "If multiple paths are possible, choose the one that increases overall condition coverage the most. "
                    "Prefer modifying an existing passing test when possible, but allow generating a new test if needed."
                )
                sections.append("\n".join(mcdc_lines))
        elif has_mcdc and mcdc_done:
            sections.append(f"MC/DC: 100% covered ({self.total_mcdc_conditions}/{self.total_mcdc_conditions} pairs).")

        # --- Statement section ---
        if not stmt_done:
            stmt_pct = f"{self.statement_coverage_pct * 100:.0f}"
            stmt_lines = [
                f"Statement coverage: {stmt_pct}% — "
                f"{len(cum_stmts)} statement(s) not yet executed by any test:"
            ]
            for s in sorted(cum_stmts, key=lambda x: (x.line_in_function is None, x.line_in_function))[:15]:
                line_tag = f"line+{s.line_in_function}" if s.line_in_function is not None else "line+?"
                stmt_lines.append(f"  • [{line_tag}] {s.statement!r}")
            if len(cum_stmts) > 15:
                stmt_lines.append(f"  ... and {len(cum_stmts) - 15} more")
            sections.append("\n".join(stmt_lines))

        # --- Branch section ---
        if not branch_done:
            branch_pct = f"{self.branch_coverage_pct * 100:.0f}"
            branch_lines = [
                f"Branch coverage: {branch_pct}% — "
                f"{len(cum_branches)} branch node(s) with uncovered side(s) across all tests:"
            ]
            for b in sorted(cum_branches, key=lambda x: (x.line_in_function is None, x.line_in_function))[:15]:
                line_tag = f"line+{b.line_in_function}" if b.line_in_function is not None else "line+?"
                missing_sides = []
                if not b.true_visited:
                    missing_sides.append("TRUE")
                if not b.false_visited:
                    missing_sides.append("FALSE")
                branch_lines.append(
                    f"  • [{line_tag}] {b.condition!r} — missing: {', '.join(missing_sides)}"
                )
            if len(cum_branches) > 15:
                branch_lines.append(f"  ... and {len(cum_branches) - 15} more")
            sections.append("\n".join(branch_lines))

        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        return {
            "function_path": self.function_path,
            "iteration_count": self.iteration_count,
            "statement_coverage_pct": round(self.statement_coverage_pct, 4),
            "branch_coverage_pct": round(self.branch_coverage_pct, 4),
            "mcdc_coverage_pct": round(self.mcdc_coverage_pct, 4),
            "total_statements": self.total_statements,
            "total_branches": self.total_branches,
            "covered_mcdc_pairs": len(self.covered_keys),
            "total_mcdc_pairs": self.total_mcdc_conditions,
            "redundancy_rate": round(self.redundancy_rate, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "tests": [t.model_dump() for t in self.tests],
        }
