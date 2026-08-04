from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from covxplore.coverage.path_comparison import (
    expected_path_from_action,
    is_proper_path_prefix,
    path_signature,
    polarity_to_bool,
)
from covxplore.driver.executor import TestCaseExecutor
from covxplore.status import TestStatus, is_failure_status, normalize_test_status
from covxplore.types.expected_path_step import ExpectedPathStep
from covxplore.types.test_result import TestResult
from covxplore.types.test_suite import TestSuite


@dataclass
class TestcaseCandidate:
    test_body: str
    test_name: str | None = None
    target_node_id: int | None = None
    target_polarity: str | None = None
    target_reason: str | None = None
    expected_path: list[ExpectedPathStep] = field(default_factory=list)


class _HasTargetPath(Protocol):
    test_name: str | None
    target_node_id: int | None
    target_polarity: str | None
    expected_path: Sequence[ExpectedPathStep]


@dataclass
class PreflightFilterResult:
    """Candidates kept for execution plus human-readable skip notes."""

    executable: list
    notes: list[str] = field(default_factory=list)


@dataclass
class BatchMergeSummary:
    accepted: list[TestResult] = field(default_factory=list)
    rejected_redundant: list[TestResult] = field(default_factory=list)
    # Candidates skipped before execution (duplicate/already-covered target within this
    # batch), so no AkaUT compile/run cycle was spent on them. Populated by the caller.
    preflight_notes: list[str] = field(default_factory=list)

    def record_redundancy(self, suite: TestSuite) -> None:
        if not self.rejected_redundant:
            return
        if any(_is_coverage_result(result) and not result.is_redundant for result in self.accepted):
            return
        if any(is_failure_status(result.status) for result in self.accepted):
            return
        suite.coverage.consecutive_redundant += 1


def _marginal_gain(suite: TestSuite, result: TestResult) -> int:
    return suite.coverage.structural_gain(result)


def _is_coverage_result(result: TestResult) -> bool:
    return normalize_test_status(result.status) in {TestStatus.PASSED, TestStatus.RUNTIME_ERROR}


def filter_preflight_candidates(
    suite: TestSuite,
    candidates: Sequence[_HasTargetPath],
) -> PreflightFilterResult:
    """Drop candidates whose predicted path is already covered, duplicated, or dominated.

    Dominance: if path A is a proper prefix of path B in the same batch (A ⊂ B as an ordered
    trace), keep B and skip A — executing B already forces every step of A.

    Exploratory candidates (no ``expected_path`` and no target metadata) are always kept.
    When the suite has not yet established branch-key coverage, "already covered" checks are
    skipped (``is_branch_satisfied`` returns ``None``) so we never silently drop useful tests.
    """
    exploratory: list = []
    path_bearing: list[tuple[object, list[ExpectedPathStep], tuple[tuple[int, str], ...]]] = []
    notes: list[str] = []
    seen_signatures: set[tuple[tuple[int, str], ...]] = set()

    for candidate in candidates:
        label = candidate.test_name or "(unnamed)"
        path = expected_path_from_action(
            expected_path=candidate.expected_path,
            target_node_id=candidate.target_node_id,
            target_polarity=candidate.target_polarity,
        )
        if not path:
            exploratory.append(candidate)
            continue

        signature = path_signature(path)
        if signature in seen_signatures:
            notes.append(
                f"{label}: skipped preflight — duplicate expected_path "
                f"{_format_path_short(path)} already in this batch"
            )
            continue

        target = path[-1]
        satisfied = suite.coverage.is_branch_satisfied(
            target.node_id, polarity_to_bool(target.polarity)
        )
        if satisfied is True:
            # Branch already covered: still allow execution when statement gaps remain,
            # otherwise the agent can get stuck targeting a covered branch while a
            # statement (e.g. return false) is still open.
            stmt_gaps = suite.coverage._cumulative_uncovered_stmt_ids
            if stmt_gaps is None or not stmt_gaps:
                notes.append(
                    f"{label}: skipped preflight — target node {target.node_id} "
                    f"{target.polarity} already covered by the suite"
                )
                continue

        seen_signatures.add(signature)
        path_bearing.append((candidate, path, signature))

    kept_paths: list[tuple[object, list[ExpectedPathStep], tuple[tuple[int, str], ...]]] = []
    for candidate, path, signature in path_bearing:
        dominator = next(
            (
                other_path
                for _, other_path, other_sig in path_bearing
                if is_proper_path_prefix(signature, other_sig)
            ),
            None,
        )
        if dominator is not None:
            label = getattr(candidate, "test_name", None) or "(unnamed)"
            notes.append(
                f"{label}: skipped preflight — expected_path "
                f"{_format_path_short(path)} is a proper prefix of "
                f"{_format_path_short(dominator)} in this batch"
            )
            continue
        kept_paths.append((candidate, path, signature))

    # Preserve original candidate order: exploratory and kept path-bearing interleaved
    # by scanning the input list once more against the kept identity set.
    kept_ids = {id(candidate) for candidate, _, _ in kept_paths}
    kept_ids.update(id(candidate) for candidate in exploratory)
    executable = [candidate for candidate in candidates if id(candidate) in kept_ids]
    return PreflightFilterResult(executable=executable, notes=notes)


def _format_path_short(path: Sequence[ExpectedPathStep]) -> str:
    return " → ".join(f"node{step.node_id}={step.polarity}" for step in path)


def merge_batch_results(
    suite: TestSuite,
    results: list[TestResult],
    min_suite_size: int = 3,
    *,
    preflight_notes: list[str] | None = None,
) -> BatchMergeSummary:
    """Greedily add visible/progressing results from one generated batch."""
    indexed = list(enumerate(results))
    ranked = sorted(
        indexed,
        key=lambda item: (
            0 if _is_coverage_result(item[1]) else 1,
            -_marginal_gain(suite, item[1]),
            len(item[1].test_body),
            item[0],
        ),
    )

    summary = BatchMergeSummary(preflight_notes=list(preflight_notes or []))
    for _, result in ranked:
        gain = _marginal_gain(suite, result)
        if _is_coverage_result(result) and gain == 0 and suite.tests:
            result.is_redundant = True
            summary.rejected_redundant.append(result)
            suite.rejected_tests.append(result)
            continue

        suite.add_result(result, min_suite_size=min_suite_size)
        summary.accepted.append(result)

    return summary


async def async_execute_candidates(
    function_path: str,
    candidates: list[TestcaseCandidate],
    executor: TestCaseExecutor,
):
    results = []
    for candidate in candidates:
        results.append(
            await asyncio.to_thread(
                executor.execute,
                function_path,
                candidate.test_body,
                candidate.test_name,
            )
        )
    return results
