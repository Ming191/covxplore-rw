from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

from covxplore.agents.schemas import GenerateTestAction, PathGuidedTestAction
from covxplore.status import TestStatus, is_failure_status, normalize_test_status
from covxplore.types import (
    PathPrediction,
    RuntimeValue,
    TargetFunctionConditionStep,
    TestResult,
    TraceSummary,
)
from covxplore.types.path_prediction import format_compact_expected_path

PathKey = tuple[int, str]


@dataclass(frozen=True)
class PathRejection:
    candidate: PathGuidedTestAction
    diagnostic: str


@dataclass(frozen=True)
class PathPruneResult:
    kept: list[GenerateTestAction]
    rejected: list[PathRejection]


class PathGuidance:
    @staticmethod
    def prune(candidates: list[GenerateTestAction]) -> PathPruneResult:
        path_candidates = [
            candidate for candidate in candidates if isinstance(candidate, PathGuidedTestAction)
        ]
        if not path_candidates:
            return PathPruneResult(list(candidates), [])

        all_paths = [PathGuidance._path(candidate) for candidate in path_candidates]
        seen: set[tuple[PathKey, ...]] = set()
        kept: list[GenerateTestAction] = []
        rejected: list[PathRejection] = []
        for candidate in candidates:
            if not isinstance(candidate, PathGuidedTestAction):
                kept.append(candidate)
                continue
            path = PathGuidance._path(candidate)
            if path in seen:
                rejected.append(
                    PathRejection(candidate, f"DUPLICATE_EXPECTED_PATH: {candidate.path_id}")
                )
            elif any(
                len(path) < len(other) and other[: len(path)] == path
                for other in all_paths
            ):
                rejected.append(
                    PathRejection(candidate, f"EXPECTED_PATH_PREFIX: {candidate.path_id}")
                )
            else:
                kept.append(candidate)
            seen.add(path)
        return PathPruneResult(kept, rejected)

    @staticmethod
    def pending(candidate: PathGuidedTestAction) -> PathPrediction:
        return PathPrediction.pending(candidate.path_id, candidate.expected_path)

    @staticmethod
    def score(candidate: PathGuidedTestAction, trace: TraceSummary | None) -> PathPrediction:
        expected = PathGuidance._path(candidate)
        actual = [
            outcome
            for step in (trace.target_function_condition_steps if trace else [])
            if (outcome := PathGuidance._observed_outcome(step)) is not None
        ]
        remaining = Counter(actual)
        supported: list[bool] = []
        for hypothesis in expected:
            found = remaining[hypothesis] > 0
            supported.append(found)
            if found:
                remaining[hypothesis] -= 1

        count = sum(supported)
        prediction = PathGuidance.pending(candidate)
        prediction.supported_hypotheses = count
        prediction.branch_support_pct = round(count / len(expected) * 100, 4)
        prediction.full_path_match = PathGuidance._is_ordered_subsequence(expected, actual)
        prediction.first_unsupported_index = next(
            (index for index, found in enumerate(supported) if not found),
            None,
        )
        return prediction

    @staticmethod
    def mismatch_feedback(results: list[TestResult], limit: int = 3) -> str:
        """Path-outcome feedback only for executed tests with a usable trace."""
        lines = []
        for result in results:
            if normalize_test_status(result.status) != TestStatus.PASSED:
                continue
            prediction = result.path_prediction
            index = prediction.first_unsupported_index if prediction else None
            if prediction is None or prediction.full_path_match is not False or index is None:
                continue
            expected = prediction.expected_path[index]
            observed, values = PathGuidance._diagnostic_at(
                result.trace_summary, expected.node_id
            )
            line = (
                f"- {result.test_name} [{prediction.path_id}]: "
                f"support={prediction.branch_support_pct}%, full_match=false; "
                f"first_unsupported[{index}]: nodeId={expected.node_id} "
                f"expected={expected.outcome}, observed={observed or 'NOT_OBSERVED'}; "
                f"keep path_id/expected_path and repair inputs so this node outcome matches."
            )
            if values:
                line += "; runtime: " + ", ".join(
                    f"{value.expression}={PathGuidance._compact_value(value)}"
                    for value in values[:5]
                )
            lines.append(line)
            if len(lines) >= limit:
                break
        return "PATH FEEDBACK:\n" + "\n".join(lines) if lines else ""

    @staticmethod
    def harness_repair_feedback(results: list[TestResult], limit: int = 5) -> str:
        """Tell the next batch to repair failed bodies before inventing new paths."""
        lines = []
        for result in results:
            status = normalize_test_status(result.status)
            if not is_failure_status(status):
                continue
            prediction = result.path_prediction
            path_hint = " keep the same path_id and expected_path;"
            if prediction is not None and prediction.path_id and prediction.expected_path:
                path_hint = (
                    f" keep path_id={prediction.path_id} and expected_path="
                    f"{format_compact_expected_path(prediction.expected_path)};"
                )
            lines.append(
                f"- {result.test_name} ({status.value}):{path_hint} "
                "fix test_body using Failed execution logs "
                "(qualify namespaces, match harness contract); "
                "do not replace with a new path until this compiles and runs."
            )
            if len(lines) >= limit:
                break
        return "HARNESS REPAIR:\n" + "\n".join(lines) if lines else ""

    @staticmethod
    def _diagnostic_at(
        trace: TraceSummary | None, node_id: int
    ) -> tuple[str | None, list[RuntimeValue]]:
        observed = None
        values: list[RuntimeValue] = []
        for step in trace.target_function_condition_steps if trace else []:
            if step.node_id != node_id:
                continue
            outcome = PathGuidance._observed_outcome(step)
            if outcome is not None:
                observed = outcome[1]
            if step.runtime_values:
                values = step.runtime_values
        return observed, values

    @staticmethod
    def _compact_value(value: RuntimeValue) -> str:
        rendered = json.dumps(value.value, ensure_ascii=True)
        return rendered if len(rendered) <= 80 else rendered[:77] + "..."

    @staticmethod
    def _path(candidate: PathGuidedTestAction) -> tuple[PathKey, ...]:
        return tuple(step.key for step in candidate.expected_path)

    @staticmethod
    def _observed_outcome(step: TargetFunctionConditionStep) -> PathKey | None:
        if step.node_id is None:
            return None
        if step.branch is not None:
            return step.node_id, str(step.branch).upper()
        if step.condition_value is not None:
            return step.node_id, "TRUE" if step.condition_value else "FALSE"
        return None

    @staticmethod
    def _is_ordered_subsequence(expected: tuple[PathKey, ...], actual: list[PathKey]) -> bool:
        cursor = iter(actual)
        return all(any(observed == hypothesis for observed in cursor) for hypothesis in expected)
