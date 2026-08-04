from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from covxplore.types.expected_path_step import ExpectedPathStep
from covxplore.types.test_result import TestResult
from covxplore.types.trace_summary import RuntimeValue, TargetFunctionConditionStep

_MAX_RUNTIME_OPERANDS = 4
_MAX_RENDERED_VALUE_CHARS = 80


@dataclass
class PathComparison:
    """Ordered comparison of a predicted path against one test's runtime trace."""

    matched: bool | None
    divergence_index: int | None = None
    divergence_step: ExpectedPathStep | None = None
    observed_step: TargetFunctionConditionStep | None = None


def path_signature(path: Sequence[ExpectedPathStep]) -> tuple[tuple[int, str], ...]:
    """Stable identity for an expected path (used for within-batch dedupe)."""
    return tuple((step.node_id, step.polarity) for step in path)


def _as_path_signature(
    path: Sequence[tuple[int, str]] | Sequence[ExpectedPathStep],
) -> tuple[tuple[int, str], ...]:
    if not path:
        return ()
    first = path[0]
    if isinstance(first, ExpectedPathStep):
        return path_signature(path)  # type: ignore[arg-type]
    return tuple(path)  # type: ignore[arg-type]


def is_proper_path_prefix(
    shorter: Sequence[tuple[int, str]] | Sequence[ExpectedPathStep],
    longer: Sequence[tuple[int, str]] | Sequence[ExpectedPathStep],
) -> bool:
    short_sig = _as_path_signature(shorter)
    long_sig = _as_path_signature(longer)
    return bool(short_sig) and len(short_sig) < len(long_sig) and long_sig[: len(short_sig)] == short_sig


def polarity_to_bool(polarity: str) -> bool:
    return polarity == "TRUE"


def effective_expected_path(result: TestResult) -> list[ExpectedPathStep]:
    if result.expected_path:
        return list(result.expected_path)
    if result.target_node_id is not None and result.target_polarity is not None:
        return [ExpectedPathStep(node_id=result.target_node_id, polarity=result.target_polarity)]
    return []


def expected_path_from_action(
    *,
    expected_path: Sequence[ExpectedPathStep] | None = None,
    target_node_id: int | None = None,
    target_polarity: str | None = None,
) -> list[ExpectedPathStep]:
    if expected_path:
        return list(expected_path)
    if target_node_id is not None and target_polarity is not None:
        return [ExpectedPathStep(node_id=target_node_id, polarity=target_polarity)]
    return []


def ordered_runtime_path(result: TestResult) -> list[TargetFunctionConditionStep]:
    """Return evaluated target-function conditions in execution order."""
    if result.trace_summary is None:
        return []
    return [
        step
        for step in result.trace_summary.target_function_condition_steps
        if step.node_id is not None and _step_polarity(step) is not None
    ]


def runtime_path_contains(result: TestResult, step: ExpectedPathStep) -> bool:
    return any(
        observed.node_id == step.node_id and _step_polarity(observed) == step.polarity
        for observed in ordered_runtime_path(result)
    )


def compare_expected_path(
    result: TestResult,
    *,
    known_node_ids: set[int] | None = None,
) -> PathComparison:
    """Compare expected_path with the ordered runtime trace, with no coverage fallback."""
    expected = effective_expected_path(result)
    if not expected:
        return PathComparison(matched=None)

    actual = ordered_runtime_path(result)
    for index, predicted in enumerate(expected):
        if known_node_ids is not None and predicted.node_id not in known_node_ids:
            return PathComparison(False, index, predicted)
        if index >= len(actual):
            return PathComparison(False, index, predicted)
        observed = actual[index]
        if observed.node_id != predicted.node_id or _step_polarity(observed) != predicted.polarity:
            return PathComparison(False, index, predicted, observed)
    return PathComparison(matched=True)


def format_divergence(
    result: TestResult,
    *,
    known_node_ids: set[int] | None = None,
) -> str | None:
    comparison = compare_expected_path(result, known_node_ids=known_node_ids)
    if comparison.matched is not False:
        return None

    path = effective_expected_path(result)
    step = comparison.divergence_step
    observed = comparison.observed_step
    assert step is not None and comparison.divergence_index is not None

    prior = " → ".join(f"node{s.node_id}={s.polarity}" for s in path[: comparison.divergence_index])
    prefix = f"after {prior}, " if prior else ""

    if known_node_ids is not None and step.node_id not in known_node_ids:
        observed_desc = (
            "nodeId is not in the BRANCH NODE CATALOG (do not use source line numbers as nodeId; "
            "copy [nodeId=N] from the catalog/gap)"
        )
    elif observed is None:
        observed_desc = "the ordered runtime trace ended before this condition was evaluated"
    else:
        observed_desc = (
            f"runtime evaluated node {observed.node_id}={_step_polarity(observed)} instead — "
            "adjust the predicted order or the input controlling this decision"
        )

    runtime_desc = _format_runtime_operands(result, step, observed)
    runtime_sentence = f" runtime operands: {runtime_desc}." if runtime_desc else ""
    return (
        f"Predicted path diverged at step {comparison.divergence_index + 1}/{len(path)} "
        f"(node {step.node_id}, predicted {step.polarity}): {prefix}{observed_desc}."
        f"{runtime_sentence}"
    )


def _step_polarity(step: TargetFunctionConditionStep) -> str | None:
    value = step.branch if step.branch is not None else step.condition_value
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"TRUE", "FALSE"}:
            return normalized
    return None


def _format_runtime_operands(
    result: TestResult,
    step: ExpectedPathStep,
    observed: TargetFunctionConditionStep | None,
) -> str | None:
    steps = _matching_condition_steps(result, step, observed)
    values = _latest_runtime_values(steps)
    if not values:
        return None
    rendered = [_format_runtime_value(value) for value in values[:_MAX_RUNTIME_OPERANDS]]
    return ", ".join(rendered)


def _matching_condition_steps(
    result: TestResult,
    step: ExpectedPathStep,
    observed: TargetFunctionConditionStep | None,
) -> list[TargetFunctionConditionStep]:
    if not result.trace_summary:
        return []
    condition_steps = result.trace_summary.target_function_condition_steps
    by_node = [candidate for candidate in condition_steps if candidate.node_id == step.node_id]
    if _latest_runtime_values(by_node):
        return by_node
    if observed is not None and observed.runtime_values:
        return [observed]
    return by_node


def _latest_runtime_values(steps: list[TargetFunctionConditionStep]) -> list[RuntimeValue]:
    values_by_expr: dict[str, RuntimeValue] = {}
    for condition_step in steps[-2:]:
        for value in condition_step.runtime_values:
            values_by_expr.pop(value.expression, None)
            values_by_expr[value.expression] = value
    return list(values_by_expr.values())


def _format_runtime_value(value: RuntimeValue) -> str:
    rendered_value = _truncate(str(value.value))
    type_suffix = f" ({value.type})" if value.type else ""
    return f"{value.expression}={rendered_value}{type_suffix}"


def _truncate(value: str) -> str:
    if len(value) <= _MAX_RENDERED_VALUE_CHARS:
        return value
    return value[: _MAX_RENDERED_VALUE_CHARS - 1] + "…"
