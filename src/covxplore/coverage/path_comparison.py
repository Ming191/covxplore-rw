from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from covxplore.types.expected_path_step import ExpectedPathStep
from covxplore.types.test_result import TestResult
from covxplore.types.trace_summary import RuntimeValue, TargetFunctionConditionStep
from covxplore.types.unvisited_branch import UnvisitedBranch

_MAX_RUNTIME_OPERANDS = 4
_MAX_RENDERED_VALUE_CHARS = 80


@dataclass
class PathComparison:
    """Result of walking a predicted branch-outcome path against one test's actual run.

    ``matched`` is ``None`` when the candidate carried no target/expected_path at all (nothing
    to compare), ``True`` when every predicted step was corroborated, and ``False`` when the walk
    stopped at ``divergence_index`` because that step's predicted polarity was not observed.
    """

    matched: bool | None
    divergence_index: int | None = None
    divergence_step: ExpectedPathStep | None = None
    observed_branch: UnvisitedBranch | None = None


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
    """True when ``shorter`` is a proper prefix of ``longer`` (ordered path A ⊂ path B).

    Accepts either ``ExpectedPathStep`` sequences or already-computed ``path_signature`` tuples.
    """
    short_sig = _as_path_signature(shorter)
    long_sig = _as_path_signature(longer)
    return bool(short_sig) and len(short_sig) < len(long_sig) and long_sig[: len(short_sig)] == short_sig


def polarity_to_bool(polarity: str) -> bool:
    return polarity == "TRUE"


def effective_expected_path(result: TestResult) -> list[ExpectedPathStep]:
    """The path to compare against: ``expected_path`` if given, else the single legacy target."""
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
    """Build the effective path from an action/candidate before a TestResult exists."""
    if expected_path:
        return list(expected_path)
    if target_node_id is not None and target_polarity is not None:
        return [ExpectedPathStep(node_id=target_node_id, polarity=target_polarity)]
    return []


def compare_expected_path(
    result: TestResult,
    *,
    known_node_ids: set[int] | None = None,
) -> PathComparison:
    """Find the first predicted step not corroborated by this single test's own trace.

    ``result.unvisited_branches`` is scoped to this one execution, not cumulative suite
    coverage: a node_id's absence from it means BOTH polarities were observed by this run
    (fully covered on its own, e.g. a loop body); presence with the predicted side unset means
    that side was not observed this run — either the condition took the other branch, or
    control flow never reached it at all (both true_visited and false_visited are False).

    When ``known_node_ids`` is provided (BRANCH catalog), a step whose node_id is outside
    that set is treated as an immediate divergence — typically the model confused a source
    line number for a CFG nodeId.
    """
    path = effective_expected_path(result)
    if not path:
        return PathComparison(matched=None)

    observed_by_node = {
        branch.node_id: branch for branch in result.unvisited_branches if branch.node_id is not None
    }
    for index, step in enumerate(path):
        if known_node_ids is not None and step.node_id not in known_node_ids:
            return PathComparison(
                matched=False,
                divergence_index=index,
                divergence_step=step,
                observed_branch=None,
            )
        branch = observed_by_node.get(step.node_id)
        if branch is None:
            continue
        observed = branch.true_visited if step.polarity == "TRUE" else branch.false_visited
        if not observed:
            return PathComparison(
                matched=False,
                divergence_index=index,
                divergence_step=step,
                observed_branch=branch,
            )
    return PathComparison(matched=True)


def format_divergence(
    result: TestResult,
    *,
    known_node_ids: set[int] | None = None,
) -> str | None:
    """Human-readable divergence explanation for the agent, or ``None`` if nothing diverged."""
    comparison = compare_expected_path(result, known_node_ids=known_node_ids)
    if comparison.matched is not False:
        return None

    path = effective_expected_path(result)
    step = comparison.divergence_step
    branch = comparison.observed_branch
    assert step is not None and comparison.divergence_index is not None

    prior = " → ".join(f"node{s.node_id}={s.polarity}" for s in path[: comparison.divergence_index])
    prefix = f"after {prior}, " if prior else ""

    if known_node_ids is not None and step.node_id not in known_node_ids:
        observed_desc = (
            "nodeId is not in the BRANCH NODE CATALOG (do not use source line numbers as nodeId; "
            "copy [nodeId=N] from the catalog/gap)"
        )
    elif branch is not None and not branch.true_visited and not branch.false_visited:
        observed_desc = (
            "the condition was never evaluated in this run (control flow did not reach it) — "
            "shorten the path and change inputs so earlier decisions lead here"
        )
    elif branch is not None:
        observed_only = "TRUE" if branch.true_visited else "FALSE"
        observed_desc = (
            f"the condition only evaluated {observed_only} in this run, not {step.polarity} — "
            f"adjust the input that controls this decision and keep the same nodeId"
        )
    else:
        observed_desc = f"the predicted outcome ({step.polarity}) was not observed in this run"

    condition_text = f' ("{branch.condition}")' if branch and branch.condition else ""
    runtime_desc = _format_runtime_operands(result, step, branch)
    runtime_sentence = f" runtime operands: {runtime_desc}." if runtime_desc else ""
    return (
        f"Predicted path diverged at step {comparison.divergence_index + 1}/{len(path)} "
        f"(node {step.node_id}{condition_text}, predicted {step.polarity}): {prefix}{observed_desc}."
        f"{runtime_sentence}"
    )


def _format_runtime_operands(
    result: TestResult,
    step: ExpectedPathStep,
    branch: UnvisitedBranch | None,
) -> str | None:
    steps = _matching_condition_steps(result, step, branch)
    values = _latest_runtime_values(steps)
    if not values:
        return None
    rendered = [_format_runtime_value(value) for value in values[:_MAX_RUNTIME_OPERANDS]]
    return ", ".join(rendered)


def _matching_condition_steps(
    result: TestResult,
    step: ExpectedPathStep,
    branch: UnvisitedBranch | None,
) -> list[TargetFunctionConditionStep]:
    if not result.trace_summary:
        return []
    condition_steps = result.trace_summary.target_function_condition_steps
    by_node = [s for s in condition_steps if s.node_id == step.node_id]
    if by_node:
        return by_node
    if branch is None:
        return []
    by_offset = [
        s
        for s in condition_steps
        if s.node_id is None
        and s.start == branch.start_offset
        and s.end == branch.end_offset
        and (branch.line_in_function is None or s.line == branch.line_in_function)
    ]
    if by_offset:
        return by_offset
    return [
        s
        for s in condition_steps
        if s.node_id is None
        and branch.line_in_function is not None
        and s.line == branch.line_in_function
    ]


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
