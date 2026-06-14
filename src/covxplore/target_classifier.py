from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from covxplore.test_suite import TestSuite


@dataclass(frozen=True)
class TargetHint:
    node_id: int | None
    kind: str
    condition: str
    missing: str
    line_in_function: int | None
    strategy: str


def classify_condition(condition: str) -> str:
    text = condition.strip()
    lowered = text.lower()
    if any(token in lowered for token in ("null", "nullptr")) or "->" in text:
        return "pointer_null"
    if "[" in text or any(token in lowered for token in ("strlen", "size", "length")):
        return "array_string"
    if "&&" in text or "||" in text or "!" in text:
        return "boolean_logic"
    if re.search(r"(<=|>=|==|!=|<|>)\s*-?\d", text) or re.search(
        r"-?\d\s*(<=|>=|==|!=|<|>)", text
    ):
        return "numeric_boundary"
    if "." in text:
        return "struct_field"
    return "hard_unknown"


def strategy_for(category: str, missing: str) -> str:
    if category == "numeric_boundary":
        return f"Use boundary values around the comparison to force {missing}."
    if category == "boolean_logic":
        return f"Choose boolean sub-condition values deliberately; vary one operand at a time for {missing}."
    if category == "pointer_null":
        return f"Try null vs allocated/non-null setup and pointee values to force {missing}."
    if category == "array_string":
        return f"Vary length, index, empty/non-empty content, and boundary elements to force {missing}."
    if category == "struct_field":
        return f"Initialize the relevant field values directly and minimally to force {missing}."
    return f"Derive a distinct input path from the source/context to force {missing}."


def build_target_hints(suite: TestSuite, limit: int = 12) -> str:
    hints: list[TargetHint] = []

    for item in suite.unvisited_summary():
        missing_parts = []
        if item["needs_true"]:
            missing_parts.append("TRUE")
        if item["needs_false"]:
            missing_parts.append("FALSE")
        missing = "/".join(missing_parts)
        category = classify_condition(item["condition"])
        hints.append(
            TargetHint(
                node_id=item["condition_id"],
                kind=category,
                condition=item["condition"],
                missing=missing,
                line_in_function=item["line_in_function"],
                strategy=strategy_for(category, missing),
            )
        )

    for branch in suite.cumulative_unvisited_branches:
        missing_parts = []
        if not branch.true_visited:
            missing_parts.append("TRUE")
        if not branch.false_visited:
            missing_parts.append("FALSE")
        missing = "/".join(missing_parts)
        category = classify_condition(branch.condition)
        hints.append(
            TargetHint(
                node_id=branch.node_id,
                kind=f"branch_{category}",
                condition=branch.condition,
                missing=missing,
                line_in_function=branch.line_in_function,
                strategy=strategy_for(category, missing),
            )
        )

    unique = _dedupe_hints(hints)
    if not unique:
        return ""

    lines = ["TARGET STRATEGY HINTS:"]
    for hint in unique[:limit]:
        node = hint.node_id if hint.node_id is not None else "?"
        line = hint.line_in_function if hint.line_in_function is not None else "?"
        lines.append(
            f"  • [node:{node} line+{line}] {hint.kind}, missing {hint.missing}: "
            f"{hint.condition!r}. {hint.strategy}"
        )
    if len(unique) > limit:
        lines.append(f"  ... and {len(unique) - limit} more target hints")
    return "\n".join(lines)


def _dedupe_hints(hints: Iterable[TargetHint]) -> list[TargetHint]:
    seen: set[tuple[int | None, str, str, str]] = set()
    result: list[TargetHint] = []
    for hint in hints:
        key = (hint.node_id, hint.kind, hint.condition, hint.missing)
        if key in seen:
            continue
        seen.add(key)
        result.append(hint)
    return result
