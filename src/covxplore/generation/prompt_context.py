from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from covxplore.api_client import ConditionInfo


@dataclass
class StaticPromptData:
    context_text: str
    source_text: str
    branch_catalog_text: str = ""
    branch_catalog_ids: list[int] = field(default_factory=list)


def branch_catalog_ids(
    conditions: Sequence[ConditionInfo] | Sequence[dict[str, Any]],
) -> list[int]:
    """Extract BRANCH decision nodeIds (same space as expected_path / gaps)."""
    ids: list[int] = []
    seen: set[int] = set()
    for item in conditions:
        if isinstance(item, ConditionInfo):
            node_id = item.node_id
        else:
            node_id = item.get("node_id", item.get("nodeId"))
        if node_id is None:
            continue
        node_id = int(node_id)
        if node_id in seen:
            continue
        seen.add(node_id)
        ids.append(node_id)
    return ids


def format_branch_catalog(conditions: Sequence[ConditionInfo] | Sequence[dict[str, Any]]) -> str:
    """Render BRANCH CFG decisions for the agent — same nodeId space as coverage gaps."""
    if not conditions:
        return ""
    lines = [
        "BRANCH NODE CATALOG (use these nodeId values in expected_path / target_node_id):",
        "Do NOT invent nodeIds from source line numbers — line+N ≠ nodeId.",
        "expected_path may only reference BRANCH nodeIds below (never statement nodeIds).",
    ]
    for item in conditions:
        if isinstance(item, ConditionInfo):
            node_id = item.node_id
            condition = item.condition
            line = item.line_in_function
        else:
            node_id = item.get("node_id", item.get("nodeId"))
            condition = item.get("condition") or item.get("expression") or ""
            line = item.get("line_in_function", item.get("lineInFunction"))
        tag = f"nodeId={node_id}" if node_id is not None else "nodeId=?"
        line_hint = f" (source line+{line})" if line is not None else ""
        lines.append(f"- [{tag}]{line_hint}: {condition!r}")
    return "\n".join(lines)


def format_context_v2_for_prompt(data: dict[str, Any]) -> str:
    lines = ["STATIC PROGRAM FACTS (context-v2):"]
    focal = data.get("focal") or {}
    if focal:
        lines.append(f"Focal: {focal.get('signature') or focal.get('name')}")

    conditions = data.get("conditions") or []
    if conditions:
        lines.append("Conditions:")
        for item in conditions[:20]:
            reads = _refs_text(item.get("reads") or [])
            suffix = f" reads={reads}" if reads else ""
            lines.append(
                f"- node:{item.get('nodeId')} line+{item.get('lineInFunction')}: "
                f"{item.get('expression')}{suffix}"
            )

    accesses = data.get("fieldAccesses") or []
    if accesses:
        lines.append("Field accesses:")
        for item in accesses[:30]:
            lines.append(
                f"- {item.get('access')} {item.get('base')}->{item.get('field')} "
                f"line+{item.get('lineInFunction')}: {item.get('expression')}"
            )

    call_sites = data.get("callSites") or []
    if call_sites:
        lines.append("Call sites:")
        for site in call_sites[:3]:
            lines.append(f"- {site.get('callerSignature') or site.get('caller')}")
            for arg in site.get("argumentMap") or []:
                lines.append(
                    f"  arg {arg.get('focalParam')} <- {arg.get('callerExpr')} "
                    f"base={arg.get('baseVariable')}"
                )
            for stmt in site.get("statementsBeforeCall") or []:
                lines.append(f"  line {stmt.get('line')}: {stmt.get('text')}")

    helpers = data.get("helperEffects") or []
    if helpers:
        lines.append("Helper effects:")
        for helper in helpers[:12]:
            lines.append(f"- {helper.get('signature') or helper.get('function')}")
            reads = _refs_text(helper.get("reads") or [])
            writes = _refs_text(helper.get("writes") or [])
            calls = ", ".join(
                call.get("name", "") for call in (helper.get("calls") or []) if call.get("name")
            )
            if reads:
                lines.append(f"  reads: {reads}")
            if writes:
                lines.append(f"  writes: {writes}")
            if calls:
                lines.append(f"  calls: {calls}")

    return "\n".join(lines)


def _refs_text(refs: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{ref.get('base')}->{ref.get('field')}"
        for ref in refs
        if ref.get("base") and ref.get("field")
    )


def _cap_context(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [legacy /api/context truncated]"


def fetch_static_prompt_data(function_path: str) -> StaticPromptData:
    """Fetch static data once and return prompt-ready text blocks."""
    from covxplore.api_client import AkaUTClient
    from covxplore.tools.get_source import _number_lines

    with AkaUTClient() as client:
        ctx = client.get_function_context(function_path)
        src = client.get_node_source(function_path)
        try:
            branch_nodes = client.get_node_conditions(function_path, coverage_type="BRANCH")
            catalog = format_branch_catalog(branch_nodes.conditions)
            catalog_ids = branch_catalog_ids(branch_nodes.conditions)
        except Exception:
            catalog = ""
            catalog_ids = []

    return StaticPromptData(
        context_text=_cap_context(ctx.context),
        source_text=f"// Source: {function_path}\n{_number_lines(src.source)}",
        branch_catalog_text=catalog,
        branch_catalog_ids=catalog_ids,
    )
