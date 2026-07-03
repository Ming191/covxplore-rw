from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StaticPromptData:
    conditions_text: str
    context_text: str
    source_text: str


def seed_suite_conditions(suite, console=None) -> None:
    """Call /api/node/conditions before kickoff to pre-populate total_mcdc_pairs."""
    from covxplore.api_client import AkaUTClient, AkaUTError

    try:
        with AkaUTClient() as client:
            result = client.get_node_conditions(suite.function_path)
        if result.total_mcdc_pairs > 0:
            suite.coverage.seed_conditions(result.conditions, result.total_mcdc_pairs)
            if console is not None:
                console.print(
                    f"[dim]Static CFG: {result.total_conditions} conditions "
                    f"({result.total_mcdc_pairs} MC/DC pairs)[/]"
                )
    except AkaUTError as exc:
        if console is not None:
            console.print(f"[yellow]Could not prefetch conditions: {exc}[/]")


def format_conditions_for_prompt(result) -> str:
    lines = [
        f"Found {result.total_conditions} conditions "
        f"({result.total_mcdc_pairs} MC/DC pairs to cover):",
        "",
    ]
    for i, c in enumerate(result.conditions, start=1):
        if c.node_id is None:
            raise RuntimeError(
                "Backend payload missing nodeId in /api/node/conditions. "
                "nodeId is required for MC/DC identity."
            )
        line = c.line_in_function if c.line_in_function is not None else "?"
        start = c.start_offset if c.start_offset is not None else "?"
        end = c.end_offset if c.end_offset is not None else "?"
        lines.append(
            f"  {i}. [node:{c.node_id} line+{line}, offset {start}–{end}] {c.condition!r}"
        )
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
        cond = client.get_node_conditions(function_path)
        ctx = client.get_function_context(function_path)
        src = client.get_node_source(function_path)

    return StaticPromptData(
        conditions_text=format_conditions_for_prompt(cond),
        context_text=_cap_context(ctx.context),
        source_text=f"// Source: {function_path}\n{_number_lines(src.source)}",
    )
