from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class StaticPromptData:
    context_text: str
    source_text: str
    total_statements: int | None = None
    total_branches: int | None = None


def format_context_v2_for_prompt(data: dict[str, Any]) -> str:
    lines = ["STATIC PROGRAM FACTS (context-v2):"]
    focal = data.get("focal") or {}
    if focal:
        lines.append(f"Focal: {focal.get('signature') or focal.get('name')}")

    inputs = data.get("inputs") or []
    if inputs:
        lines.append("Inputs:")
        for item in inputs:
            lines.append(f"- {item.get('type')} {item.get('name')}")
            structure = item.get("structure") or {}
            if structure:
                lines.append(f"  construct: {structure.get('name')}()")
                fields = structure.get("publicFields") or []
                if fields:
                    lines.append("  public fields:")
                    for field in fields:
                        lines.append(f"  - {field.get('type')} {field.get('name')}")

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
                mapping = f"  arg {arg.get('focalParam')} <- {arg.get('callerExpr')}"
                if arg.get("baseVariable"):
                    mapping += f" base={arg.get('baseVariable')}"
                lines.append(mapping)
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
            source = helper.get("source")
            if source:
                lines.append("  source:")
                lines.extend(f"    {line}" for line in source.splitlines())
            elif helper.get("sourceMode"):
                lines.append(f"  source: [{helper.get('sourceMode')}]")

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


class StaticContextProvider(ABC):
    @abstractmethod
    def context_text(self, client, function_path: str) -> str:
        """Return prompt-ready static context for one function."""


class LegacyContextProvider(StaticContextProvider):
    def context_text(self, client, function_path: str) -> str:
        return _cap_context(client.get_function_context(function_path).context)


class StructuredContextProvider(StaticContextProvider):
    def context_text(self, client, function_path: str) -> str:
        return format_context_v2_for_prompt(client.get_function_context_v2(function_path).raw)


_CONTEXT_PROVIDERS: dict[str, StaticContextProvider] = {
    "v1": LegacyContextProvider(),
    "v2": StructuredContextProvider(),
}


def get_context_provider(version: str) -> StaticContextProvider:
    try:
        return _CONTEXT_PROVIDERS[version]
    except KeyError as exc:
        raise ValueError("context version must be 'v1' or 'v2'") from exc


def _number_lines(source: str) -> str:
    lines = source.splitlines()
    width = len(str(max(len(lines) - 1, 0)))
    return "\n".join(f"{index:{width}}| {line}" for index, line in enumerate(lines))


def fetch_static_prompt_data(
    function_path: str,
    context_version: str = "v1",
) -> StaticPromptData:
    """Fetch fixed context, source, and structural coverage totals once."""
    from covxplore.api_client import AkaUTClient

    provider = get_context_provider(context_version)
    with AkaUTClient() as client:
        context_text = provider.context_text(client, function_path)
        src = client.get_node_source(function_path)
        try:
            coverage = client.get_node_conditions(function_path, coverage_type="BRANCH")
            total_statements = coverage.total_statements
            total_branches = coverage.total_branches
        except Exception:
            total_statements = None
            total_branches = None

    return StaticPromptData(
        context_text=context_text,
        source_text=f"// Source: {function_path}\n{_number_lines(src.source)}",
        total_statements=total_statements,
        total_branches=total_branches,
    )
