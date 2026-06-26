from __future__ import annotations

from dataclasses import dataclass


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
        context_text=ctx.context,
        source_text=f"// Source: {function_path}\n{_number_lines(src.source)}",
    )
