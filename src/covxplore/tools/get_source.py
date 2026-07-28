"""GetNodeSourceTool — wraps GET /api/node/source."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError
from covxplore.tools.execute_testcase import RunContext


class _Input(BaseModel):
    absolute_path: str = Field(
        ...,
        description=(
            "The absolute path of any AST node (function, struct, enum, etc.) "
            "as returned by search_nodes or from context output."
        ),
    )


class GetNodeSourceTool(BaseTool):
    """Retrieve the raw C/C++ source code of any AST node.

    Use when the function context mentions a type or helper function whose
    implementation you need to understand in order to construct valid test
    inputs or understand complex boolean conditions.
    """

    name: str = "get_node_source"
    description: str = (
        "Get the raw C/C++ source code of a source-backed AST node: FUNCTION, "
        "STRUCT, CLASS, ENUM, TYPEDEF, or MACRO. Do not use this for VARIABLE/"
        "GLOBAL_VAR member-field paths like .../Struct/field; variables have no body "
        "and return SOURCE_UNAVAILABLE. If this tool returns SOURCE_UNAVAILABLE or "
        "404/no source, do not retry the same symbol with path variants. For "
        "pointer-param fields (e.g. e->needsEscape), initialize fields directly in "
        "the test body with minimal values when enough to drive coverage."
    )
    args_schema: type[BaseModel] = _Input
    _run_context: RunContext | None = None

    def __init__(self, run_context: RunContext | None = None, **data):
        super().__init__(**data)
        object.__setattr__(self, "_run_context", run_context)

    def _run(self, absolute_path: str) -> str:
        if self._run_context:
            blocked = self._run_context.allow_discovery("source")
            if blocked:
                return blocked
        key = "source:" + absolute_path.replace("\\", "/")
        if self._run_context and key in self._run_context.dynamic_knowledge:
            return self._run_context.dynamic_knowledge[key]
        try:
            with AkaUTClient() as client:
                try:
                    result = client.get_node_source(absolute_path)
                except AkaUTError as exc:
                    if exc.status_code == 404:
                        output = _source_unavailable(absolute_path)
                    else:
                        raise
                else:
                    output = _format_source(absolute_path, result)
        except AkaUTError as exc:
            return f"[ERROR] get_node_source failed: {exc}"
        if self._run_context:
            self._run_context.remember_knowledge(key, output)
        return output


def _format_source(path: str, result, note: str | None = None) -> str:
    numbered = _number_lines(result.source)
    parts = []
    if note:
        parts.append(f"// Note: {note}")
    parts.extend([f"// Source: {path}", numbered])
    if result.value is not None:
        parts.append(f"// Resolved value: {result.value}")
    return "\n".join(parts)


def _source_unavailable(path: str) -> str:
    return (
        f"[SOURCE_UNAVAILABLE] No source body exists for {path!r}. "
        "If this is a VARIABLE/GLOBAL_VAR or member-field path, use the parent "
        "STRUCT/CLASS source from search_nodes and initialize the field directly "
        "in the test body. Do not retry this path or spelling variants."
    )


def _number_lines(source: str) -> str:
    """Prepend 0-based line numbers matching the lineInFunction values from get_conditions_static.

    lineInFunction is computed as (absolute file line - function start line), so
    lineInFunction=0 is the function signature line, lineInFunction=3 is 3 lines below.
    Numbering here starts at 0 so the LLM can look up a condition directly by its
    lineInFunction value without any arithmetic.

    Format: '  3| actual source line'
    """
    raw_lines = source.splitlines()
    width = len(str(len(raw_lines) - 1))
    return "\n".join(
        f"{i:{width}}| {line}" for i, line in enumerate(raw_lines, start=0)
    )
