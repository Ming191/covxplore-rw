"""SearchNodesTool — wraps POST /api/search."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError
from covxplore.tools.execute_testcase import RunContext


class _Input(BaseModel):
    query: str = Field(
        ...,
        description="Substring / name to search for in the AST.",
    )
    types: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of node types to filter by. "
            "Valid values: FUNCTION, CLASS, STRUCT, ENUM, ENUM_VALUE, "
            "TYPEDEF, NAMESPACE, MACRO, VARIABLE, GLOBAL_VAR."
        ),
    )


class SearchNodesTool(BaseTool):
    """Search the AkaUT AST for nodes by name and/or type.

    Use this when the function context mentions a symbol whose absolutePath
    you do not yet know, or when you want to find related functions /
    types that might affect the function under test.
    Returns up to 20 matches with name, qualifiedName, absolutePath, type,
    and source line number.
    """

    name: str = "search_nodes"
    description: str = (
        "Search the loaded C/C++ project AST for nodes matching a name query. "
        "Optionally filter by node type (FUNCTION, STRUCT, ENUM, etc.). "
        "Pass only source-backed absolutePath values (FUNCTION, STRUCT, CLASS, ENUM, "
        "TYPEDEF, MACRO) to get_node_source. Do not call get_node_source for "
        "VARIABLE/GLOBAL_VAR member-field paths like .../Struct/field; variables have "
        "no function body. If a search returns no nodes, do not retry query variants."
    )
    args_schema: type[BaseModel] = _Input
    _run_context: RunContext | None = None

    def __init__(self, run_context: RunContext | None = None, **data):
        super().__init__(**data)
        object.__setattr__(self, "_run_context", run_context)

    def _run(self, query: str, types: list[str] | None = None) -> str:
        if self._run_context:
            blocked = self._run_context.allow_discovery("search")
            if blocked:
                return blocked
        key = f"search:{query}:{','.join(types or [])}"
        if self._run_context and key in self._run_context.dynamic_knowledge:
            return self._run_context.dynamic_knowledge[key]
        try:
            with AkaUTClient() as client:
                nodes = client.search_nodes(query, types)
        except AkaUTError as exc:
            return f"[ERROR] search_nodes failed: {exc}"

        if not nodes:
            result = f"No nodes found matching query={query!r} types={types}."
        else:
            lines = [f"Found {len(nodes)} node(s) for query={query!r}:"]
            for n in nodes[:20]:
                lines.append(
                    f"  [{n.type}] {n.qualified_name}  line={n.line}\n"
                    f"    absolutePath={n.absolute_path!r}"
                )
            result = "\n".join(lines)
        if self._run_context:
            self._run_context.remember_knowledge(key, result)
        return result
