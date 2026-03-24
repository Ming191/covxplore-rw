"""GetNodeSourceTool — wraps GET /api/node/source."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError


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
        "Get the raw C/C++ source code of any node (function, struct, class, "
        "enum, macro) identified by its absolutePath. Use this when the context "
        "references a type or dependency whose definition you need to understand."
    )
    args_schema: type[BaseModel] = _Input

    def _run(self, absolute_path: str) -> str:
        try:
            with AkaUTClient() as client:
                result = client.get_node_source(absolute_path)
            lines = [f"// Source: {absolute_path}", result.source]
            if result.value is not None:
                lines.append(f"// Resolved value: {result.value}")
            return "\n".join(lines)
        except AkaUTError as exc:
            return f"[ERROR] get_node_source failed: {exc}"
