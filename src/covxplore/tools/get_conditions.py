from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError
from covxplore.tools.execute_testcase import get_shared_suite


class _Input(BaseModel):
    absolute_path: str = Field(
        ...,
        description=(
            "The absolute path of the function node as returned by search_nodes, "
            "e.g. '/project/src/foo.cpp/MyNS::bar(int)'."
        ),
    )


class GetConditionsStaticTool(BaseTool):
    name: str = "get_conditions_static"
    description: str = (
        "Retrieve every MC/DC condition (boolean sub-expression) in a C/C++ function "
        "using static analysis of its control-flow graph. Returns condition text and "
        "source location for each. Call this once at the start of test generation to "
        "know the full coverage target before writing any tests."
    )
    args_schema: type[BaseModel] = _Input

    def _run(self, absolute_path: str) -> str:
        try:
            with AkaUTClient() as client:
                result = client.get_node_conditions(absolute_path)
        except AkaUTError as exc:
            return f"[ERROR] get_conditions_static failed: {exc}"

        suite = get_shared_suite()
        if suite is not None:
            if suite.total_mcdc_conditions == 0:
                suite.total_mcdc_conditions = result.total_mcdc_pairs
            suite.all_conditions = [c.condition for c in result.conditions]
            for c in result.conditions:
                cid = getattr(c, "node_id", None)
                if cid is None:
                    raise RuntimeError(
                        "Backend payload missing nodeId in /api/node/conditions. "
                        "nodeId is required for MC/DC identity."
                    )
                suite.condition_id_to_text[cid] = c.condition
                suite.condition_id_to_line[cid] = c.line_in_function

        if not result.conditions:
            return (
                f"No MC/DC conditions found for '{absolute_path}'. "
                "The function may have no branching logic."
            )

        lines = [
            f"Found {result.total_conditions} conditions "
            f"({result.total_mcdc_pairs} MC/DC pairs to cover):\n"
        ]
        for i, c in enumerate(result.conditions, 1):
            loc = ""
            if c.line_in_function is not None:
                loc = f"  [line+{c.line_in_function}, offset {c.start_offset}–{c.end_offset}]"
            lines.append(f"  {i}. {c.condition!r}{loc}")

        return "\n".join(lines)
