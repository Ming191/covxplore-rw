"""GetFunctionContextTool — wraps POST /api/context."""
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from covxplore.api_client import AkaUTClient, AkaUTError


class _Input(BaseModel):
    absolute_path: str = Field(
        ...,
        description=(
            "The absolute path of the function node as returned by search_nodes, "
            "e.g. '/project/src/foo.cpp\\\\MyNS::bar(int)'."
        ),
    )


class GetFunctionContextTool(BaseTool):
    """Fetch the dependency context for a C/C++ function from AkaUT.

    Use this FIRST when starting test generation for a new function.
    The context includes: parameter types, return type, dependent structs/
    classes, stub information, and global variables — everything needed to
    write a compilable test driver.
    """

    name: str = "get_function_context"
    description: str = (
        "Retrieve the full dependency context (types, globals, stubs) for a "
        "C/C++ function identified by its absolutePath. Call this once at the "
        "start of test generation to understand the function's signature and "
        "all required input types."
    )
    args_schema: Type[BaseModel] = _Input

    def _run(self, absolute_path: str) -> str:
        try:
            with AkaUTClient() as client:
                result = client.get_function_context(absolute_path)
            return result.context
        except AkaUTError as exc:
            return f"[ERROR] get_function_context failed: {exc}"
