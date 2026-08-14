from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from covxplore.api_client import AkaUTClient
from covxplore.reasoning.schemas import ContextAction

_ALLOWED_TYPES = {
    "FUNCTION",
    "VARIABLE",
    "ENUM",
    "ENUM_VALUE",
    "TYPEDEF",
    "MACRO",
    "STRUCT",
    "CLASS",
}


@dataclass(frozen=True)
class ContextEvidence:
    text: str
    discovered_paths: set[str] = field(default_factory=set)


class AkaUTContextTools:
    def __init__(
        self,
        client_factory: Callable[[], AkaUTClient] = AkaUTClient,
        *,
        max_search_results: int = 5,
        max_result_chars: int = 4000,
    ):
        self._client_factory = client_factory
        self._max_search_results = max_search_results
        self._max_result_chars = max_result_chars

    def execute(self, action: ContextAction, allowed_paths: set[str]) -> ContextEvidence:
        try:
            if action.action == "search_nodes":
                return self._search(action)
            path = (action.absolute_path or "").replace("\\", "/")
            if not path or path not in allowed_paths:
                return ContextEvidence(f"Tool rejected path not returned by search: {path!r}")
            with self._client_factory() as client:
                source = client.get_node_source(path)
            text = source.source
            if source.value is not None:
                text += f"\nValue: {source.value}"
            return ContextEvidence(self._cap(f"Source for {path}:\n{text}"))
        except Exception as exc:
            return ContextEvidence(f"Tool {action.action} failed: {type(exc).__name__}: {exc}")

    def _search(self, action: ContextAction) -> ContextEvidence:
        query = (action.query or "").strip()
        if not query:
            return ContextEvidence("Tool search_nodes rejected an empty query")
        requested = {item.upper() for item in action.types}
        invalid = requested - _ALLOWED_TYPES
        if invalid:
            return ContextEvidence(
                "Tool search_nodes rejected unsupported types: " + ", ".join(sorted(invalid))
            )
        types = sorted(requested or _ALLOWED_TYPES)
        with self._client_factory() as client:
            results = client.search_nodes(query, types)[: self._max_search_results]
        paths = {item.absolute_path.replace("\\", "/") for item in results if item.absolute_path}
        payload = [
            {
                "name": item.name,
                "qualifiedName": item.qualified_name,
                "absolutePath": item.absolute_path,
                "type": item.type,
                "line": item.line,
            }
            for item in results
        ]
        return ContextEvidence(
            self._cap(f"Search results for {query!r}:\n{json.dumps(payload, ensure_ascii=False)}"),
            paths,
        )

    def _cap(self, text: str) -> str:
        if len(text) <= self._max_result_chars:
            return text
        return text[: self._max_result_chars] + "\n... [tool result truncated]"
