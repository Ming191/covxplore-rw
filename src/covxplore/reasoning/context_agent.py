from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from covxplore.reasoning.context_tools import AkaUTContextTools
from covxplore.reasoning.schemas import ContextAction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextRequest:
    function_path: str
    system_prompt: str
    task_prompt: str


class ContextAgent:
    def __init__(
        self,
        decide: Callable[[str, type[ContextAction]], ContextAction],
        tools: AkaUTContextTools | None = None,
        *,
        max_searches: int = 2,
        max_source_fetches: int = 2,
        max_evidence_chars: int = 12000,
    ):
        self._decide = decide
        self._tools = tools or AkaUTContextTools()
        self._max_searches = max_searches
        self._max_source_fetches = max_source_fetches
        self._max_evidence_chars = max_evidence_chars

    def resolve(self, request: ContextRequest) -> str:
        evidence: list[str] = []
        allowed_paths = {request.function_path.replace("\\", "/")}
        seen: set[str] = set()
        searches = 0
        source_fetches = 0
        invalid_actions = 0
        source_required = False

        while searches < self._max_searches or source_fetches < self._max_source_fetches:
            allowed_actions = (
                ("get_node_source", "done")
                if source_required
                else ("search_nodes", "get_node_source", "done")
            )
            action = self._decide(
                self._prompt(
                    request,
                    evidence,
                    allowed_actions,
                    self._max_searches - searches,
                    self._max_source_fetches - source_fetches,
                ),
                ContextAction,
            )
            if action.action == "done":
                break
            if action.action not in allowed_actions:
                evidence.append(
                    f"Action {action.action} rejected: choose one of {', '.join(allowed_actions)}"
                )
                invalid_actions += 1
                if invalid_actions >= 1:
                    break
                continue
            if action.action == "search_nodes" and searches >= self._max_searches:
                evidence.append("Action search_nodes rejected: search budget exhausted")
                break
            if action.action == "get_node_source" and source_fetches >= self._max_source_fetches:
                evidence.append("Action get_node_source rejected: source budget exhausted")
                break
            key = json.dumps(action.model_dump(), sort_keys=True)
            if key in seen:
                logger.info("Context agent stopped after duplicate action %s", action.action)
                break
            seen.add(key)
            result = self._tools.execute(action, allowed_paths)
            allowed_paths.update(result.discovered_paths)
            evidence.append(result.text)
            if action.action == "search_nodes":
                searches += 1
                source_required = bool(result.discovered_paths)
            else:
                source_fetches += 1
                source_required = False
            logger.warning(
                "Agentic context tool: %s query=%r path=%r result_chars=%d",
                action.action,
                action.query,
                action.absolute_path,
                len(result.text),
            )
            if sum(len(item) for item in evidence) >= self._max_evidence_chars:
                logger.info("Context agent reached evidence limit")
                break

        if not evidence:
            return ""
        joined = "\n\n".join(evidence)
        return joined[: self._max_evidence_chars]

    @staticmethod
    def _prompt(
        request: ContextRequest,
        evidence: list[str],
        allowed_actions: tuple[str, ...],
        searches_left: int,
        source_fetches_left: int,
    ) -> str:
        evidence_text = "\n\n".join(evidence) or "None"
        allowed = ", ".join(allowed_actions)
        return f"""{request.system_prompt}

You are the read-only context-discovery phase for C/C++ test generation.
Decide whether the supplied focal context is sufficient to write compilable tests. Search only to resolve missing declarations, types, constants, helper behavior, or APIs. Request one action per turn. Never request test execution. Finish as soon as evidence is sufficient.

FOCAL FUNCTION PATH:
{request.function_path}

GENERATION TASK AND PRELOADED CONTEXT:
{request.task_prompt}

DISCOVERED EVIDENCE:
{evidence_text}

Allowed actions this turn: {allowed}
Remaining budget: {searches_left} searches, {source_fetches_left} source fetches.
- search_nodes: set query and optional types from FUNCTION, VARIABLE, ENUM, ENUM_VALUE, TYPEDEF, MACRO, STRUCT, CLASS.
- get_node_source: choose one absolute_path from the latest search results and read its declaration before searching again.
- done: use when no additional evidence is needed.

Return JSON matching:
{{"action":"search_nodes|get_node_source|done","query":null,"types":[],"absolute_path":null}}"""
