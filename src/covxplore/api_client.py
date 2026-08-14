from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from covxplore.config import get_settings
from covxplore.status import normalize_test_status

__all__ = [
    "AkaUTError",
    "AkaUTClient",
    "NodeInfo",
    "ContextResult",
    "ContextV2Result",
    "SourceResult",
    "ConditionInfo",
    "PathStep",
    "ExecutionPath",
    "NodeConditionsResult",
    "ExecuteResult",
]


class AkaUTError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class NodeInfo:
    name: str
    qualified_name: str
    absolute_path: str
    type: str
    line: int


@dataclass
class ContextResult:
    context: str


@dataclass
class ContextV2Result:
    raw: dict[str, Any]


@dataclass
class SourceResult:
    source: str
    value: str | None = None


@dataclass
class ConditionInfo:
    node_id: int | None
    condition: str
    line_in_function: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass
class PathStep:
    node_id: int
    condition: str
    required_outcome: str


@dataclass
class ExecutionPath:
    execution_sequence: list[PathStep]
    target_node_id: int
    target_condition: str
    target_outcome: str


@dataclass
class NodeConditionsResult:
    absolute_path: str
    total_conditions: int
    total_statements: int | None
    total_branches: int | None
    conditions: list[ConditionInfo]
    execution_paths: list[ExecutionPath]


@dataclass
class ExecuteResult:
    raw: dict[str, Any]

    @property
    def test_name(self) -> str:
        return self.raw.get("testName", "")

    @property
    def status(self) -> str:
        return normalize_test_status(self.raw.get("status")).value

    @property
    def execute_log(self) -> str:
        return self.raw.get("executeLog", "")

    @property
    def statement_coverage(self) -> dict:
        return self.raw.get("statementCoverage") or {}

    @property
    def branch_coverage(self) -> dict:
        return self.raw.get("branchCoverage") or {}

    @property
    def trace_summary(self) -> dict | None:
        return self.raw.get("traceSummary")

    @property
    def unvisited_statements(self) -> list[dict]:
        return self.raw.get("unvisitedStatements") or []

    @property
    def unvisited_branches(self) -> list[dict]:
        return self.raw.get("unvisitedBranches") or []


class AkaUTClient:
    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        cfg = get_settings()
        self._base = (base_url or cfg.akaut_base_url).rstrip("/")
        self._timeout = timeout or cfg.request_timeout_sec
        self._http = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "AkaUTClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self._base}{path}"
        try:
            r = self._http.get(url, params=params)
        except httpx.RequestError as exc:
            raise AkaUTError(f"GET {url} failed: {exc}") from exc
        if not r.is_success:
            body = r.text[:300]
            raise AkaUTError(f"GET {url} → HTTP {r.status_code}: {body}", r.status_code)
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self._base}{path}"
        try:
            r = self._http.post(url, json=body)
        except httpx.RequestError as exc:
            raise AkaUTError(f"POST {url} failed: {exc}") from exc
        if not r.is_success:
            body_text = r.text[:300]
            raise AkaUTError(
                f"POST {url} → HTTP {r.status_code}: {body_text}", r.status_code
            )
        return r.json()

    def get_function_context(self, absolute_path: str) -> ContextResult:
        absolute_path = absolute_path.replace("\\", "/")
        data = self._post("/api/context", {"absolutePath": absolute_path})
        if "error" in data:
            raise AkaUTError(data["error"])
        return ContextResult(context=data["context"])

    def get_function_context_v2(self, absolute_path: str) -> ContextV2Result:
        absolute_path = absolute_path.replace("\\", "/")
        data = self._post("/api/context-v2", {"absolutePath": absolute_path})
        if "error" in data:
            raise AkaUTError(data["error"])
        return ContextV2Result(raw=data)

    def get_node_source(self, absolute_path: str) -> SourceResult:
        absolute_path = absolute_path.replace("\\", "/")
        data = self._get("/api/node/source", {"absolutePath": absolute_path})
        if "error" in data:
            raise AkaUTError(data["error"])
        return SourceResult(
            source=data["source"],
            value=data.get("value"),
        )

    def get_node_conditions(
        self,
        absolute_path: str,
        *,
        coverage_type: str = "BRANCH",
    ) -> NodeConditionsResult:
        """Return static structural coverage metadata and condition details."""
        absolute_path = absolute_path.replace("\\", "/")
        data = self._get(
            "/api/node/conditions",
            {"absolutePath": absolute_path, "coverageType": coverage_type},
        )
        if isinstance(data, dict) and "error" in data:
            raise AkaUTError(data["error"])
        conditions = [
            ConditionInfo(
                node_id=item.get("nodeId"),
                condition=item.get("condition") or "",
                line_in_function=item.get("lineInFunction"),
                start_offset=item.get("startOffset"),
                end_offset=item.get("endOffset"),
            )
            for item in (data.get("conditions") or [])
        ]
        execution_paths = [
            ExecutionPath(
                execution_sequence=[
                    PathStep(
                        node_id=int(step["nodeId"]),
                        condition=step.get("condition") or "",
                        required_outcome=step.get("requiredOutcome") or "",
                    )
                    for step in (item.get("executionSequence") or [])
                ],
                target_node_id=int(item["targetNodeId"]),
                target_condition=item.get("targetCondition") or "",
                target_outcome=item.get("targetOutcome") or "",
            )
            for item in (data.get("executionPaths") or [])
        ]
        return NodeConditionsResult(
            absolute_path=data.get("absolutePath") or absolute_path,
            total_conditions=int(data.get("totalConditions") or len(conditions)),
            total_statements=(
                int(data["totalStatements"]) if data.get("totalStatements") is not None else None
            ),
            total_branches=(
                int(data["totalBranches"]) if data.get("totalBranches") is not None else None
            ),
            conditions=conditions,
            execution_paths=execution_paths,
        )

    def search_nodes(
        self,
        query: str,
        types: list[str] | None = None,
    ) -> list[NodeInfo]:
        payload: dict[str, Any] = {"query": query}
        if types:
            payload["types"] = types
        data = self._post("/api/search", payload)
        if isinstance(data, dict) and "error" in data:
            raise AkaUTError(data["error"])
        return [
            NodeInfo(
                name=item.get("name", ""),
                qualified_name=item.get("qualifiedName", ""),
                absolute_path=item.get("absolutePath", ""),
                type=item.get("type", ""),
                line=item.get("line", 0),
            )
            for item in (data or [])
        ]

    def execute_testcase(
        self,
        absolute_path: str,
        test_body: str,
        test_name: str | None = None,
    ) -> ExecuteResult:
        absolute_path = absolute_path.replace("\\", "/")
        payload: dict[str, Any] = {
            "absolutePath": absolute_path,
            "testBody": test_body,
        }
        if test_name:
            payload["testName"] = test_name
        data = self._post("/api/testcase/execute", payload)
        if isinstance(data, dict) and "error" in data:
            raise AkaUTError(data["error"])
        return ExecuteResult(raw=data)
