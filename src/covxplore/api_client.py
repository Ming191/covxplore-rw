"""
Synchronous HTTP client for the AkaUT Spring Boot REST API.

All methods raise ``AkaUTError`` on non-2xx responses, so callers can
catch a single exception type without inspecting status codes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from covxplore.config import get_settings
from covxplore.status import normalize_test_status


class AkaUTError(Exception):
    """Raised when the AkaUT API returns an error or is unreachable."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Typed return types (lightweight; Pydantic models live in models.py)
# ---------------------------------------------------------------------------

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
class SourceResult:
    source: str
    value: str | None = None


@dataclass
class ConditionInfo:
    condition: str
    line_in_function: int | None
    start_offset: int | None
    end_offset: int | None


@dataclass
class ConditionsResult:
    absolute_path: str
    total_conditions: int
    conditions: list[ConditionInfo]

    @property
    def total_mcdc_pairs(self) -> int:
        """Each condition requires true + false branch → 2 pairs each."""
        return self.total_conditions * 2


@dataclass
class ExecuteResult:
    """Raw deserialized JSON from POST /api/testcase/execute."""
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
    def mcdc_coverage(self) -> dict:
        return self.raw.get("mcdcCoverage") or {}

    @property
    def unvisited_mcdc_conditions(self) -> list[dict]:
        return self.raw.get("unvisitedMcdcConditions") or []

    @property
    def condition_trace(self) -> list[dict]:
        return self.raw.get("conditionTrace") or []

    @property
    def trace_summary(self) -> dict | None:
        return self.raw.get("traceSummary")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AkaUTClient:
    """Thin synchronous wrapper around the AkaUT REST API.

    Uses a single shared ``httpx.Client`` for connection reuse.
    Thread-safety: the client is not shared across threads; create one
    instance per thread / agent worker.
    """

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

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

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
            raise AkaUTError(f"POST {url} → HTTP {r.status_code}: {body_text}", r.status_code)
        return r.json()

    # ------------------------------------------------------------------ #
    # Public API methods                                                   #
    # ------------------------------------------------------------------ #

    def get_function_context(self, absolute_path: str) -> ContextResult:
        """POST /api/context — returns the dependency context for a function."""
        absolute_path = absolute_path.replace("\\", "/")
        data = self._post("/api/context", {"absolutePath": absolute_path})
        if "error" in data:
            raise AkaUTError(data["error"])
        return ContextResult(context=data["context"])

    def get_node_source(self, absolute_path: str) -> SourceResult:
        """GET /api/node/source — returns raw C/C++ source of a node."""
        absolute_path = absolute_path.replace("\\", "/")
        data = self._get("/api/node/source", {"absolutePath": absolute_path})
        if "error" in data:
            raise AkaUTError(data["error"])
        return SourceResult(
            source=data["source"],
            value=data.get("value"),
        )

    def search_nodes(
        self,
        query: str,
        types: list[str] | None = None,
    ) -> list[NodeInfo]:
        """POST /api/search — search AST nodes by name/type."""
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

    def get_node_conditions(self, absolute_path: str) -> ConditionsResult:
        """GET /api/node/conditions — returns all MC/DC conditions via static CFG analysis."""
        absolute_path = absolute_path.replace("\\", "/")
        data = self._get("/api/node/conditions", {"absolutePath": absolute_path})
        if "error" in data:
            raise AkaUTError(data["error"])
        return ConditionsResult(
            absolute_path=data["absolutePath"],
            total_conditions=data["totalConditions"],
            conditions=[
                ConditionInfo(
                    condition=c["condition"],
                    line_in_function=c.get("lineInFunction"),
                    start_offset=c.get("startOffset"),
                    end_offset=c.get("endOffset"),
                )
                for c in (data.get("conditions") or [])
            ],
        )

    def execute_testcase(
        self,
        absolute_path: str,
        test_body: str,
        test_name: str | None = None,
    ) -> ExecuteResult:
        """POST /api/testcase/execute — compile, run, and return coverage."""
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
