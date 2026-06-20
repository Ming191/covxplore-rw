from __future__ import annotations

from typing import Protocol, runtime_checkable

from covxplore.api_client import AkaUTClient, ExecuteResult


@runtime_checkable
class TestCaseExecutor(Protocol):
    """Backend that compiles/runs a generated test body."""

    def execute(
        self,
        absolute_path: str,
        test_body: str,
        test_name: str | None = None,
    ) -> ExecuteResult:
        ...


class AkaUTExecutor:
    """Current executor: delegate execution to AkaUT REST API."""

    def execute(
        self,
        absolute_path: str,
        test_body: str,
        test_name: str | None = None,
    ) -> ExecuteResult:
        with AkaUTClient() as client:
            return client.execute_testcase(absolute_path, test_body, test_name)
