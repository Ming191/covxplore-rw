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


def create_executor(
    backend: str = "akaut",
    *,
    function_path: str | None = None,
    gtest_source_root: str | None = None,
    gtest_project_sources: list[str] | None = None,
    gtest_extra_compile_flags: list[str] | None = None,
    gtest_coverage_backend: str = "gcov",
    gtest_compiler: str | None = None,
) -> TestCaseExecutor:
    """Build a TestCaseExecutor from a backend name (used by generator / CLI)."""
    name = backend.strip().lower()
    if name in ("akaut", "aka", "rest"):
        return AkaUTExecutor()
    if name in ("gtest", "gcov", "local"):
        from covxplore.driver.gtest_executor import infer_gtest_config, make_gtest_executor

        inferred: dict[str, object] = {}
        if function_path:
            inferred = infer_gtest_config(
                function_path,
                source_root=gtest_source_root,
                project_sources=gtest_project_sources,
                extra_compile_flags=gtest_extra_compile_flags,
            )

        root = gtest_source_root or inferred.get("gtest_source_root")
        sources = gtest_project_sources
        if sources is None:
            sources = inferred.get("gtest_project_sources")  # type: ignore[assignment]
        flags = gtest_extra_compile_flags
        if not flags:
            flags = inferred.get("gtest_extra_compile_flags")  # type: ignore[assignment]

        return make_gtest_executor(
            source_root=root,
            project_sources=sources,
            extra_compile_flags=flags,
            coverage_backend=gtest_coverage_backend,
            compiler=gtest_compiler,
            function_path=function_path,
        )
    raise ValueError(
        f"Unknown executor backend {backend!r}; expected 'akaut' or 'gtest'"
    )
