import pytest

from covxplore.api_client import AkaUTError, ExecuteResult
from covxplore.driver import AkaUTExecutor, TestCaseExecutor


def test_akaut_executor_matches_protocol():
    executor: TestCaseExecutor = AkaUTExecutor()

    assert executor is not None
    assert isinstance(executor, TestCaseExecutor)


def test_akaut_executor_calls_client(monkeypatch):
    calls = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def execute_testcase(self, absolute_path, test_body, test_name=None):
            calls.append((absolute_path, test_body, test_name))
            return ExecuteResult(raw={"testName": "t1", "status": "PASSED"})

    monkeypatch.setattr("covxplore.driver.executor.AkaUTClient", FakeClient)

    result = AkaUTExecutor().execute("/x.cpp::f()", "f();", "t1")

    assert result.status == "PASSED"
    assert calls == [("/x.cpp::f()", "f();", "t1")]


def test_akaut_executor_propagates_client_error(monkeypatch):
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def execute_testcase(self, *_args, **_kwargs):
            raise AkaUTError("boom")

    monkeypatch.setattr("covxplore.driver.executor.AkaUTClient", FakeClient)

    with pytest.raises(AkaUTError, match="boom"):
        AkaUTExecutor().execute("/x.cpp::f()", "f();", "t1")
