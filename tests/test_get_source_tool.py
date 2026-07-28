from dataclasses import dataclass

from covxplore.api_client import AkaUTError
from covxplore.tools.get_source import GetNodeSourceTool
from covxplore.tools.execute_testcase import RunContext


@dataclass
class Source:
    source: str
    value: str | None = None


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get_node_source(self, path):
        raise AkaUTError("missing", 404)


def test_get_source_reuses_shared_dynamic_knowledge(monkeypatch):
    calls = []

    class SourceClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get_node_source(self, path):
            calls.append(path)
            return Source("void helper() {}")

    monkeypatch.setattr("covxplore.tools.get_source.AkaUTClient", SourceClient)
    context = RunContext()

    first = GetNodeSourceTool(run_context=context)._run("E:\\p.cpp\\helper()")
    second = GetNodeSourceTool(run_context=context)._run("E:/p.cpp/helper()")

    assert calls == ["E:\\p.cpp\\helper()"]
    assert second == first


def test_get_source_maps_404_to_terminal_unavailable_note(monkeypatch):
    monkeypatch.setattr("covxplore.tools.get_source.AkaUTClient", FakeClient)

    result = GetNodeSourceTool()._run("E:/p.cpp/Hjson/Encoder/field")

    assert result.startswith("[SOURCE_UNAVAILABLE]")
    assert "No source body exists" in result
    assert "Do not retry this path" in result
    assert "// Source:" not in result
