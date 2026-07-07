from dataclasses import dataclass

from covxplore.api_client import AkaUTError
from covxplore.tools.get_source import GetNodeSourceTool


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


def test_get_source_maps_404_to_terminal_unavailable_note(monkeypatch):
    monkeypatch.setattr("covxplore.tools.get_source.AkaUTClient", FakeClient)

    result = GetNodeSourceTool()._run("E:/p.cpp/Hjson/Encoder/field")

    assert result.startswith("[SOURCE_UNAVAILABLE]")
    assert "No source body exists" in result
    assert "Do not retry this path" in result
    assert "// Source:" not in result
