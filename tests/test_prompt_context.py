from types import SimpleNamespace

from covxplore.generation import prompt_context


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def get_node_conditions(self, path):
        return SimpleNamespace(total_conditions=0, total_mcdc_pairs=0, conditions=[])

    def get_function_context(self, path):
        return SimpleNamespace(context="legacy context")

    def get_function_context_v2(self, path):
        raise AssertionError("v2 should not be fetched by default")

    def get_node_source(self, path):
        return SimpleNamespace(source="int f() { return 0; }")


def test_fetch_static_prompt_data_uses_legacy_context(monkeypatch):
    monkeypatch.setattr("covxplore.api_client.AkaUTClient", lambda: FakeClient())
    monkeypatch.setattr("covxplore.tools.get_source._number_lines", lambda s: s)

    data = prompt_context.fetch_static_prompt_data("f")

    assert data.context_text == "legacy context"
    assert "int f()" in data.source_text
