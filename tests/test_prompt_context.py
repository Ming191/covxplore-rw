from types import SimpleNamespace

from covxplore.generation import prompt_context
from covxplore.api_client import ConditionInfo


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def get_node_conditions(self, path, *, coverage_type="BRANCH"):
        assert coverage_type == "BRANCH"
        return SimpleNamespace(
            total_conditions=2,
            conditions=[
                ConditionInfo(node_id=28, condition="p.ch == '.'", line_in_function=32),
                ConditionInfo(node_id=51, condition="stopAtNext", line_in_function=53),
            ],
        )

    def get_function_context(self, path):
        return SimpleNamespace(context="legacy context")

    def get_function_context_v2(self, path):
        raise AssertionError("v2 should not be fetched by default")

    def get_node_source(self, path):
        return SimpleNamespace(source="int f() { return 0; }")


def test_fetch_static_prompt_data_includes_branch_catalog(monkeypatch):
    monkeypatch.setattr("covxplore.api_client.AkaUTClient", lambda: FakeClient())
    monkeypatch.setattr("covxplore.tools.get_source._number_lines", lambda s: s)

    data = prompt_context.fetch_static_prompt_data("f")

    assert data.context_text == "legacy context"
    assert "int f()" in data.source_text
    assert "BRANCH NODE CATALOG" in data.branch_catalog_text
    assert "[nodeId=28]" in data.branch_catalog_text
    assert "[nodeId=51]" in data.branch_catalog_text
    assert "line+32" in data.branch_catalog_text
    assert data.branch_catalog_ids == [28, 51]


def test_format_branch_catalog_warns_against_line_as_id():
    text = prompt_context.format_branch_catalog(
        [ConditionInfo(node_id=6, condition="p.ch == '-'", line_in_function=13)]
    )
    assert "Do NOT invent nodeIds from source line numbers" in text
    assert "[nodeId=6]" in text
