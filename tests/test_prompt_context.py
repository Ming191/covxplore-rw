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
            total_statements=7,
            total_branches=4,
            conditions=[
                ConditionInfo(node_id=28, condition="p.ch == '.'", line_in_function=32),
                ConditionInfo(node_id=51, condition="stopAtNext", line_in_function=53),
            ],
        )

    def get_function_context(self, path):
        return SimpleNamespace(context="legacy context")

    def get_function_context_v2(self, path):
        return SimpleNamespace(
            raw={
                "focal": {"signature": "int f()"},
                "inputs": [
                    {
                        "name": "p",
                        "type": "Parser*",
                        "structure": {
                            "name": "Hjson::Parser",
                            "publicFields": [{"name": "ch", "type": "unsigned char"}],
                        },
                    }
                ],
                "conditions": [
                    {
                        "nodeId": 28,
                        "lineInFunction": 32,
                        "expression": "p.ch == '.'",
                        "reads": [{"base": "p", "field": "ch"}],
                    }
                ],
                "fieldAccesses": [
                    {
                        "access": "read",
                        "base": "p",
                        "field": "ch",
                        "lineInFunction": 32,
                        "expression": "p.ch",
                    }
                ],
                "callSites": [
                    {
                        "callerSignature": "int caller(Parser*)",
                        "argumentMap": [{"focalParam": "p", "callerExpr": "&parser"}],
                        "statementsBeforeCall": [{"line": 12, "text": "return f(&parser);"}],
                    }
                ],
                "helperEffects": [
                    {
                        "signature": "bool Hjson::_next(Hjson::Parser*)",
                        "reads": [{"base": "p", "field": "indexNext"}],
                        "writes": [{"base": "p", "field": "ch"}],
                        "calls": [],
                        "sourceMode": "full",
                        "source": "static bool _next(Parser *p) { return ++p->indexNext; }",
                    }
                ],
            }
        )

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
    assert data.total_statements == 7
    assert data.total_branches == 4


def test_fetch_static_prompt_data_uses_v2_provider(monkeypatch):
    monkeypatch.setattr("covxplore.api_client.AkaUTClient", lambda: FakeClient())
    monkeypatch.setattr("covxplore.tools.get_source._number_lines", lambda s: s)

    data = prompt_context.fetch_static_prompt_data("f", "v2")

    assert "STATIC PROGRAM FACTS (context-v2)" in data.context_text
    assert "Focal: int f()" in data.context_text
    assert "- Parser* p" in data.context_text
    assert "- unsigned char ch" in data.context_text
    assert "reads=p->ch" in data.context_text
    assert "Call sites:" in data.context_text
    assert "arg p <- &parser" in data.context_text
    assert "base=None" not in data.context_text
    assert "Helper effects:" in data.context_text
    assert "bool Hjson::_next(Hjson::Parser*)" in data.context_text
    assert "static bool _next(Parser *p)" in data.context_text
    assert "legacy context" not in data.context_text


def test_get_context_provider_rejects_unknown_version():
    try:
        prompt_context.get_context_provider("v3")
    except ValueError as exc:
        assert str(exc) == "context version must be 'v1' or 'v2'"
    else:
        raise AssertionError("unknown context version must fail")


def test_format_branch_catalog_warns_against_line_as_id():
    text = prompt_context.format_branch_catalog(
        [ConditionInfo(node_id=6, condition="p.ch == '-'", line_in_function=13)]
    )
    assert "Do NOT invent nodeIds from source line numbers" in text
    assert "[nodeId=6]" in text
