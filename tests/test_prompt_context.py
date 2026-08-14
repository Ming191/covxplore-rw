from types import SimpleNamespace

from covxplore.generation import prompt_context


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def get_node_conditions(self, path, *, coverage_type="BRANCH"):
        return SimpleNamespace(total_statements=7, total_branches=4)

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
                "externalDeclarations": [
                    {
                        "kind": "enum",
                        "name": "Hjson::Settings",
                        "source": "enum Settings { Strict, Permissive };",
                        "usedBy": ["focal", "helper:is_strict"],
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
                "helpers": [
                    {
                        "signature": "bool Hjson::_next(Hjson::Parser*)",
                        "source": "static bool _next(Parser *p) { return ++p->indexNext; }",
                    }
                ],
            }
        )

    def get_node_source(self, path):
        return SimpleNamespace(source="int f() { return 0; }")


def test_fetch_static_prompt_data_includes_context_source_and_totals(monkeypatch):
    monkeypatch.setattr("covxplore.api_client.AkaUTClient", lambda: FakeClient())
    data = prompt_context.fetch_static_prompt_data("f")
    assert data.context_text == "legacy context"
    assert "int f()" in data.source_text
    assert data.total_statements == 7
    assert data.total_branches == 4


def test_fetch_static_prompt_data_uses_v2_provider(monkeypatch):
    monkeypatch.setattr("covxplore.api_client.AkaUTClient", lambda: FakeClient())
    data = prompt_context.fetch_static_prompt_data("f", "v2")
    assert "STATIC PROGRAM FACTS (context-v2)" in data.context_text
    assert "Focal: int f()" in data.context_text
    assert "- Parser* p" in data.context_text
    assert "reads=p->ch" in data.context_text
    assert "[enum] Hjson::Settings used by focal, helper:is_strict" in data.context_text
    assert "enum Settings { Strict, Permissive };" in data.context_text
    assert "static bool _next(Parser *p)" in data.context_text
    assert "legacy context" not in data.context_text


def test_get_context_provider_rejects_unknown_version():
    try:
        prompt_context.get_context_provider("v3")
    except ValueError as exc:
        assert str(exc) == "context version must be 'v1' or 'v2'"
    else:
        raise AssertionError("unknown context version must fail")
