import json
from types import SimpleNamespace

from covxplore.reasoning.context_agent import ContextAgent, ContextRequest
from covxplore.reasoning.context_tools import AkaUTContextTools
from covxplore.reasoning.schemas import ContextAction


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def search_nodes(self, query, types):
        assert query == "Settings"
        assert "VARIABLE" in types
        return [
            SimpleNamespace(
                name="Settings",
                qualified_name="jsonxx::Settings",
                absolute_path="/jsonxx/jsonxx.cc/jsonxx/Settings",
                type="VARIABLE",
                line=45,
            )
        ]

    def get_node_source(self, path):
        assert path == "/jsonxx/jsonxx.cc/jsonxx/Settings"
        return SimpleNamespace(source="ParserMode Settings = Strict;", value=None)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def decide(self, prompt, schema):
        self.prompts.append(prompt)
        return schema.model_validate_json(next(self.responses))


def _request(feedback=None):
    return ContextRequest(
        function_path="/jsonxx/parse_string",
        system_prompt="system",
        task_prompt="task" + (f"\nPRIOR EXECUTION FEEDBACK:\n{feedback}" if feedback else ""),
    )


def test_agent_searches_then_fetches_source():
    llm = FakeLLM([
        json.dumps({
            "action": "search_nodes",
            "query": "Settings",
            "types": ["VARIABLE"],
            "absolute_path": None,
        }),
        json.dumps({
            "action": "get_node_source",
            "query": None,
            "types": [],
            "absolute_path": "/jsonxx/jsonxx.cc/jsonxx/Settings",
        }),
        json.dumps({
            "action": "done",
            "query": None,
            "types": [],
            "absolute_path": None,
        }),
    ])
    agent = ContextAgent(
        llm.decide,
        AkaUTContextTools(client_factory=FakeClient),
    )

    evidence = agent.resolve(_request("jsonxx::Parser is not assignable"))

    assert "jsonxx::Settings" in evidence
    assert "ParserMode Settings = Strict" in evidence
    assert "jsonxx::Parser is not assignable" in llm.prompts[0]
    assert "Search results for 'Settings'" in llm.prompts[1]


def test_agent_rejects_unsearched_path():
    tools = AkaUTContextTools(client_factory=FakeClient)
    action = ContextAction(action="get_node_source", absolute_path="/other/path")
    result = tools.execute(action, {"/jsonxx/parse_string"})
    assert "rejected path" in result.text


def test_agent_requires_source_after_successful_search():
    search = json.dumps({
        "action": "search_nodes",
        "query": "Settings",
        "types": ["VARIABLE"],
        "absolute_path": None,
    })
    llm = FakeLLM([search, search])
    agent = ContextAgent(llm.decide, AkaUTContextTools(client_factory=FakeClient))

    evidence = agent.resolve(_request())

    assert evidence.count("Search results") == 1
    assert "rejected: choose one of get_node_source, done" in evidence
    assert len(llm.prompts) == 2
