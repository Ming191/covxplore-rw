from types import SimpleNamespace

from covxplore.generation.prompt_context import StaticPromptData
from covxplore.generator import GenerationConfig
from covxplore.flows.generation_flow import GenerationFlowRunner
from covxplore.types import CoverageDetail, TestResult, UnvisitedBranch, UnvisitedStatement


class _FakeBuilder:
    def system_prompt(self):
        return "system"

    def task_description(self, **kwargs):
        return "task"


class _FakeCrewInst:
    def __init__(self, calls, prompt, completion, candidates_per_batch=1, cover_false_on_second=True):
        self.calls = calls
        self.candidates_per_batch = candidates_per_batch
        self.cover_false_on_second = cover_false_on_second
        self._crew = SimpleNamespace(
            usage_metrics=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
        )

    def crew(self):
        return self

    @property
    def usage_metrics(self):
        return self._crew.usage_metrics

    def kickoff(self, inputs):
        ctx = self.calls[-1]["run_context"]
        suite = ctx.active_suite()
        call_index = len(self.calls)
        for i in range(self.candidates_per_batch):
            result = TestResult(
                test_name=f"t{call_index}_{i}", test_body="f();", status="PASSED",
                statement_coverage=CoverageDetail(visited=1, total=3),
                branch_coverage=CoverageDetail(visited=1, total=3),
                unvisited_statements=[UnvisitedStatement(node_id=2, statement="later")],
                unvisited_branches=[UnvisitedBranch(node_id=2, condition="later", true_visited=False, false_visited=False)],
            )
            suite.add_result(result, min_suite_size=99)
        suite.record_batch(False)


def test_flow_spawns_new_crew_per_batch_and_aggregates_tokens(monkeypatch):
    calls = []

    def crew_builder(prompt_config, *, agent_max_iter, start_batch, run_context):
        calls.append({"start_batch": start_batch, "run_context": run_context})
        return _FakeCrewInst(calls, 10, 20), _FakeBuilder()

    monkeypatch.setattr("covxplore.flows.generation_flow.init_observability", lambda: None)
    monkeypatch.setattr("covxplore.flows.generation_flow.trace_observation", _null_trace)
    monkeypatch.setattr("covxplore.generation.tokens.TokenLedger.record_trace", lambda self, trace_id: None)

    config = GenerationConfig(
        function_path="/f.cpp::f()",
        prompt_variant="full",
        max_batches=2,
        run_id="run-flow",
    )
    config.redundant_streak_limit = 99
    result = GenerationFlowRunner(
        crew_builder=crew_builder,
        static_fetcher=lambda path: StaticPromptData("context", "source"),
    ).run(config)

    assert len(calls) == 2
    assert [call["start_batch"] for call in calls] == [0, 1]
    assert result.stop_reason == "max_batches"
    assert result.total_input_tokens == 20
    assert result.total_output_tokens == 40
    assert result.batches_used == 2
    assert len(result.suite.tests) == 2


def test_flow_stops_on_max_batches_not_candidate_count(monkeypatch):
    calls = []

    def crew_builder(prompt_config, *, agent_max_iter, start_batch, run_context):
        calls.append({"start_batch": start_batch, "run_context": run_context})
        return _FakeCrewInst(
            calls,
            1,
            1,
            candidates_per_batch=2,
            cover_false_on_second=False,
        ), _FakeBuilder()

    monkeypatch.setattr("covxplore.flows.generation_flow.init_observability", lambda: None)
    monkeypatch.setattr("covxplore.flows.generation_flow.trace_observation", _null_trace)
    monkeypatch.setattr("covxplore.generation.tokens.TokenLedger.record_trace", lambda self, trace_id: None)

    config = GenerationConfig(
        function_path="/f.cpp::f()",
        prompt_variant="full",
        max_batches=3,
        run_id="run-max",
    )
    config.redundant_streak_limit = 99
    result = GenerationFlowRunner(
        crew_builder=crew_builder,
        static_fetcher=lambda path: StaticPromptData("context", "source"),
    ).run(config)

    assert len(calls) == 3
    assert [call["start_batch"] for call in calls] == [0, 1, 2]
    assert result.stop_reason == "max_batches"
    assert result.batches_used == 3
    assert result.accepted_test_count == 6


class _null_trace:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return None
