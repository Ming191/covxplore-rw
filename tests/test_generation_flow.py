import time
from types import SimpleNamespace

from covxplore.generation.prompt_context import StaticPromptData
from covxplore.generator import GenerationConfig
from covxplore.flows.generation_flow import GenerationFlowRunner
from covxplore.generation.runtime_session import GenerationFlowState, TestSuiteCodec
from covxplore.tools.execute_testcase import HardStop
from covxplore.types import CoverageDetail, TestResult, TestSuite, UnvisitedBranch, UnvisitedStatement


class _FakeBuilder:
    def system_prompt(self):
        return "system"

    def task_description(self, **kwargs):
        return "task"


class _KnowledgeBuilder(_FakeBuilder):
    def task_description(self, **kwargs):
        return kwargs.get("dynamic_knowledge_text") or "task"


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

    def crew_builder(prompt_config, *, agent_max_iter, start_batch, run_context, reasoning=False):
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
    elapsed = result.elapsed_sec
    time.sleep(0.01)
    assert result.elapsed_sec == elapsed


def test_flow_persists_dynamic_knowledge_between_sessions(monkeypatch):
    calls = []

    class KnowledgeCrew:
        usage_metrics = None

        def __init__(self, run_context):
            self.run_context = run_context

        def kickoff(self, inputs):
            prior_knowledge = dict(self.run_context.dynamic_knowledge)
            calls.append((prior_knowledge, inputs["task_description"]))
            if not prior_knowledge:
                self.run_context.remember_knowledge("search:Parser:CLASS", "Found Parser")
            suite = self.run_context.active_suite()
            suite.add_result(
                TestResult(
                    test_name=f"t{len(calls)}", test_body="f();", status="PASSED",
                    statement_coverage=CoverageDetail(visited=1, total=3),
                    branch_coverage=CoverageDetail(visited=1, total=3),
                    unvisited_statements=[UnvisitedStatement(node_id=2, statement="later")],
                    unvisited_branches=[
                        UnvisitedBranch(
                            node_id=2,
                            condition="later",
                            true_visited=False,
                            false_visited=False,
                        )
                    ],
                ),
                min_suite_size=99,
            )
            suite.record_batch(False)

    class CrewInst:
        def __init__(self, run_context):
            self._crew = KnowledgeCrew(run_context)

        def crew(self):
            return self._crew

    def crew_builder(prompt_config, *, agent_max_iter, start_batch, run_context, reasoning=False):
        return CrewInst(run_context), _KnowledgeBuilder()

    monkeypatch.setattr("covxplore.flows.generation_flow.init_observability", lambda: None)
    monkeypatch.setattr("covxplore.flows.generation_flow.trace_observation", _null_trace)
    monkeypatch.setattr("covxplore.generation.tokens.TokenLedger.record_trace", lambda self, trace_id: None)

    config = GenerationConfig(function_path="/f.cpp::f()", prompt_variant="full", max_batches=2, run_id="cache")
    result = GenerationFlowRunner(
        crew_builder=crew_builder,
        static_fetcher=lambda path: StaticPromptData("context", "source"),
    ).run(config)

    assert result.batches_used == 2
    assert calls[1][0] == {"search:Parser:CLASS": "Found Parser"}
    assert calls[1][1] == "[search:Parser:CLASS]\nFound Parser"


def test_flow_stops_on_max_batches_not_candidate_count(monkeypatch):
    calls = []

    def crew_builder(prompt_config, *, agent_max_iter, start_batch, run_context, reasoning=False):
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


def test_runner_prefers_flow_usage_metrics_over_zero_ledger(monkeypatch):
    class FakeFlow:
        def __init__(self, *, initial_state, **kwargs):
            suite = TestSuite("/f.cpp::f()")
            self.state = GenerationFlowState(
                config=initial_state.config,
                suite=TestSuiteCodec().dump_suite(suite),
                static_prompt={"context_text": "", "source_text": "", "branch_catalog_text": "", "branch_catalog_ids": []},
                stop_reason="agent_done",
                token_ledger={"chosen": {"prompt": 0, "completion": 0, "total": 0}},
            )
            self.usage_metrics = SimpleNamespace(prompt_tokens=123, completion_tokens=45)

        def kickoff(self):
            return self.state

    monkeypatch.setattr("covxplore.flows.generation_flow.GenerationFlow", FakeFlow)

    config = GenerationConfig(
        function_path="/f.cpp::f()",
        prompt_variant="full",
        max_batches=1,
        run_id="run-flow-metrics",
    )
    result = GenerationFlowRunner(
        crew_builder=lambda *args, **kwargs: None,
        static_fetcher=lambda path: StaticPromptData("context", "source"),
    ).run(config)

    assert result.total_input_tokens == 123
    assert result.total_output_tokens == 45


def test_flow_records_usage_when_hard_stop_short_circuits_kickoff(monkeypatch):
    class InterruptedCrew:
        usage_metrics = None

        def __init__(self, run_context):
            self.run_context = run_context

        def kickoff(self, inputs):
            suite = self.run_context.active_suite()
            suite.add_result(
                TestResult(
                    test_name="deterministic_stop",
                    test_body="f();",
                    status="PASSED",
                    statement_coverage=CoverageDetail(visited=1, total=1),
                    branch_coverage=CoverageDetail(visited=1, total=1),
                ),
                min_suite_size=99,
            )
            suite.record_batch(False)
            raise HardStop("coverage_target", "done")

        def calculate_usage_metrics(self):
            self.usage_metrics = SimpleNamespace(prompt_tokens=77, completion_tokens=33)
            return self.usage_metrics

    class CrewInst:
        def __init__(self, crew):
            self._crew = crew

        def crew(self):
            return self._crew

    def crew_builder(prompt_config, *, agent_max_iter, start_batch, run_context, reasoning=False):
        return CrewInst(InterruptedCrew(run_context)), _FakeBuilder()

    monkeypatch.setattr("covxplore.flows.generation_flow.init_observability", lambda: None)
    monkeypatch.setattr("covxplore.flows.generation_flow.trace_observation", _null_trace)
    monkeypatch.setattr("covxplore.generation.tokens.TokenLedger.record_trace", lambda self, trace_id: None)

    config = GenerationConfig(
        function_path="/f.cpp::f()",
        prompt_variant="full",
        max_batches=3,
        run_id="run-hard-stop-tokens",
    )
    result = GenerationFlowRunner(
        crew_builder=crew_builder,
        static_fetcher=lambda path: StaticPromptData("context", "source"),
    ).run(config)

    assert result.stop_reason == "coverage_target"
    assert result.total_input_tokens == 77
    assert result.total_output_tokens == 33
    assert result.suite.tests[0].token_input == 77
    assert result.suite.tests[0].token_output == 33


class _null_trace:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return None
