import time
from types import SimpleNamespace

import pytest

from covxplore.agents.schemas import GenerateTestBatchAction
from covxplore.flows.generation_flow import GenerationFlowRunner
from covxplore.generation.prompt_context import StaticPromptData
from covxplore.generation.runtime_session import GenerationFlowState, TestSuiteCodec
from covxplore.generator import GenerationConfig
from covxplore.tools.execute_testcase import HardStop
from covxplore.types import CoverageDetail, TestResult, TestSuite, UnvisitedBranch, UnvisitedStatement


class _FakeBuilder:
    def system_prompt(self):
        return "system"

    def task_description(self, **kwargs):
        return "task"


@pytest.fixture(autouse=True)
def _fake_batch_execution(monkeypatch):
    def execute(tool, candidates):
        suite = tool._run_context.active_suite()
        for candidate in candidates:
            total = 1 if candidate.test_name == "deterministic_stop" else 3
            suite.add_result(
                TestResult(
                    test_name=candidate.test_name or "test",
                    test_body=candidate.test_body,
                    status="PASSED",
                    statement_coverage=CoverageDetail(visited=1, total=total),
                    branch_coverage=CoverageDetail(visited=1, total=total),
                    unvisited_statements=(
                        [] if total == 1 else [UnvisitedStatement(node_id=2, statement="later")]
                    ),
                    unvisited_branches=(
                        []
                        if total == 1
                        else [
                            UnvisitedBranch(
                                node_id=2,
                                condition="later",
                                true_visited=False,
                                false_visited=False,
                            )
                        ]
                    ),
                ),
                min_suite_size=99,
            )
        suite.record_batch(False)
        if any(candidate.test_name == "deterministic_stop" for candidate in candidates):
            raise HardStop("coverage_target", "done")
        return "batch executed"

    monkeypatch.setattr(
        "covxplore.tools.execute_testcase.ExecuteTestcaseBatchTool._run", execute
    )


class _FakeCrewInst:
    def __init__(self, calls, prompt, completion, candidates_per_batch=1):
        self.calls = calls
        self.candidates_per_batch = candidates_per_batch
        self._crew = SimpleNamespace(
            usage_metrics=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
        )

    def crew(self):
        return self

    @property
    def usage_metrics(self):
        return self._crew.usage_metrics

    def kickoff(self, inputs):
        call_index = len(self.calls)
        return SimpleNamespace(
            pydantic=GenerateTestBatchAction.model_validate(
                {
                    "candidates": [
                        {"test_name": f"t{call_index}_{i}", "test_body": "f();"}
                        for i in range(self.candidates_per_batch)
                    ]
                }
            )
        )


def test_flow_seeds_static_coverage_totals():
    calls = []

    def crew_builder(technique):
        calls.append(technique)
        return _FakeCrewInst(calls, 1, 1), _FakeBuilder()

    result = GenerationFlowRunner(
        crew_builder=crew_builder,
        static_fetcher=lambda path, version: StaticPromptData(
            "context", "source", total_statements=7, total_branches=4
        ),
    ).run(
        GenerationConfig(
            function_path="/f.cpp::f()",
            prompt_variant="none",
            max_batches=1,
            run_id="seed-totals",
        )
    )

    metrics = result.suite.coverage.metrics(result.suite.tests)
    assert metrics.total_statements == 7
    assert metrics.total_branches == 4


def test_flow_spawns_new_crew_per_batch_and_aggregates_tokens():
    calls = []
    fetched = []

    def crew_builder(technique):
        calls.append(technique)
        return _FakeCrewInst(calls, 10, 20), _FakeBuilder()

    config = GenerationConfig(
        function_path="/f.cpp::f()",
        prompt_variant="none",
        max_batches=2,
        redundant_streak_limit=99,
        run_id="run-flow",
    )
    result = GenerationFlowRunner(
        crew_builder=crew_builder,
        static_fetcher=lambda path, version: (
            fetched.append((path, version)) or StaticPromptData("context", "source")
        ),
    ).run(config)

    assert fetched == [("/f.cpp::f()", "v1")]
    assert len(calls) == 2
    assert result.stop_reason == "max_batches"
    assert result.total_input_tokens == 20
    assert result.total_output_tokens == 40
    assert result.batches_used == 2
    elapsed = result.elapsed_sec
    time.sleep(0.01)
    assert result.elapsed_sec == elapsed


def test_flow_persists_execution_feedback_between_sessions():
    prompts = []

    class FeedbackBuilder(_FakeBuilder):
        def task_description(self, **kwargs):
            feedback = kwargs.get("execution_feedback_text") or ""
            prompts.append(feedback)
            return feedback or "task"

    def crew_builder(technique):
        return _FakeCrewInst([], 1, 1), FeedbackBuilder()

    result = GenerationFlowRunner(
        crew_builder=crew_builder,
        static_fetcher=lambda path, version: StaticPromptData("context", "source"),
    ).run(
        GenerationConfig(
            function_path="/f.cpp::f()",
            prompt_variant="none",
            max_batches=2,
            run_id="feedback",
        )
    )

    assert result.batches_used == 2
    assert prompts == ["", "batch executed"]


def test_runner_prefers_flow_usage_metrics_over_zero_ledger(monkeypatch):
    class FakeFlow:
        def __init__(self, *, initial_state, **kwargs):
            suite = TestSuite("/f.cpp::f()")
            self.state = GenerationFlowState(
                config=initial_state.config,
                suite=TestSuiteCodec().dump_suite(suite),
                static_prompt={"context_text": "", "source_text": ""},
                stop_reason="agent_done",
                token_ledger={"chosen": {"prompt": 0, "completion": 0, "total": 0}},
            )
            self.usage_metrics = SimpleNamespace(prompt_tokens=123, completion_tokens=45)

        def kickoff(self):
            return self.state

    monkeypatch.setattr("covxplore.flows.generation_flow.GenerationFlow", FakeFlow)
    result = GenerationFlowRunner(
        crew_builder=lambda *args, **kwargs: None,
        static_fetcher=lambda path, version: StaticPromptData("context", "source"),
    ).run(
        GenerationConfig(
            function_path="/f.cpp::f()",
            prompt_variant="none",
            max_batches=1,
            run_id="run-flow-metrics",
        )
    )

    assert result.total_input_tokens == 123
    assert result.total_output_tokens == 45


def test_flow_records_usage_when_hard_stop_short_circuits_kickoff():
    class InterruptedCrew:
        usage_metrics = None

        def kickoff(self, inputs):
            return SimpleNamespace(
                pydantic=GenerateTestBatchAction.model_validate(
                    {
                        "candidates": [
                            {"test_name": "deterministic_stop", "test_body": "f();"}
                        ]
                    }
                )
            )

        def calculate_usage_metrics(self):
            self.usage_metrics = SimpleNamespace(prompt_tokens=77, completion_tokens=33)
            return self.usage_metrics

    class CrewInst:
        def __init__(self, crew):
            self._crew = crew

        def crew(self):
            return self._crew

    result = GenerationFlowRunner(
        crew_builder=lambda technique: (CrewInst(InterruptedCrew()), _FakeBuilder()),
        static_fetcher=lambda path, version: StaticPromptData("context", "source"),
    ).run(
        GenerationConfig(
            function_path="/f.cpp::f()",
            prompt_variant="none",
            max_batches=3,
            run_id="run-hard-stop-tokens",
        )
    )

    assert result.stop_reason == "coverage_target"
    assert result.total_input_tokens == 77
    assert result.total_output_tokens == 33
    assert result.suite.tests[0].token_input == 77
    assert result.suite.tests[0].token_output == 33
