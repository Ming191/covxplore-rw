from __future__ import annotations

from types import SimpleNamespace

from covxplore.api_client import AkaUTError
from covxplore.hybrid.llm_completion import CompletionUsage, IntentCompletionError
from covxplore.hybrid.models import (
    ActionKind,
    BranchEdge,
    CoverageDetail,
    CoverageModel,
    ExecutionResult,
    FunctionFeatures,
    SolverStatus,
    SourceRange,
    Statement,
    SymbolicAttemptResponse,
    TestAction,
    TestIntent,
    TargetKind,
)
from covxplore.hybrid.runner import HybridGenerationConfig, HybridGenerationRunner


def model(*, complex_cpp: bool = False) -> CoverageModel:
    return CoverageModel(
        function_path="D:/src/f.cpp/f(int)",
        function_name="f(int)",
        model_hash="model-hash",
        source="void f(int x) { use(x); }",
        statements=[
            Statement(
                key="s1",
                text="use(x);",
                source_range=SourceRange(
                    file="D:/src/f.cpp", start_offset=16, end_offset=23
                ),
                expected_path=["s1"],
            )
        ],
        features=FunctionFeatures(pointer_count=1 if complex_cpp else 0),
    )


def two_statement_model() -> CoverageModel:
    return CoverageModel(
        function_path="D:/src/f.cpp/f(int)",
        function_name="f(int)",
        model_hash="two-statement-model",
        source="void f(int x) { first(x); second(x); }",
        statements=[
            Statement(
                key=key,
                text=f"{key}();",
                source_range=SourceRange(
                    file="D:/src/f.cpp",
                    start_offset=index * 10,
                    end_offset=index * 10 + 5,
                ),
                expected_path=[key],
            )
            for index, key in enumerate(("s1", "s2"), start=1)
        ],
        features=FunctionFeatures(pointer_count=1),
    )


def many_statement_model(count: int = 5) -> CoverageModel:
    return CoverageModel(
        function_path="D:/src/f.cpp/f(int)",
        function_name="f(int)",
        model_hash="many-statement-model",
        source="void f(int x) {}",
        statements=[
            Statement(
                key=f"s{index}",
                text=f"s{index}();",
                source_range=SourceRange(
                    file="D:/src/f.cpp",
                    start_offset=index * 10,
                    end_offset=index * 10 + 5,
                ),
                expected_path=[f"s{index}"],
            )
            for index in range(1, count + 1)
        ],
        features=FunctionFeatures(pointer_count=1),
    )


def branch_and_statement_model() -> CoverageModel:
    source_range = SourceRange(
        file="D:/src/f.cpp", start_offset=10, end_offset=20
    )
    return CoverageModel(
        function_path="D:/src/f.cpp/f(int)",
        function_name="f(int)",
        model_hash="branch-and-statement-model",
        source="void f(int x) { if (x) use(); }",
        statements=[
            Statement(
                key="s1",
                text="if (x)",
                source_range=source_range,
                expected_path=["s1"],
            )
        ],
        branch_edges=[
            BranchEdge(
                key="b1",
                source_statement_key="s1",
                outcome="TRUE",
                source_range=source_range,
                expected_path=["s1", "b1"],
            )
        ],
    )


def executable_intent(target, intent_id: str = "intent-1") -> TestIntent:
    return TestIntent(
        intent_id=intent_id,
        function_path="D:/src/f.cpp/f(int)",
        target=target,
        actions=[
            TestAction(id="invoke", kind=ActionKind.INVOKE, code="f(1);")
        ],
    )


def execution(*, reached: bool, covered: bool) -> ExecutionResult:
    statements = ["s1"] if covered else []
    return ExecutionResult(
        test_name="tc",
        status="PASSED",
        statement_coverage=CoverageDetail(
            visited=len(statements), total=1, progress=float(bool(statements))
        ),
        branch_coverage=CoverageDetail(visited=0, total=0, progress=1.0),
        visited_statement_keys=statements,
        target_reached=reached,
    )


class FakeClient:
    def __init__(
        self,
        coverage_model: CoverageModel,
        symbolic: list[SymbolicAttemptResponse],
        executions: list[ExecutionResult | Exception],
    ) -> None:
        self.coverage_model = coverage_model
        self.symbolic = list(symbolic)
        self.executions = list(executions)
        self.execute_calls = 0
        self.symbolic_calls = 0

    def get_coverage_model(self, _path: str) -> CoverageModel:
        return self.coverage_model

    def get_function_context(self, _path: str):
        return SimpleNamespace(context="")

    def get_node_source(self, _path: str):
        return SimpleNamespace(source=self.coverage_model.source)

    def create_symbolic_attempt(self, request):
        self.symbolic_calls += 1
        response = self.symbolic[min(self.symbolic_calls - 1, len(self.symbolic) - 1)]
        if response.intent is not None:
            response = response.model_copy(deep=True)
            response.intent.target = request.target
        return response

    def execute_intent(self, _intent: TestIntent, test_name: str | None = None):
        self.execute_calls += 1
        value = self.executions.pop(0)
        if isinstance(value, Exception):
            raise value
        return value.model_copy(update={"test_name": test_name or value.test_name})


class FakeCompleter:
    def __init__(self) -> None:
        self.interactions: list[dict] = []
        self.intent_ids: list[str] = []
        self.previous_results: list[ExecutionResult | None] = []

    def complete(
        self,
        intent: TestIntent,
        *,
        function_source: str,
        function_context: str,
        previous_execution: ExecutionResult | None = None,
    ):
        self.intent_ids.append(intent.intent_id)
        self.previous_results.append(previous_execution)
        completed = intent.model_copy(deep=True)
        completed.actions = [
            TestAction(id="invoke", kind=ActionKind.INVOKE, code="f(1);")
        ]
        return completed, CompletionUsage(input_tokens=100, output_tokens=20)


class TargetAwareClient(FakeClient):
    def __init__(self) -> None:
        super().__init__(
            two_statement_model(),
            [symbolic_response(SolverStatus.PARTIAL, executable=False)],
            [],
        )
        self.executed_targets: list[str] = []

    def execute_intent(self, intent: TestIntent, test_name: str | None = None):
        self.execute_calls += 1
        self.executed_targets.append(intent.target.key)
        if self.execute_calls <= 3:
            return ExecutionResult(
                test_name=test_name or "failed",
                status="RUNTIME_ERROR",
                target_reached=False,
            )
        return ExecutionResult(
            test_name=test_name or "passed",
            status="PASSED",
            statement_coverage=CoverageDetail(visited=1, total=2, progress=0.5),
            branch_coverage=CoverageDetail(visited=0, total=0, progress=1.0),
            visited_statement_keys=[intent.target.key],
            target_reached=True,
        )


class FailingCompleter(FakeCompleter):
    def complete(self, *args, **kwargs):
        raise OSError("provider temporarily unavailable")


class UsageFailingCompleter(FakeCompleter):
    def complete(self, *args, **kwargs):
        raise IntentCompletionError(
            "invalid structured response",
            CompletionUsage(
                input_tokens=321,
                output_tokens=45,
                interactions=({"call": 1}, {"call": 2}),
            ),
        )


def symbolic_response(status: SolverStatus, *, executable: bool) -> SymbolicAttemptResponse:
    placeholder = executable_intent(
        target={"kind": "STATEMENT", "key": "s1"}
    )
    if not executable:
        placeholder.actions = []
    return SymbolicAttemptResponse(status=status, intent=placeholder)


def config(strategy: str = "hybrid") -> HybridGenerationConfig:
    return HybridGenerationConfig(
        function_path="D:/src/f.cpp/f(int)",
        strategy=strategy,
        wall_time_minutes=1,
        max_llm_calls=10,
        max_symbolic_attempts=10,
        max_test_executions=10,
    )


def test_symbolic_solved_intent_reaches_coverage_without_llm() -> None:
    client = FakeClient(
        model(),
        [symbolic_response(SolverStatus.SOLVED, executable=True)],
        [execution(reached=True, covered=True)],
    )
    completer = FakeCompleter()

    result = HybridGenerationRunner(client=client, completer=completer).run(
        config("symbolic")
    )

    assert result.stop_reason == "coverage_target"
    assert result.state.statement_fraction == 1.0
    assert client.execute_calls == 1
    assert completer.intent_ids == []


def test_symbolic_route_never_executes_unsupported_intent() -> None:
    client = FakeClient(
        model(),
        [symbolic_response(SolverStatus.UNSUPPORTED, executable=True)],
        [],
    )

    result = HybridGenerationRunner(
        client=client, completer=FakeCompleter()
    ).run(config("symbolic"))

    assert result.stop_reason == "routes_exhausted"
    assert client.execute_calls == 0
    assert len(result.attempts) == 1
    assert all("requires SOLVED" in (attempt.error or "") for attempt in result.attempts)


def test_hybrid_completes_partial_intent_and_records_tokens() -> None:
    client = FakeClient(
        model(complex_cpp=True),
        [symbolic_response(SolverStatus.PARTIAL, executable=False)],
        [execution(reached=True, covered=True)],
    )
    completer = FakeCompleter()

    result = HybridGenerationRunner(client=client, completer=completer).run(config())
    summary = result.to_summary_dict()

    assert result.stop_reason == "coverage_target"
    assert summary["metrics"]["total_input_tokens"] == 100
    assert summary["metrics"]["total_output_tokens"] == 20
    assert 0.0 <= summary["metrics"]["statement_coverage_time_auc"] <= 1.0


def test_hybrid_does_not_execute_partial_intent_without_llm_review() -> None:
    client = FakeClient(
        model(complex_cpp=True),
        [symbolic_response(SolverStatus.PARTIAL, executable=True)],
        [execution(reached=True, covered=True)],
    )
    completer = FakeCompleter()

    result = HybridGenerationRunner(client=client, completer=completer).run(config())

    assert result.stop_reason == "coverage_target"
    assert len(completer.intent_ids) == 1
    assert result.to_summary_dict()["metrics"]["total_input_tokens"] == 100


def test_hybrid_repairs_same_intent_after_target_miss() -> None:
    client = FakeClient(
        model(complex_cpp=True),
        [symbolic_response(SolverStatus.PARTIAL, executable=False)],
        [
            execution(reached=False, covered=False),
            execution(reached=True, covered=True),
        ],
    )
    completer = FakeCompleter()

    result = HybridGenerationRunner(client=client, completer=completer).run(config())

    assert result.stop_reason == "coverage_target"
    assert len(completer.intent_ids) == 2
    assert completer.intent_ids[0] == completer.intent_ids[1]
    assert completer.previous_results[0] is None
    assert completer.previous_results[1] is not None


def test_hybrid_defers_stuck_target_then_reopens_it_after_progress() -> None:
    client = TargetAwareClient()
    run_config = config()
    run_config.max_llm_calls = 5

    result = HybridGenerationRunner(
        client=client, completer=FakeCompleter()
    ).run(run_config)

    assert result.stop_reason == "coverage_target"
    assert client.executed_targets == ["s1", "s1", "s1", "s2", "s1"]


def test_hybrid_stops_after_three_symbolic_misses_when_llm_budget_is_exhausted() -> None:
    client = FakeClient(
        many_statement_model(),
        [symbolic_response(SolverStatus.PARTIAL, executable=False)],
        [],
    )
    run_config = config()
    run_config.max_llm_calls = 0

    result = HybridGenerationRunner(
        client=client, completer=FakeCompleter()
    ).run(run_config)

    assert result.stop_reason == "routes_exhausted"
    assert client.symbolic_calls == 3
    assert client.execute_calls == 0
    assert len(result.attempts) == 3


def test_runner_targets_statements_once_branch_target_is_already_met() -> None:
    client = FakeClient(
        branch_and_statement_model(),
        [],
        [execution(reached=True, covered=True)],
    )
    run_config = config("llm")
    run_config.branch_target = 0.0

    result = HybridGenerationRunner(
        client=client, completer=FakeCompleter()
    ).run(run_config)

    assert result.stop_reason == "coverage_target"
    assert result.attempts[0].target.kind == TargetKind.STATEMENT


def test_no_divergence_repair_ablation_removes_intent_reuse_and_feedback() -> None:
    client = FakeClient(
        model(complex_cpp=True),
        [symbolic_response(SolverStatus.PARTIAL, executable=False)],
        [
            execution(reached=False, covered=False),
            execution(reached=True, covered=True),
        ],
    )
    completer = FakeCompleter()
    run_config = config()
    run_config.use_divergence_repair = False

    result = HybridGenerationRunner(client=client, completer=completer).run(run_config)

    assert result.stop_reason == "coverage_target"
    assert completer.previous_results == [None, None]


def test_akaut_transport_error_stops_as_infra_error() -> None:
    client = FakeClient(
        model(),
        [symbolic_response(SolverStatus.SOLVED, executable=True)],
        [AkaUTError("connection lost")],
    )

    result = HybridGenerationRunner(
        client=client, completer=FakeCompleter()
    ).run(config("symbolic"))

    assert result.stop_reason == "infra_error"
    assert result.error == "connection lost"
    assert client.execute_calls == 1


def test_invalid_intent_http_400_is_route_failure_not_infra_error() -> None:
    client = FakeClient(
        model(),
        [],
        [AkaUTError("invalid TestIntent", status_code=400)] * 3,
    )
    result = HybridGenerationRunner(
        client=client, completer=FakeCompleter()
    ).run(config("llm"))

    assert result.stop_reason == "routes_exhausted"
    assert result.error is None
    assert len(result.attempts) == 3
    assert all("invalid TestIntent" in (item.error or "") for item in result.attempts)


def test_llm_provider_failure_is_recorded_as_route_failure_not_infra_error() -> None:
    client = FakeClient(model(), [], [])
    result = HybridGenerationRunner(
        client=client, completer=FailingCompleter()
    ).run(config("llm"))

    assert result.stop_reason == "routes_exhausted"
    assert result.error is None
    assert len(result.attempts) == 3
    assert all("provider temporarily unavailable" in (item.error or "") for item in result.attempts)


def test_initial_exact_coverage_can_complete_a_sequential_phase_without_calls() -> None:
    client = FakeClient(model(), [], [])
    result = HybridGenerationRunner(client=client, completer=FakeCompleter()).run(
        config("llm"),
        initial_statement_keys={"s1", "not-in-model"},
    )
    assert result.stop_reason == "coverage_target"
    assert result.state.covered_statement_keys == {"s1"}
    assert client.execute_calls == 0


def test_redundancy_rate_uses_executed_tests_not_failed_route_attempts() -> None:
    client = FakeClient(
        model(),
        [
            symbolic_response(SolverStatus.TIMEOUT, executable=False),
            symbolic_response(SolverStatus.SOLVED, executable=True),
        ],
        [execution(reached=True, covered=True)],
    )
    result = HybridGenerationRunner(client=client, completer=FakeCompleter()).run(
        config("symbolic")
    )
    metrics = result.to_summary_dict()["metrics"]
    assert metrics["num_testcases"] == 1
    assert metrics["redundancy_rate"] == 0.0


def test_failed_structured_completion_preserves_observed_token_usage() -> None:
    client = FakeClient(model(), [], [])
    result = HybridGenerationRunner(
        client=client, completer=UsageFailingCompleter()
    ).run(config("llm"))

    metrics = result.to_summary_dict()["metrics"]
    assert metrics["total_input_tokens"] == 321 * 3
    assert metrics["total_output_tokens"] == 45 * 3
    assert metrics["llm_calls"] == 6
    assert all(item.execution is None for item in result.attempts)
