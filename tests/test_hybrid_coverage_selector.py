from __future__ import annotations

from covxplore.hybrid.coverage import ExactCoverageState
from covxplore.hybrid.models import (
    BranchEdge,
    CoverageDetail,
    CoverageModel,
    ExecutionResult,
    SourceRange,
    Statement,
    TargetKind,
    TraceStep,
)
from covxplore.hybrid.selector import SuccessfulSeed, TargetSelector, longest_common_prefix


def sample_model() -> CoverageModel:
    source_range = SourceRange(file="D:/src/a.cpp", start_offset=0, end_offset=1)
    return CoverageModel(
        function_path="D:/src/a.cpp/f(int)",
        function_name="f(int)",
        model_hash="hash",
        statements=[
            Statement(key="s1", text="if (x)", source_range=source_range, expected_path=["s1"]),
            Statement(key="s2", text="a();", source_range=source_range, path_depth=2, expected_path=["s1", "b1", "s2"]),
            Statement(key="s3", text="b();", source_range=source_range, path_depth=2, expected_path=["s1", "b2", "s3"]),
        ],
        branch_edges=[
            BranchEdge(
                key="b1",
                source_statement_key="s1",
                destination_statement_key="s2",
                outcome="TRUE",
                source_range=source_range,
                downstream_statement_keys=["s2"],
                expected_path=["s1", "b1"],
                path_depth=1,
            ),
            BranchEdge(
                key="b2",
                source_statement_key="s1",
                destination_statement_key="s3",
                outcome="FALSE",
                source_range=source_range,
                downstream_statement_keys=["s2", "s3"],
                expected_path=["s1", "b2"],
                path_depth=1,
            ),
        ],
    )


def execution(*, status: str, statements: list[str], branches: list[str]) -> ExecutionResult:
    return ExecutionResult(
        test_name="tc",
        status=status,
        statement_coverage=CoverageDetail(visited=len(statements), total=3),
        branch_coverage=CoverageDetail(visited=len(branches), total=2),
        visited_statement_keys=statements,
        visited_branch_keys=branches,
        ordered_trace=[
            TraceStep(index=index, kind=TargetKind.STATEMENT, key=key)
            for index, key in enumerate(statements)
        ],
    )


def test_passed_and_runtime_execution_with_trace_contribute_exact_keys() -> None:
    state = ExactCoverageState(sample_model())
    runtime = state.add(execution(status="RUNTIME_ERROR", statements=["s1"], branches=["b1"]))
    assert runtime.accepted
    assert runtime.new_statement_keys == frozenset({"s1"})
    assert runtime.new_branch_keys == frozenset({"b1"})
    assert state.statement_fraction == 1 / 3

    delta = state.add(execution(status="PASSED", statements=["s1", "unknown"], branches=["b1"]))
    assert delta.new_statement_keys == frozenset()
    assert delta.new_branch_keys == frozenset()
    assert state.statement_fraction == 1 / 3

    duplicate = state.add(execution(status="PASSED", statements=["s1"], branches=["b1"]))
    assert duplicate.redundant


def test_compile_error_and_runtime_without_trace_do_not_contribute_coverage() -> None:
    state = ExactCoverageState(sample_model())

    runtime_without_trace = state.add(
        execution(status="RUNTIME_ERROR", statements=[], branches=[])
    )
    compile_error = state.add(
        execution(status="COMPILE_ERROR", statements=["s1"], branches=["b1"])
    )

    assert not runtime_without_trace.accepted
    assert not compile_error.accepted
    assert state.statement_fraction == 0


def test_selector_prioritizes_branch_with_more_uncovered_downstream_statements() -> None:
    state = ExactCoverageState(sample_model())
    selection = TargetSelector().select(state, [])
    assert selection is not None
    assert selection.target.key == "b2"
    assert selection.target.kind == TargetKind.BRANCH_EDGE
    assert selection.target.desired_outcome is False


def test_selector_skips_remaining_branches_after_branch_target_is_met() -> None:
    state = ExactCoverageState(sample_model())
    selection = TargetSelector().select(
        state,
        [],
        include_branches=False,
        include_statements=True,
    )

    assert selection is not None
    assert selection.target.kind == TargetKind.STATEMENT
    assert selection.target.key == "s1"


def test_selector_uses_nearest_successful_trace_seed() -> None:
    state = ExactCoverageState(sample_model())
    near = SuccessfulSeed(
        intent_id="near",
        bindings=(),
        result=execution(status="PASSED", statements=["s1", "b2"], branches=[]),
    )
    far = SuccessfulSeed(
        intent_id="far",
        bindings=(),
        result=execution(status="PASSED", statements=["s3"], branches=[]),
    )
    selection = TargetSelector().select(state, [far, near])
    assert selection is not None and selection.seed is not None
    assert selection.seed.intent_id == "near"
    assert longest_common_prefix(["a", "b"], ["a", "c"]) == 1


def test_seed_prefix_precedes_path_depth_when_branch_gain_is_equal() -> None:
    model = sample_model()
    model.branch_edges[0].downstream_statement_keys = ["s2"]
    model.branch_edges[1].downstream_statement_keys = ["s3"]
    model.branch_edges[0].path_depth = 1
    model.branch_edges[1].path_depth = 5
    state = ExactCoverageState(model)
    near_b2 = SuccessfulSeed(
        intent_id="near-b2",
        bindings=(),
        result=ExecutionResult(
            test_name="seed",
            status="PASSED",
            ordered_trace=[
                TraceStep(index=0, kind=TargetKind.STATEMENT, key="s1"),
                TraceStep(index=1, kind=TargetKind.BRANCH_EDGE, key="b2"),
            ],
        ),
    )

    selection = TargetSelector().select(state, [near_b2])
    assert selection is not None
    assert selection.target.key == "b2"
    assert selection.seed is not None and selection.seed.intent_id == "near-b2"
