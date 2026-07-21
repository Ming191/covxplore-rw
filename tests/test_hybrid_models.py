from __future__ import annotations

import json
from pathlib import Path

import pytest

from covxplore.hybrid.models import (
    ActionKind,
    CoverageModel,
    SourceRange,
    Statement,
    Target,
    TargetKind,
    TestAction,
    TestBinding,
    TestIntent,
    FunctionFeatures,
    SolverStatus,
    SymbolicAttemptResponse,
)


def test_canonical_cross_repository_contract_fixture() -> None:
    fixture = (
        Path(__file__).parents[1] / "contracts" / "cov127-v1-test-intent.json"
    )
    intent = TestIntent.model_validate(json.loads(fixture.read_text(encoding="utf-8")))
    assert intent.version == "1.0"
    assert intent.is_executable
    assert intent.bindings[0].locked
    assert intent.wire_dict()["target"]["desiredOutcome"] is True


def test_static_constraint_feature_uses_camel_case() -> None:
    assert FunctionFeatures(constraint_count=3).wire_dict()["constraintCount"] == 3


def test_contract_uses_camel_case_on_wire() -> None:
    model = CoverageModel(
        function_path="D:/src/a.cpp/F(int)",
        function_name="F(int)",
        model_hash="abc",
        statements=[
            Statement(
                key="stmt:a:1:2",
                text="return 1;",
                source_range=SourceRange(
                    file="D:/src/a.cpp", line_in_function=1, start_offset=1, end_offset=2
                ),
            )
        ],
    )

    wire = model.wire_dict()
    assert wire["version"] == "1.0"
    assert wire["functionPath"] == "D:/src/a.cpp/F(int)"
    assert wire["statements"][0]["sourceRange"]["startOffset"] == 1
    assert CoverageModel.model_validate(wire) == model


def test_executable_intent_requires_exactly_one_invoke() -> None:
    intent = TestIntent(
        intent_id="i1",
        function_path="f",
        target=Target(kind=TargetKind.STATEMENT, key="stmt:1"),
        actions=[TestAction(id="a", kind=ActionKind.DECLARE, code="int x = 0;")],
    )
    with pytest.raises(ValueError, match="exactly one INVOKE"):
        intent.validate_executable()

    intent.actions.append(
        TestAction(id="invoke", kind=ActionKind.INVOKE, code="(void)f(x);")
    )
    intent.validate_executable()


def test_intent_rejects_unknown_action_dependency() -> None:
    with pytest.raises(ValueError, match="unknown actions"):
        TestIntent(
            intent_id="i1",
            function_path="f",
            target=Target(kind=TargetKind.STATEMENT, key="stmt:1"),
            actions=[
                TestAction(
                    id="invoke",
                    kind=ActionKind.INVOKE,
                    code="f();",
                    depends_on=["missing"],
                )
            ],
        )


def test_intent_rejects_action_dependency_cycle() -> None:
    with pytest.raises(ValueError, match="dependency graph contains a cycle"):
        TestIntent(
            intent_id="cycle",
            function_path="f",
            target=Target(kind=TargetKind.STATEMENT, key="stmt:1"),
            actions=[
                TestAction(
                    id="setup",
                    kind=ActionKind.RAW_CPP,
                    code="int x = 0;",
                    depends_on=["invoke"],
                ),
                TestAction(
                    id="invoke",
                    kind=ActionKind.INVOKE,
                    code="f();",
                    depends_on=["setup"],
                ),
            ],
        )


def test_locked_binding_round_trip() -> None:
    binding = TestBinding(
        cpp_lvalue="x", cpp_type="int", value="4", origin="SYMBOLIC", locked=True
    )
    assert TestBinding.model_validate(binding.wire_dict()) == binding


def test_intent_rejects_duplicate_binding_lvalues() -> None:
    binding = TestBinding(
        cpp_lvalue="x", cpp_type="int", value="4", origin="SYMBOLIC", locked=True
    )
    with pytest.raises(ValueError, match="cppLvalue values must be unique"):
        TestIntent(
            intent_id="duplicates",
            function_path="f",
            target=Target(kind=TargetKind.STATEMENT, key="stmt:1"),
            bindings=[binding, binding.model_copy()],
        )


def test_symbolic_solver_telemetry_round_trip_uses_camel_case() -> None:
    response = SymbolicAttemptResponse(
        status=SolverStatus.PARTIAL,
        solver_status=SolverStatus.SOLVED,
        solver_calls=2,
        solver_status_counts={"UNSAT": 1, "SOLVED": 1},
        path_candidate_count=3,
        path_planning_elapsed_ms=1.5,
        constraint_extraction_elapsed_ms=2.5,
        solver_elapsed_ms=30.0,
        model_binding_count=2,
    )

    wire = response.wire_dict()

    assert wire["solverStatus"] == "SOLVED"
    assert wire["solverCalls"] == 2
    assert wire["solverStatusCounts"] == {"UNSAT": 1, "SOLVED": 1}
    assert wire["solverElapsedMs"] == 30.0
    assert SymbolicAttemptResponse.model_validate(wire) == response
