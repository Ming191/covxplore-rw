from __future__ import annotations

import pytest

from covxplore.hybrid.llm_completion import (
    IntentCompleter,
    IntentCompletionError,
    IntentPatch,
)
from covxplore.hybrid.models import (
    ActionKind,
    ExecutionResult,
    Target,
    TargetKind,
    TestAction,
    TestBinding,
    TestIntent,
    TraceStep,
)


def intent_with_locked_binding(value: str = "7") -> TestIntent:
    return TestIntent(
        intent_id="intent-1",
        function_path="D:/f.cpp/f(int)",
        target=Target(kind=TargetKind.STATEMENT, key="s1"),
        bindings=[
            TestBinding(
                cpp_lvalue="x",
                cpp_type="int",
                value=value,
                origin="SYMBOLIC",
                locked=True,
            )
        ],
    )


def test_patch_cannot_change_locked_symbolic_binding() -> None:
    original = intent_with_locked_binding("7")
    patch = IntentPatch(
        actions=[TestAction(id="invoke", kind=ActionKind.INVOKE, code="f(8);")],
        bindings=[
            TestBinding(
                cpp_lvalue="x",
                cpp_type="int",
                value="8",
                origin="LLM",
                locked=False,
            )
        ],
    )

    completed = IntentCompleter._merge_patch(original, patch)

    assert completed.bindings == original.bindings


def test_patch_preserves_protected_test_intent_metadata() -> None:
    original = intent_with_locked_binding()
    patch = IntentPatch(
        actions=[TestAction(id="invoke", kind=ActionKind.INVOKE, code="f(7);")]
    )

    completed = IntentCompleter._merge_patch(original, patch)

    assert completed.intent_id == original.intent_id
    assert completed.function_path == original.function_path
    assert completed.target == original.target


def test_llm_cannot_create_a_locked_or_symbolic_binding() -> None:
    original = intent_with_locked_binding()
    patch = IntentPatch(
        actions=[TestAction(id="invoke", kind=ActionKind.INVOKE, code="f(7);")],
        bindings=[
            TestBinding(
                cpp_lvalue="state.value",
                cpp_type="int",
                value="9",
                origin="SYMBOLIC",
                locked=True,
            )
        ],
    )

    completed = IntentCompleter._merge_patch(original, patch)
    llm_binding = next(
        binding for binding in completed.bindings if binding.cpp_lvalue == "state.value"
    )

    assert llm_binding.origin == "LLM"
    assert llm_binding.locked is False


@pytest.mark.parametrize(
    "cpp_lvalue,value",
    [
        ("p[0].vState.empty()", "false"),
        ("p[0].vState", ""),
    ],
)
def test_invalid_llm_binding_shape_is_rejected_locally(
    cpp_lvalue: str, value: str
) -> None:
    with pytest.raises(ValueError):
        TestBinding(
            cpp_lvalue=cpp_lvalue,
            cpp_type=None,
            value=value,
            origin="LLM",
            locked=True,
        )


def test_raw_completion_salvages_actions_and_drops_only_invalid_bindings() -> None:
    raw = {
        "actions": [
            {
                "id": "invoke",
                "kind": "INVOKE",
                "code": "f(7);",
            }
        ],
        "bindings": [
            {
                "cppLvalue": "p[0].vState.empty()",
                "value": "false",
                "origin": "LLM",
                "locked": True,
            }
        ],
    }

    patch = IntentCompleter._coerce_patch(raw)

    assert patch.actions[0].kind == ActionKind.INVOKE
    assert patch.bindings == []
    assert "LLM_DROPPED_INVALID_BINDINGS:1" in patch.provenance


def test_invalid_structured_response_keeps_provider_usage(monkeypatch) -> None:
    class StubLogger:
        def __init__(self) -> None:
            self.interactions = [
                {
                    "usage": {
                        "prompt_tokens": 123,
                        "completion_tokens": 17,
                    }
                }
            ]

        def attach(self) -> None:
            pass

        def detach(self) -> None:
            pass

    class StubLlm:
        def call(self, *_args, **kwargs):
            assert len(kwargs.get("callbacks", [])) == 1
            return "not-json"

    monkeypatch.setattr(
        "covxplore.hybrid.llm_completion.LLMInteractionLogger", StubLogger
    )
    completer = IntentCompleter(llm=StubLlm())
    with pytest.raises(IntentCompletionError) as captured:
        completer.complete(
            intent_with_locked_binding(),
            function_source="void f(int);",
            function_context="",
        )

    assert captured.value.usage.input_tokens == 123
    assert captured.value.usage.output_tokens == 17
    assert len(completer.interactions) == 1


def test_completion_uses_one_raw_call_with_schema(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class StubLogger:
        def __init__(self) -> None:
            self.interactions = []

        def attach(self) -> None:
            pass

        def detach(self) -> None:
            pass

    class StubLlm:
        def call(self, messages, **kwargs):
            observed["messages"] = messages
            observed["kwargs"] = kwargs
            completed = intent_with_locked_binding().model_copy(deep=True)
            completed.actions = [
                TestAction(id="invoke", kind=ActionKind.INVOKE, code="f(7);")
            ]
            return IntentPatch(
                actions=completed.actions,
                bindings=completed.bindings,
            ).model_dump_json(by_alias=True)

    monkeypatch.setattr(
        "covxplore.hybrid.llm_completion.LLMInteractionLogger", StubLogger
    )
    completed, _usage = IntentCompleter(llm=StubLlm()).complete(
        intent_with_locked_binding(),
        function_source="void f(int);",
        function_context="",
    )

    assert completed.actions[0].kind == ActionKind.INVOKE
    assert list(observed["kwargs"]) == ["callbacks"]
    assert len(observed["kwargs"]["callbacks"]) == 1
    messages = observed["messages"]
    assert isinstance(messages, list)
    assert "REQUIRED JSON SCHEMA" in messages[1]["content"]
    assert "p[0].field" in messages[0]["content"]


def test_runtime_repair_prompt_includes_target_trace_and_previous_body() -> None:
    previous = ExecutionResult(
        test_name="runtime_case",
        status="RUNTIME_ERROR",
        execute_log="missing END marker",
        generated_test_body="Hjson::_parseLoop(p);",
        target_reached=True,
        ordered_trace=[
            TraceStep(
                index=0,
                kind=TargetKind.BRANCH_EDGE,
                key="branch:false",
                outcome="FALSE",
            )
        ],
    )

    messages = IntentCompleter._messages(
        intent_with_locked_binding(),
        function_source="void f(int);",
        function_context="",
        previous_execution=previous,
    )
    repair = messages[1]["content"]

    assert "Status: RUNTIME_ERROR" in repair
    assert "Target reached: True" in repair
    assert '"key": "branch:false"' in repair
    assert "Hjson::_parseLoop(p);" in repair
    assert "Runtime-error traces may contribute coverage" in repair
