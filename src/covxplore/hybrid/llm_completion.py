from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from covxplore.hybrid.models import (
    ContractModel,
    ExecutionResult,
    TestAction,
    TestBinding,
    TestIntent,
)
from covxplore.llm import build_llm
from covxplore.llm_logger import LLMInteractionLogger


@dataclass(frozen=True)
class CompletionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    interactions: tuple[dict[str, Any], ...] = ()


class IntentCompletionError(RuntimeError):
    """A completion failed after any observable provider usage was captured."""

    def __init__(self, message: str, usage: CompletionUsage) -> None:
        super().__init__(message)
        self.usage = usage


class IntentPatch(ContractModel):
    """LLM-owned fields only; protected TestIntent metadata stays deterministic."""

    actions: list[TestAction] = Field(default_factory=list)
    bindings: list[TestBinding] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)


@dataclass
class IntentCompleter:
    """Use an LLM only to complete or minimally repair a TestIntent."""

    llm: Any | None = None
    interactions: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        intent: TestIntent,
        *,
        function_source: str,
        function_context: str,
        previous_execution: ExecutionResult | None = None,
    ) -> tuple[TestIntent, CompletionUsage]:
        # Retries belong to HybridGenerationRunner so every provider call is
        # visible to the scheduler and its hard budget.
        llm = self.llm or build_llm(empty_retries=0)
        logger = LLMInteractionLogger()
        messages = self._messages(
            intent,
            function_source=function_source,
            function_context=function_context,
            previous_execution=previous_execution,
        )
        raw: Any = None
        call_error: Exception | None = None
        call_started = time.monotonic()
        try:
            # CrewAI's response_model path delegates to Instructor, whose
            # internal retries are invisible to the scheduler budget. Keep one
            # raw provider call here and validate the JSON locally instead.
            raw = llm.call(messages, callbacks=[logger])
        except Exception as exc:
            call_error = exc

        calls = logger.interactions
        if calls:
            last = calls[-1]
            if not last.get("answer") and raw is not None:
                last["answer"] = str(raw)
            if last.get("elapsed_ms") is None:
                last["elapsed_ms"] = round(
                    (time.monotonic() - call_started) * 1000, 1
                )
        self.interactions.extend(calls)
        usage = self._usage(calls)
        if call_error is not None:
            raise IntentCompletionError(
                f"{type(call_error).__name__}: {call_error}", usage
            ) from call_error

        try:
            patch = self._coerce_patch(raw)
            completed = self._merge_patch(intent, patch)
            completed.validate_executable()
        except Exception as exc:
            raise IntentCompletionError(
                f"{type(exc).__name__}: {exc}", usage
            ) from exc
        return completed, usage

    @staticmethod
    def _usage(calls: list[dict[str, Any]]) -> CompletionUsage:
        input_tokens = sum(
            int(call.get("usage", {}).get("prompt_tokens", 0) or 0)
            for call in calls
        )
        output_tokens = sum(
            int(call.get("usage", {}).get("completion_tokens", 0) or 0)
            for call in calls
        )
        return CompletionUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            interactions=tuple(calls),
        )

    @staticmethod
    def _messages(
        intent: TestIntent,
        *,
        function_source: str,
        function_context: str,
        previous_execution: ExecutionResult | None,
    ) -> list[dict[str, str]]:
        repair = ""
        if previous_execution is not None:
            divergence = (
                previous_execution.first_divergence.wire_dict()
                if previous_execution.first_divergence
                else None
            )
            trace = [step.wire_dict() for step in previous_execution.ordered_trace]
            repair = (
                "\nPrevious execution failed or missed the target. Repair only the "
                "necessary setup/actions.\n"
                f"Status: {previous_execution.status}\n"
                f"Target reached: {previous_execution.target_reached}\n"
                f"Log: {previous_execution.execute_log[-5000:]}\n"
                f"First divergence: {json.dumps(divergence)}\n"
                f"Ordered focal trace: {json.dumps(trace)}\n"
                "Previous generated body:\n"
                f"{previous_execution.generated_test_body}\n"
                "Runtime-error traces may contribute coverage when AkaUT reports stable "
                "visited keys, but a clean PASSED test is still preferred. If this "
                "target remains uncovered or missed, change the setup/actions so the "
                "focal invocation terminates without crashing, aborting, or timing out. "
                "Do not return the same failing patch.\n"
            )

        system = (
            "You complete a version 1.0 TestIntent for a C++ unit-test driver. "
            "Return only a JSON intent patch matching the supplied schema. Do not repeat or "
            "change functionPath, target, intentId, expectedTrace, or locked bindings. Do not emit "
            "main(), includes, assertions, markdown, or prose. Add exactly one INVOKE "
            "action. Declare or construct any state needed to make every locked cppLvalue "
            "valid; AkaUT reapplies its value immediately before invocation. A locked root "
            "such as p[0].field means p must be declared as a pointer named exactly p and "
            "passed directly, never declared as an object and passed as &p. Use RAW_CPP only when "
            "DECLARE/CONSTRUCT/ASSIGN/CALL/STUB cannot "
            "represent required C++ setup. The body runs inside AkaUT's existing driver. "
            "Prefer actions for all LLM-created setup. Never encode a container, object, "
            "or member-function call such as vector.empty() or vector.back() as a binding. "
            "Any binding you add must be a concrete assignable scalar lvalue; it remains "
            "unlocked and cannot override a solver binding. "
            "When repairing a failed execution, the patch must materially change the "
            "failing setup or invocation and must finish with PASSED status."
        )
        schema = json.dumps(IntentPatch.model_json_schema(), separators=(",", ":"))
        user = (
            f"REQUIRED JSON SCHEMA:\n{schema}\n\n"
            f"TARGET INTENT:\n{json.dumps(intent.wire_dict(), indent=2)}\n\n"
            f"FOCAL FUNCTION SOURCE:\n{function_source}\n\n"
            f"RELEVANT C++ CONTEXT:\n{function_context}\n"
            f"{repair}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _coerce_patch(raw: Any) -> IntentPatch:
        if isinstance(raw, IntentPatch):
            return raw
        if isinstance(raw, dict):
            return IntentCompleter._validate_patch_payload(raw)
        if hasattr(raw, "model_dump"):
            return IntentCompleter._validate_patch_payload(raw.model_dump())
        if not isinstance(raw, str):
            raise ValueError(f"unsupported LLM response type: {type(raw).__name__}")
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        return IntentCompleter._validate_patch_payload(json.loads(text))

    @staticmethod
    def _validate_patch_payload(payload: Any) -> IntentPatch:
        if not isinstance(payload, dict):
            raise ValueError("LLM intent patch must be a JSON object")
        clean = dict(payload)
        valid_bindings: list[dict[str, Any]] = []
        dropped = 0
        for item in clean.get("bindings") or []:
            try:
                binding = TestBinding.model_validate(item)
            except Exception:
                dropped += 1
                continue
            valid_bindings.append(binding.model_dump(by_alias=True))
        clean["bindings"] = valid_bindings
        if dropped:
            provenance = list(clean.get("provenance") or [])
            provenance.append(f"LLM_DROPPED_INVALID_BINDINGS:{dropped}")
            clean["provenance"] = provenance
        return IntentPatch.model_validate(clean)

    @staticmethod
    def _merge_patch(original: TestIntent, patch: IntentPatch) -> TestIntent:
        protected: dict[str, TestBinding] = {
            binding.cpp_lvalue: binding
            for binding in original.bindings
            if binding.locked
        }
        merged_bindings = {
            binding.cpp_lvalue: binding.model_copy(deep=True)
            for binding in original.bindings
        }
        for binding in patch.bindings:
            if binding.cpp_lvalue not in protected:
                llm_binding = binding.model_copy(
                    deep=True,
                    update={"origin": "LLM", "locked": False},
                )
                merged_bindings[binding.cpp_lvalue] = TestBinding.model_validate(
                    llm_binding.model_dump()
                )

        completed = original.model_copy(deep=True)
        completed.actions = [action.model_copy(deep=True) for action in patch.actions]
        completed.bindings = list(merged_bindings.values())
        completed.unresolved_requirements = list(patch.unresolved_requirements)
        completed.provenance = list(
            dict.fromkeys([*original.provenance, *patch.provenance, "LLM"])
        )
        return TestIntent.model_validate(completed.model_dump())
