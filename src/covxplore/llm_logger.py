from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    return str(value)


class LLMInteractionLogger(CustomLogger):
    """Per-run LiteLLM CustomLogger. Call attach() after build_crew(), detach() in finally."""

    _provider_patch_lock = threading.Lock()
    _provider_original_emit: Any = None
    _active_provider_loggers: list["LLMInteractionLogger"] = []

    def __init__(self) -> None:
        super().__init__()
        self._interactions: list[dict] = []
        self._call_index: int = 0
        self._lock = threading.Lock()

    @staticmethod
    def _as_list(callbacks: Any) -> list[Any]:
        if callbacks is None:
            return []
        if isinstance(callbacks, list):
            return callbacks
        return [callbacks]

    @staticmethod
    def _prepend_logger(callbacks: Any, logger: "LLMInteractionLogger") -> list[Any]:
        items = LLMInteractionLogger._as_list(callbacks)
        cleaned = [cb for cb in items if not isinstance(cb, LLMInteractionLogger)]
        return [logger] + cleaned

    @staticmethod
    def _remove_logger(callbacks: Any, logger: "LLMInteractionLogger") -> list[Any]:
        items = LLMInteractionLogger._as_list(callbacks)
        return [cb for cb in items if cb is not logger]

    def attach(self) -> None:
        # Legacy callback entrypoint (older LiteLLM versions)
        litellm.callbacks = self._prepend_logger(litellm.callbacks, self)
        # Active success callback entrypoints (current LiteLLM versions)
        litellm.success_callback = self._prepend_logger(
            getattr(litellm, "success_callback", []), self
        )
        litellm._async_success_callback = self._prepend_logger(
            getattr(litellm, "_async_success_callback", []), self
        )
        self._attach_crewai_provider_logger()

    def detach(self) -> None:
        litellm.callbacks = self._remove_logger(litellm.callbacks, self)
        litellm.success_callback = self._remove_logger(
            getattr(litellm, "success_callback", []), self
        )
        litellm._async_success_callback = self._remove_logger(
            getattr(litellm, "_async_success_callback", []), self
        )
        self._detach_crewai_provider_logger()

    def _attach_crewai_provider_logger(self) -> None:
        """Capture usage from CrewAI's native OpenAI-compatible provider.

        Newer CrewAI versions may bypass LiteLLM callbacks entirely and call the
        OpenAI-compatible provider directly. Its completion event already carries
        normalized usage, so patch that event emitter during this run.
        """
        try:
            from crewai.llms.providers.openai_compatible.completion import (
                OpenAICompatibleCompletion,
            )
        except Exception:
            return

        with self._provider_patch_lock:
            if self not in self._active_provider_loggers:
                self._active_provider_loggers.append(self)

            if self._provider_original_emit is not None:
                return

            original_emit = OpenAICompatibleCompletion._emit_call_completed_event
            LLMInteractionLogger._provider_original_emit = original_emit

            def patched_emit(provider_self: Any, *args: Any, **kwargs: Any) -> Any:
                for logger in list(LLMInteractionLogger._active_provider_loggers):
                    logger._record_crewai_provider_event(provider_self, kwargs)
                return original_emit(provider_self, *args, **kwargs)

            OpenAICompatibleCompletion._emit_call_completed_event = patched_emit

    def _detach_crewai_provider_logger(self) -> None:
        try:
            from crewai.llms.providers.openai_compatible.completion import (
                OpenAICompatibleCompletion,
            )
        except Exception:
            return

        with self._provider_patch_lock:
            if self in self._active_provider_loggers:
                self._active_provider_loggers.remove(self)
            if self._active_provider_loggers or self._provider_original_emit is None:
                return
            OpenAICompatibleCompletion._emit_call_completed_event = (
                self._provider_original_emit
            )
            LLMInteractionLogger._provider_original_emit = None

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._record(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._record(kwargs, response_obj, start_time, end_time)

    def _record(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        with self._lock:
            self._call_index += 1
            idx = self._call_index

        model: str = kwargs.get("model", "unknown")
        messages: list[dict] = kwargs.get("messages", [])

        thinking: str | None = None
        answer: str | None = None
        tool_calls: list[dict] | None = None
        usage: dict = {}

        try:
            choice = response_obj.choices[0]
            thinking = getattr(choice, "reasoning_content", None) or None
            answer = getattr(choice.message, "content", None) or ""
            raw_tc = getattr(choice.message, "tool_calls", None)
            if raw_tc:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in raw_tc
                ]
        except Exception:
            pass

        try:
            u = (
                response_obj.get("usage")
                if isinstance(response_obj, dict)
                else response_obj.usage
            )
            usage = {
                "prompt_tokens": (
                    u.get("prompt_tokens", 0)
                    if isinstance(u, dict)
                    else getattr(u, "prompt_tokens", 0)
                ),
                "completion_tokens": (
                    u.get("completion_tokens", 0)
                    if isinstance(u, dict)
                    else getattr(u, "completion_tokens", 0)
                ),
                "total_tokens": (
                    u.get("total_tokens", 0)
                    if isinstance(u, dict)
                    else getattr(u, "total_tokens", 0)
                ),
            }
        except Exception:
            pass

        try:
            elapsed_ms = round((end_time - start_time).total_seconds() * 1000, 1)
        except Exception:
            elapsed_ms = None

        with self._lock:
            self._append_if_new(
                {
                    "call_index": idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "litellm",
                    "model": model,
                    "elapsed_ms": elapsed_ms,
                    "usage": usage,
                    "messages": messages,
                    "thinking": thinking,
                    "answer": answer,
                    "tool_calls": tool_calls,
                }
            )

    def _record_crewai_provider_event(self, provider: Any, event: dict) -> None:
        usage = event.get("usage") or {}
        if not usage:
            return

        with self._lock:
            self._call_index += 1
            idx = self._call_index
            self._append_if_new(
                {
                    "call_index": idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "crewai_provider",
                    "model": getattr(provider, "model", "unknown"),
                    "elapsed_ms": None,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    "messages": _json_safe(event.get("messages") or []),
                    "thinking": None,
                    "answer": _json_safe(event.get("response")),
                    "tool_calls": None,
                }
            )

    def _append_if_new(self, interaction: dict) -> None:
        """Drop only the cross-callback duplicate for one provider completion."""
        for existing in reversed(self._interactions):
            if existing.get("source") == interaction.get("source"):
                continue
            if (
                existing.get("model") == interaction.get("model")
                and existing.get("usage") == interaction.get("usage")
            ):
                return
        self._interactions.append(interaction)

    @property
    def interactions(self) -> list[dict]:
        with self._lock:
            return list(self._interactions)
