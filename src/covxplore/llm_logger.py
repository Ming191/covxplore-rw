from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


class LLMInteractionLogger(CustomLogger):
    """Per-run LiteLLM CustomLogger. Call attach() after build_crew(), detach() in finally."""

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

    def detach(self) -> None:
        litellm.callbacks = self._remove_logger(litellm.callbacks, self)
        litellm.success_callback = self._remove_logger(
            getattr(litellm, "success_callback", []), self
        )
        litellm._async_success_callback = self._remove_logger(
            getattr(litellm, "_async_success_callback", []), self
        )

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
            u = response_obj.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            }
        except Exception:
            pass

        try:
            elapsed_ms = round((end_time - start_time).total_seconds() * 1000, 1)
        except Exception:
            elapsed_ms = None

        with self._lock:
            self._interactions.append(
                {
                    "call_index": idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "elapsed_ms": elapsed_ms,
                    "usage": usage,
                    "messages": messages,
                    "thinking": thinking,
                    "answer": answer,
                    "tool_calls": tool_calls,
                }
            )

    @property
    def interactions(self) -> list[dict]:
        with self._lock:
            return list(self._interactions)
