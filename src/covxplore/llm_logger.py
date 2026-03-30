from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

# ---------------------------------------------------------------------------
# Process-level registry  {worker_thread_id -> LLMInteractionLogger}
# ---------------------------------------------------------------------------

_registry: dict[int, "LLMInteractionLogger"] = {}
_registry_lock = threading.Lock()


def _register(logger: "LLMInteractionLogger", thread_id: int) -> None:
    with _registry_lock:
        _registry[thread_id] = logger


def _unregister(thread_id: int) -> None:
    with _registry_lock:
        _registry.pop(thread_id, None)


def _all_active() -> list["LLMInteractionLogger"]:
    with _registry_lock:
        return list(_registry.values())


# ---------------------------------------------------------------------------
# LiteLLM shim — registered once, dispatches to all active loggers
# ---------------------------------------------------------------------------

class _GlobalCallbackShim(CustomLogger):
    """Registered once in litellm.callbacks; dispatches to all active loggers."""

    def _dispatch(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        for logger in _all_active():
            logger._record(kwargs, response_obj, start_time, end_time)

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._dispatch(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._dispatch(kwargs, response_obj, start_time, end_time)


_SHIM_LOCK = threading.Lock()
_SHIM_REGISTERED = False


def _ensure_shim_registered() -> None:
    global _SHIM_REGISTERED
    with _SHIM_LOCK:
        if not _SHIM_REGISTERED:
            litellm.callbacks = [_GlobalCallbackShim()] + list(litellm.callbacks)
            _SHIM_REGISTERED = True


# ---------------------------------------------------------------------------
# CrewAI @after_llm_call hook — records iteration number per LLM call
# ---------------------------------------------------------------------------

def _install_crewai_hook() -> None:
    """Register @after_llm_call if CrewAI hooks are available (no-op otherwise)."""
    try:
        from crewai.hooks import after_llm_call, LLMCallHookContext  # type: ignore

        @after_llm_call
        def _track_iteration(context: LLMCallHookContext) -> None:
            for logger in _all_active():
                logger._set_last_iteration(getattr(context, "iterations", 0))
            return None  # keep original response

    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public logger
# ---------------------------------------------------------------------------

class LLMInteractionLogger:
    """Collects LLM interactions for one experiment run."""

    def __init__(self) -> None:
        self._interactions: list[dict] = []
        self._call_index: int = 0
        self._last_iteration: int = 0
        self._lock = threading.Lock()
        self._thread_id: int = threading.current_thread().ident or 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Start collecting calls made from this experiment run."""
        _ensure_shim_registered()
        _register(self, self._thread_id)

    def detach(self) -> None:
        """Stop collecting and remove this logger from the registry."""
        _unregister(self._thread_id)

    def _set_last_iteration(self, n: int) -> None:
        with self._lock:
            self._last_iteration = n

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        with self._lock:
            self._call_index += 1
            idx = self._call_index
            iteration = self._last_iteration

        model: str = kwargs.get("model", "unknown")
        messages: list[dict] = kwargs.get("messages", [])

        thinking: str | None = None
        answer: str | None = None
        tool_calls: list[dict] | None = None
        usage: dict = {}

        try:
            choice = response_obj.choices[0]
            # reasoning_content lives on the Choices object (DeepSeek-R1, Gemini, etc.)
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

        entry: dict = {
            "call_index": idx,
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "elapsed_ms": elapsed_ms,
            "usage": usage,
            "messages": messages,
            "thinking": thinking,
            "answer": answer,
            "tool_calls": tool_calls,
        }

        with self._lock:
            self._interactions.append(entry)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def interactions(self) -> list[dict]:
        with self._lock:
            return list(self._interactions)


# Install CrewAI hook at import time
_install_crewai_hook()
