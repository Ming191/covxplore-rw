from __future__ import annotations

import time
from typing import Any

from crewai import LLM
from rich.console import Console

from covxplore.config import get_settings

_console = Console()


def _is_empty(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        return not result.strip()
    return False


class RetryingLLM(LLM):
    def __init__(
        self,
        *args: Any,
        empty_retries: int = 5,
        retry_backoff_sec: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_empty_retries", max(0, int(empty_retries)))
        object.__setattr__(self, "_retry_backoff_sec", max(0.0, float(retry_backoff_sec)))

    def call(self, *args: Any, **kwargs: Any) -> Any:
        attempts = self._empty_retries + 1
        result: Any = None
        for attempt in range(1, attempts + 1):
            result = super().call(*args, **kwargs)
            if not _is_empty(result):
                return result
            if attempt < attempts:
                backoff = self._retry_backoff_sec * attempt
                _console.print(
                    f"[yellow]LLM returned empty completion "
                    f"(attempt {attempt}/{attempts}); retrying in {backoff:.1f}s[/]"
                )
                if backoff > 0:
                    time.sleep(backoff)
        _console.print(
            f"[red]LLM still empty after {attempts} attempts; "
            f"deferring to CrewAI retry.[/]"
        )
        return result


def build_llm(model: str | None = None) -> RetryingLLM:
    cfg = get_settings()
    resolved = (model or cfg.deepseek_model).strip()
    resolved = resolved if "/" in resolved else f"deepseek/{resolved}"
    return RetryingLLM(
        model=resolved,
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        max_tokens=cfg.max_tokens,
        empty_retries=cfg.llm_empty_retries,
        retry_backoff_sec=cfg.llm_retry_backoff_sec,
    )
