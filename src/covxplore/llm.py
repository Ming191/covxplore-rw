from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

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
    _provider_name: str
    _empty_retries: int
    _retry_backoff_sec: float

    def __init__(
        self,
        *args: Any,
        provider_name: str = "LLM",
        empty_retries: int = 5,
        retry_backoff_sec: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_provider_name", provider_name or "LLM")
        object.__setattr__(self, "_empty_retries", max(0, int(empty_retries)))
        object.__setattr__(self, "_retry_backoff_sec", max(0.0, float(retry_backoff_sec)))

    def call(self, *args: Any, **kwargs: Any) -> Any:
        # These are RetryingLLM-only controls. CrewAI may echo custom LLM
        # attributes into call kwargs; do not let them leak into LiteLLM/OpenAI.
        kwargs.pop("empty_retries", None)
        kwargs.pop("retry_backoff_sec", None)
        kwargs.pop("provider_name", None)

        attempts = self._empty_retries + 1
        result: Any = None
        for attempt in range(1, attempts + 1):
            try:
                result = super().call(*args, **kwargs)
            except Exception as exc:
                exc_name = type(exc).__name__
                is_unavailable = (
                    "ServiceUnavailable" in exc_name
                    or "503" in str(exc)
                    or "service_unavailable" in str(exc).lower()
                )
                if is_unavailable and attempt < attempts:
                    backoff = max(30.0, self._retry_backoff_sec * attempt * 10)
                    _console.print(
                        f"[yellow]{self._provider_name} 503 unavailable "
                        f"(attempt {attempt}/{attempts}); retrying in {backoff:.0f}s[/]"
                    )
                    time.sleep(backoff)
                    continue
                raise
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



@dataclass(frozen=True)
class LLMConfig:
    provider_name: str
    model: str
    api_key: str
    base_url: str


class LLMProvider(ABC):
    """Factory abstraction for provider-specific CrewAI LLM configuration."""

    name: str

    @abstractmethod
    def resolve(self, cfg: Any, model: str | None = None) -> LLMConfig:
        """Return provider-specific LLM construction values."""

    def build(self, cfg: Any, model: str | None = None) -> RetryingLLM:
        llm_cfg = self.resolve(cfg, model)
        return cast(RetryingLLM, RetryingLLM(
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.llm_temperature,
            provider_name=llm_cfg.provider_name,
            empty_retries=cfg.llm_empty_retries,
            retry_backoff_sec=cfg.llm_retry_backoff_sec,
        ))


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def resolve(self, cfg: Any, model: str | None = None) -> LLMConfig:
        resolved = (model or cfg.deepseek_model).strip()
        resolved = resolved if "/" in resolved else f"deepseek/{resolved}"
        return LLMConfig(
            provider_name="DeepSeek",
            model=resolved,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
        )


class KimchiProvider(LLMProvider):
    name = "kimchi"

    def resolve(self, cfg: Any, model: str | None = None) -> LLMConfig:
        resolved = (model or cfg.kimchi_model).strip()
        # Kimchi exposes an OpenAI-compatible endpoint, so use LiteLLM's OpenAI provider.
        resolved = resolved if "/" in resolved else f"openai/{resolved}"
        return LLMConfig(
            provider_name="Kimchi",
            model=resolved,
            api_key=cfg.kimchi_api_key,
            base_url=cfg.kimchi_base_url,
        )


_PROVIDERS: dict[str, LLMProvider] = {
    DeepSeekProvider.name: DeepSeekProvider(),
    KimchiProvider.name: KimchiProvider(),
}


def get_llm_provider(name: str | None) -> LLMProvider:
    provider_name = (name or "deepseek").strip().lower()
    try:
        return _PROVIDERS[provider_name]
    except KeyError as exc:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unsupported LLM provider '{name}'. Available providers: {available}") from exc


def build_llm(model: str | None = None) -> RetryingLLM:
    cfg = get_settings()
    return get_llm_provider(cfg.llm_provider).build(cfg, model)
