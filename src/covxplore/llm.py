from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from crewai import LLM

from covxplore.config import get_settings
from covxplore.generation.deepseek_thinking import DeepSeekThinkingInterceptor


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    stream: bool = False


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def resolve(self, settings: Any, model: str | None = None) -> LLMConfig: ...

    def build(self, settings: Any, model: str | None = None) -> LLM:
        config = self.resolve(settings, model)
        thinking = self.name == "deepseek" and settings.llm_thinking is True
        return LLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=settings.max_tokens,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_sec,
            max_retries=settings.llm_max_retries,
            seed=settings.llm_seed,
            stream=config.stream,
            additional_params={"extra_body": {"thinking": {"type": "enabled"}}}
            if thinking
            else {},
            interceptor=DeepSeekThinkingInterceptor() if thinking else None,
        )


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def resolve(self, settings: Any, model: str | None = None) -> LLMConfig:
        resolved = (model or settings.deepseek_model).strip()
        return LLMConfig(
            model=resolved if "/" in resolved else f"deepseek/{resolved}",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )


class LocalProvider(LLMProvider):
    name = "local"

    def resolve(self, settings: Any, model: str | None = None) -> LLMConfig:
        resolved = (model or settings.local_model).strip()
        return LLMConfig(
            model=resolved if "/" in resolved else f"openai/{resolved}",
            api_key=settings.local_api_key,
            base_url=settings.local_base_url,
            stream=True,
        )


_PROVIDERS: dict[str, LLMProvider] = {
    "deepseek": DeepSeekProvider(),
    "local": LocalProvider(),
}


def get_llm_provider(name: str | None) -> LLMProvider:
    provider_name = (name or "deepseek").strip().lower()
    try:
        return _PROVIDERS[provider_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LLM provider {name!r}. Available: {', '.join(_PROVIDERS)}"
        ) from exc


def build_llm(model: str | None = None) -> LLM:
    settings = get_settings()
    return get_llm_provider(settings.llm_provider).build(settings, model)
