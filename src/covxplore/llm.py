from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from crewai import LLM

from covxplore.config import get_settings


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

    def build(self, cfg: Any, model: str | None = None) -> LLM:
        llm_cfg = self.resolve(cfg, model)
        return LLM(
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.llm_temperature,
        )


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


def build_llm(model: str | None = None) -> LLM:
    cfg = get_settings()
    return get_llm_provider(cfg.llm_provider).build(cfg, model)
