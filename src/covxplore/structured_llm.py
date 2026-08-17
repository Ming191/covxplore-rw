from __future__ import annotations

from typing import Any

import httpx
from crewai import LLM
from crewai.types.usage_metrics import UsageMetrics
from pydantic import BaseModel


class OpenAICompatibleStructuredLLM:
    """Use native JSON Schema without CrewAI's provider-specific instructor bridge."""

    native_structured_output = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        self._usage = UsageMetrics()

    def call(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel] | None = None,
        **_: Any,
    ) -> BaseModel | str:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if response_model is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            }
        response = self._http.post("chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        self._usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self._usage.completion_tokens += int(usage.get("completion_tokens") or 0)
        self._usage.total_tokens += int(usage.get("total_tokens") or 0)
        self._usage.successful_requests += 1
        content = ((payload.get("choices") or [{}])[0].get("message") or {}).get(
            "content", ""
        )
        return response_model.model_validate_json(content) if response_model else content

    def get_token_usage_summary(self) -> UsageMetrics:
        return self._usage
