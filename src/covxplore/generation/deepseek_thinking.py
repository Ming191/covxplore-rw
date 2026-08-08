from __future__ import annotations

import json
from threading import Lock
from typing import Any

import httpx
from crewai.llms.hooks.base import BaseInterceptor


class DeepSeekThinkingInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Capture DeepSeek reasoning_content without altering API responses."""

    def __init__(self) -> None:
        self._responses: dict[str, str] = {}
        self._lock = Lock()

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        if message.status_code < 200 or message.status_code >= 300:
            return message
        try:
            payload = json.loads(message.read())
            response_id = str(payload.get("id") or "")
            choices = payload.get("choices") or []
            reasoning = ((choices[0].get("message") or {}).get("reasoning_content")) if choices else None
            if response_id and reasoning:
                with self._lock:
                    self._responses[response_id] = str(reasoning)
        except (ValueError, TypeError, IndexError, AttributeError):
            pass
        return message

    async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
        return self.on_outbound(message)

    async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
        return self.on_inbound(message)

    def pop(self, response_id: str | None) -> str | None:
        if not response_id:
            return None
        with self._lock:
            return self._responses.pop(response_id, None)
