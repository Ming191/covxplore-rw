from __future__ import annotations

import base64
import os
import re
import warnings
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx

from covxplore.config import get_settings
from covxplore.generation.tokens import TokenTotals

_client: Any | None = None
_initialized = False
_warned = False


def init_observability() -> None:
    """Initialize optional Langfuse/OpenLIT tracing once per process."""
    global _client, _initialized, _warned

    cfg = get_settings()
    if not cfg.langfuse_enabled or _initialized:
        return

    if not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        _warn_once("Langfuse is enabled but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing; tracing disabled.")
        _initialized = True
        return

    os.environ["LANGFUSE_HOST"] = cfg.langfuse_host
    os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
    auth = base64.b64encode(
        f"{cfg.langfuse_public_key}:{cfg.langfuse_secret_key}".encode()
    ).decode()
    otlp_endpoint = f"{cfg.langfuse_host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"Authorization=Basic {auth},x-langfuse-ingestion-version=4"
    )

    try:
        import openlit
        from langfuse import get_client

        _client = get_client()
        openlit.init(
            otlp_endpoint=otlp_endpoint,
            disable_metrics=True,
            disable_events=True,
            disabled_instrumentors=["transformers"],
        )
        _initialized = True
    except Exception as exc:
        _warn_once(f"Langfuse tracing initialization failed: {exc}")
        _client = None
        _initialized = True


@contextmanager
def trace_observation(name: str, **metadata: Any) -> Iterator[str | None]:
    """Create a Langfuse root observation when tracing is available."""
    if _client is None:
        yield None
        return
    try:
        observation = _client.start_as_current_observation(name=name, metadata=metadata)
    except Exception as exc:
        _warn_once(f"Langfuse trace observation failed: {exc}")
        yield None
        return
    with observation as span:
        tracing_url = None
        try:
            tracing_url = _client.get_trace_url(trace_id=getattr(span, "trace_id", None))
        except Exception:
            pass
        yield tracing_url


def get_trace_url() -> str | None:
    """Return the current Langfuse trace URL when available."""
    if _client is None:
        return None
    try:
        return _client.get_trace_url()
    except Exception:
        return None


def extract_trace_id(tracing_url: str | None) -> str | None:
    """Extract a Langfuse trace id from a trace URL."""
    if not tracing_url:
        return None
    try:
        segments = [part for part in urlparse(tracing_url).path.split("/") if part]
        if "traces" in segments:
            candidate = segments[segments.index("traces") + 1]
            if _is_trace_id(candidate):
                return candidate
    except Exception:
        pass
    match = re.search(r"[0-9a-f]{32}", tracing_url)
    return match.group(0) if match else None


def fetch_trace_token_totals(trace_id: str | None) -> TokenTotals:
    """Fetch aggregate input/output tokens from Langfuse observations."""
    if not trace_id:
        return TokenTotals()
    cfg = get_settings()
    if not cfg.langfuse_enabled or not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        return TokenTotals()
    try:
        response = httpx.get(
            f"{cfg.langfuse_host.rstrip('/')}/api/public/traces/{trace_id}",
            auth=(cfg.langfuse_public_key, cfg.langfuse_secret_key),
            timeout=cfg.request_timeout_sec,
        )
        response.raise_for_status()
        return token_totals_from_trace(response.json())
    except Exception as exc:
        _warn_once(f"Langfuse token fetch failed: {exc}")
        return TokenTotals()


def token_totals_from_trace(trace: dict[str, Any]) -> TokenTotals:
    """Sum token usage from Langfuse trace observations."""
    total = TokenTotals()
    for observation in trace.get("observations") or []:
        prompt, completion = _observation_tokens(observation)
        total.prompt += prompt
        total.completion += completion
    return total


def _observation_tokens(observation: dict[str, Any]) -> tuple[int, int]:
    usage = observation.get("usage") or {}
    usage_details = observation.get("usageDetails") or {}
    prompt = _int_first(
        observation.get("inputUsage"),
        usage.get("input"),
        usage.get("promptTokens"),
        usage_details.get("input"),
        usage_details.get("promptTokens"),
    )
    completion = _int_first(
        observation.get("outputUsage"),
        usage.get("output"),
        usage.get("completionTokens"),
        usage_details.get("output"),
        usage_details.get("completionTokens"),
    )
    return prompt, completion


def _int_first(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _is_trace_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", value))


def flush_observability() -> None:
    """Flush pending Langfuse traces without making generation depend on tracing."""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as exc:
        _warn_once(f"Langfuse trace flush failed: {exc}")


def _warn_once(message: str) -> None:
    global _warned
    if _warned:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    _warned = True
