from __future__ import annotations

import base64
import os
import warnings
from typing import Any

from covxplore.config import get_settings

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
