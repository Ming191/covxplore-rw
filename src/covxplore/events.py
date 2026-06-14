from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any


EventSink = Callable[[str, dict[str, Any]], None]
CancelChecker = Callable[[], bool]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def emit_event(
    sink: EventSink | None,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if sink is None:
        return
    sink(event_type, payload or {})


def to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def keys_to_camel(value: Any) -> Any:
    if isinstance(value, list):
        return [keys_to_camel(item) for item in value]
    if isinstance(value, dict):
        return {to_camel(str(k)): keys_to_camel(v) for k, v in value.items()}
    return value
