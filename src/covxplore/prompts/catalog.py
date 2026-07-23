"""Load prompt templates from the package YAML catalog without package imports."""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

import yaml

_CATALOG_PATH = pathlib.Path(__file__).with_name("prompts.yaml")


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    data = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid prompt catalog: {_CATALOG_PATH}")
    return data


def catalog_value(group: str, name: str) -> Any:
    try:
        return load_catalog()[group][name]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Prompt template not found: {group}.{name}") from exc


def catalog_text(group: str, name: str) -> str:
    value = catalog_value(group, name)
    if not isinstance(value, str):
        raise TypeError(f"Prompt template must be text: {group}.{name}")
    return value.strip()
