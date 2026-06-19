from __future__ import annotations

from typing import NamedTuple


class CoverageCompletion(NamedTuple):
    mcdc: bool
    statement: bool
    branch: bool

    @property
    def all_done(self) -> bool:
        return self.mcdc and self.statement and self.branch


def line_tag(line_in_function: int | None) -> str:
    return f"line+{line_in_function}" if line_in_function is not None else "line+?"


def line_sort_key(line_in_function: int | None) -> tuple[bool, int | float]:
    return (
        line_in_function is None,
        line_in_function if line_in_function is not None else float("inf"),
    )


def format_pct(pct: float) -> str:
    return f"{pct * 100:.0f}"
