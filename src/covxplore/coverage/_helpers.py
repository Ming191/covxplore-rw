from __future__ import annotations

from typing import NamedTuple


class CoverageCompletion(NamedTuple):
    statement: bool
    branch: bool

    @property
    def all_done(self) -> bool:
        return self.statement and self.branch


def line_tag(line_in_function: int | None) -> str:
    return f"line+{line_in_function}" if line_in_function is not None else "line+?"


def line_sort_key(line_in_function: int | None) -> tuple[bool, int | float]:
    return (
        line_in_function is None,
        line_in_function if line_in_function is not None else float("inf"),
    )


def node_tag(node_id: int | None) -> str:
    """Stable CFG identity for gap text — never confuse with line_in_function."""
    return f"nodeId={node_id}" if node_id is not None else "nodeId=?"


def node_sort_key(node_id: int | None) -> tuple[bool, int | float]:
    return (
        node_id is None,
        node_id if node_id is not None else float("inf"),
    )


def format_pct(pct: float) -> str:
    return f"{pct * 100:.0f}"
