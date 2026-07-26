from __future__ import annotations

from covxplore.coverage._helpers import format_pct, node_sort_key, node_tag
from covxplore.prompts.catalog import catalog_text, catalog_value
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.unvisited_statement import UnvisitedStatement


def _missing_sides(obj: UnvisitedBranch) -> list[str]:
    return [
        label
        for label, visited in (("TRUE", obj.true_visited), ("FALSE", obj.false_visited))
        if not visited
    ]


class CoverageRenderer:
    """Render every gap scenario into prompt-ready strings."""

    @staticmethod
    def success() -> str:
        return catalog_text("coverage", "success")

    def statement_section(
        self, stmt_pct: float, statements: list[UnvisitedStatement]
    ) -> str:
        items = sorted(statements, key=lambda s: node_sort_key(s.node_id))
        return self._render_detail_section(
            label="Statement",
            pct=stmt_pct,
            items=items,
            max_lines=int(catalog_value("coverage", "max_statement_lines")),
            item_format=lambda s: f"[{node_tag(s.node_id)}] {s.statement!r}",
        )

    def branch_section(
        self, branch_pct: float, branches: list[UnvisitedBranch]
    ) -> str:
        items = sorted(branches, key=lambda b: node_sort_key(b.node_id))
        return self._render_detail_section(
            label="Branch",
            pct=branch_pct,
            items=items,
            max_lines=int(catalog_value("coverage", "max_branch_lines")),
            item_format=lambda b: (
                f"[{node_tag(b.node_id)}] "
                f"{b.condition!r} — missing: {', '.join(_missing_sides(b))}"
            ),
        )

    @staticmethod
    def _render_detail_section(
        *, label: str, pct: float, items: list, max_lines: int, item_format
    ) -> str:
        values = {
            "label": label,
            "label_lower": label.lower(),
            "pct": format_pct(pct),
            "count": len(items),
        }
        if not items:
            return catalog_text("coverage", "detail_unavailable").format(**values)

        lines = [catalog_text("coverage", "detail_header").format(**values)]
        lines.extend(f"  • {item_format(item)}" for item in items[:max_lines])
        if len(items) > max_lines:
            lines.append(f"  ... and {len(items) - max_lines} more")
        return "\n".join(lines)
