from __future__ import annotations

from covxplore.coverage import prompt_text as pt
from covxplore.coverage._helpers import format_pct, line_sort_key, line_tag
from covxplore.coverage.mcdc_analyzer import McdcAnalyzer, McdcTarget
from covxplore.types.coverage_gap_input import CoverageGapInput
from covxplore.types.mcdc_obligation import McdcObligation
from covxplore.types.unvisited_branch import UnvisitedBranch
from covxplore.types.unvisited_statement import UnvisitedStatement

def _missing_sides(obj: McdcObligation | UnvisitedBranch) -> list[str]:
    missing: list[str] = []
    if not obj.true_visited:
        missing.append("TRUE")
    if not obj.false_visited:
        missing.append("FALSE")
    return missing


def _format_target(t: McdcTarget) -> str:
    return (
        f"  • [node:{t.condition_id} {line_tag(t.line_in_function)}] "
        f"need {t.required_label} for {t.obligation.condition!r}"
    )

class CoverageRenderer:
    """Render every gap scenario into prompt-ready strings."""


    @staticmethod
    def no_tests(total_mcdc_pairs: int, has_mcdc: bool) -> str:
        return (
            pt.NO_TESTS_WITH_MCDC.format(total_mcdc_pairs=total_mcdc_pairs)
            if has_mcdc
            else pt.NO_TESTS_WITHOUT_MCDC
        )

    @staticmethod
    def success(has_mcdc: bool) -> str:
        return pt.SUCCESS_PREFIX + (", MC/DC" if has_mcdc else "") + pt.SUCCESS_SUFFIX

    def mcdc_section(self, ci: CoverageGapInput, mcdc_analyzer: McdcAnalyzer) -> str:
        mcdc_analyzer.validate(ci)
        targets = mcdc_analyzer.build_targets(ci)
        mcdc_analyzer.check_targets_not_empty(targets, ci)
        stuck, reachable = mcdc_analyzer.partition(targets, ci)

        if not reachable and stuck:
            return self._all_stuck(stuck)
        return self._mcdc_targets(ci, stuck)

    def statement_section(self, stmt_pct: float, statements: list[UnvisitedStatement]) -> str:
        items = sorted(statements, key=lambda s: line_sort_key(s.line_in_function))
        return self._render_detail_section(
            label="Statement",
            pct=stmt_pct,
            items=items,
            max_lines=pt.MAX_STATEMENT_LINES,
            item_format=lambda s: f"[{line_tag(s.line_in_function)}] {s.statement!r}",
        )

    def branch_section(self, branch_pct: float, branches: list[UnvisitedBranch]) -> str:
        items = sorted(branches, key=lambda b: line_sort_key(b.line_in_function))
        return self._render_detail_section(
            label="Branch",
            pct=branch_pct,
            items=items,
            max_lines=pt.MAX_BRANCH_LINES,
            item_format=lambda b: (
                f"[{line_tag(b.line_in_function)}] "
                f"{b.condition!r} — missing: {', '.join(_missing_sides(b))}"
            ),
        )

    # -- private: generic detail section -------------------------------
    @staticmethod
    def _render_detail_section(*, label: str, pct: float, items: list, max_lines: int, item_format) -> str:
        pct_str = format_pct(pct)
        label_lower = label.lower()

        if not items:
            return f"{label} coverage: {pct_str}% — uncovered {label_lower} details unavailable from backend payload."

        lines = [f"{label} coverage: {pct_str}% — {len(items)} {label_lower}(s) not yet executed by any test:"]
        for item in items[:max_lines]:
            lines.append(f"  • {item_format(item)}")
        if len(items) > max_lines:
            lines.append(f"  ... and {len(items) - max_lines} more")
        return "\n".join(lines)

    # -- private: MC/DC rendering --------------------------------------

    def _all_stuck(self, stuck: list[McdcTarget]) -> str:
        return "\n".join(self._all_stuck_block(stuck))

    def _mcdc_targets(self, ci: CoverageGapInput, stuck: list[McdcTarget]) -> str:
        lines = [pt.CONDITION_IDENTITY_NOTE, "", pt.UNCOVERED_MCDC_HEADER]

        for ob in ci.obligations:
            missing = _missing_sides(ob)
            if missing:
                lines.append(
                    f"  • [node:{ob.condition_id} {line_tag(ob.line_in_function)}] "
                    f"{ob.condition!r} — missing: {', '.join(missing)}"
                )

        if stuck:
            lines.append(f"\n{pt.STUCK_PARTIAL_HEADER}")
            lines.extend(
                self._stuck_lines(stuck, pt.MAX_PARTIAL_STUCK_LINES)
            )
            lines.append(pt.STUCK_PARTIAL_ADVICE)

        if ci.consecutive_redundant >= 3:
            lines.append(pt.REDUNDANT_WARNING_3)
        elif ci.consecutive_redundant == 2:
            lines.append(pt.REDUNDANT_CAUTION_2)

        lines.append(pt.TARGETING_ADVICE)
        return "\n".join(lines)

    # -- private: stuck-block builder -----------------------------------

    def _all_stuck_block(self, stuck: list[McdcTarget]) -> list[str]:
        return [
            pt.CONDITION_IDENTITY_NOTE,
            "",
            pt.STUCK_ALL_HEADER,
            pt.STUCK_ALL_ADVICE,
            pt.STUCK_REPORT_HEADER,
            *self._stuck_lines(stuck, pt.MAX_STUCK_LINES),
            pt.STUCK_RATIONALE,
        ]

    @staticmethod
    def _stuck_lines(stuck: list[McdcTarget], max_lines: int) -> list[str]:
        lines = [_format_target(t) for t in stuck[:max_lines]]
        if len(stuck) > max_lines:
            lines.append(f"  ... and {len(stuck) - max_lines} more")
        return lines
