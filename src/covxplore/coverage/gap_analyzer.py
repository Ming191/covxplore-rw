from __future__ import annotations

from covxplore.coverage import prompt_text as pt
from covxplore.coverage._helpers import CoverageCompletion
from covxplore.coverage.coverage_renderer import CoverageRenderer
from covxplore.coverage.mcdc_analyzer import McdcAnalyzer
from covxplore.types.coverage_gap import CoverageGap
from covxplore.types.coverage_gap_input import CoverageGapInput


class GapAnalyzer:
    """Analyze explicit coverage DTOs and render prompt guidance."""

    def __init__(self) -> None:
        self._mcdc = McdcAnalyzer()
        self._render = CoverageRenderer()

    def analyze(self, ci: CoverageGapInput) -> CoverageGap:
        m = ci.metrics
        has_mcdc = m.total_mcdc_pairs > 0

        if not ci.tests:
            return self._make_gap(self._render.no_tests(m.total_mcdc_pairs, has_mcdc), has_mcdc)

        terminal = self._terminal_gap(ci, has_mcdc)
        if terminal is not None:
            return terminal

        done = CoverageCompletion(
            mcdc=m.total_mcdc_pairs == 0 or m.covered_mcdc_pairs >= m.total_mcdc_pairs,
            statement=m.total_statements == 0 or m.covered_statements >= m.total_statements,
            branch=m.total_branches == 0 or m.covered_branches >= m.total_branches,
        )

        if done.all_done:
            return self._make_gap(self._render.success(has_mcdc), has_mcdc, done)

        return CoverageGap(
            "\n\n".join(self._build_sections(ci, done)),
            has_mcdc,
            done.mcdc,
            done.statement,
            done.branch,
        )

    def _terminal_gap(self, ci: CoverageGapInput, has_mcdc: bool) -> CoverageGap | None:
        match ci.tests[-1].status:
            case "COMPILE_ERROR":
                return self._make_gap(pt.COMPILE_ERROR, has_mcdc)
            case "FAILED":
                return self._make_gap(pt.FAILED, has_mcdc)
            case "UNKNOWN":
                return self._make_gap(pt.UNKNOWN, has_mcdc)
        return None

    def _build_sections(self, ci: CoverageGapInput, done: CoverageCompletion) -> list[str]:
        m = ci.metrics
        has_mcdc = m.total_mcdc_pairs > 0
        sections: list[str] = []

        if has_mcdc and not done.mcdc:
            sections.append(self._render.mcdc_section(ci, self._mcdc))
        elif has_mcdc:
            sections.append(f"MC/DC: 100% covered ({m.total_mcdc_pairs}/{m.total_mcdc_pairs} pairs).")

        if not done.statement:
            sections.append(self._render.statement_section(m.statement_pct, ci.cumulative_unvisited_statements))
        if not done.branch:
            sections.append(self._render.branch_section(m.branch_pct, ci.cumulative_unvisited_branches))

        return sections

    @staticmethod
    def _make_gap(
            text: str,
        has_mcdc: bool,
        done: CoverageCompletion | None = None,
    ) -> CoverageGap:
        if done is None:
            done = CoverageCompletion(False, False, False)
        return CoverageGap(text, has_mcdc, done.mcdc, done.statement, done.branch)
