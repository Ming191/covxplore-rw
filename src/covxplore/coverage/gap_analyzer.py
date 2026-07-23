from __future__ import annotations

from covxplore.coverage import prompt_text as pt
from covxplore.coverage._helpers import CoverageCompletion
from covxplore.coverage.coverage_renderer import CoverageRenderer
from covxplore.types.coverage_gap import CoverageGap
from covxplore.types.coverage_gap_input import CoverageGapInput


class GapAnalyzer:
    """Analyze structural coverage DTOs and render prompt guidance."""

    def __init__(self) -> None:
        self._render = CoverageRenderer()

    def analyze(self, ci: CoverageGapInput) -> CoverageGap:
        metrics = ci.metrics
        if not ci.tests:
            return self._make_gap(pt.NO_TESTS)
        terminal = self._terminal_gap(ci)
        if terminal is not None:
            return terminal
        done = CoverageCompletion(
            statement=metrics.total_statements == 0 or metrics.covered_statements >= metrics.total_statements,
            branch=metrics.total_branches == 0 or metrics.covered_branches >= metrics.total_branches,
        )
        if done.all_done:
            return self._make_gap(self._render.success(), done)
        sections = []
        if not done.statement:
            sections.append(self._render.statement_section(metrics.statement_pct, ci.cumulative_unvisited_statements))
        if not done.branch:
            sections.append(self._render.branch_section(metrics.branch_pct, ci.cumulative_unvisited_branches))
        return CoverageGap("\n\n".join(sections), done.statement, done.branch)

    def _terminal_gap(self, ci: CoverageGapInput) -> CoverageGap | None:
        match ci.tests[-1].status:
            case "COMPILE_ERROR":
                return self._make_gap(pt.COMPILE_ERROR)
            case "FAILED":
                return self._make_gap(pt.FAILED)
            case "UNKNOWN":
                return self._make_gap(pt.UNKNOWN)
        return None

    @staticmethod
    def _make_gap(text: str, done: CoverageCompletion | None = None) -> CoverageGap:
        done = done or CoverageCompletion(False, False)
        return CoverageGap(text, done.statement, done.branch)
