from __future__ import annotations

from covxplore.coverage import prompt_text as text
from covxplore.types.condition_key import ConditionKey
from covxplore.types.coverage_gap import CoverageGap
from covxplore.types.coverage_gap_input import CoverageGapInput
from covxplore.types.mcdc_obligation import McdcObligation
from covxplore.status import TestStatus


class GapAnalyzer:
    """Analyze explicit coverage DTOs and render prompt guidance."""

    def analyze(self, coverage_input: CoverageGapInput) -> CoverageGap:  # noqa: C901, PLR0912, PLR0915
        metrics = coverage_input.metrics
        has_mcdc = metrics.total_mcdc_pairs > 0

        if not coverage_input.tests:
            gap_text = (
                text.NO_TESTS_WITH_MCDC.format(total_mcdc_pairs=metrics.total_mcdc_pairs)
                if has_mcdc
                else text.NO_TESTS_WITHOUT_MCDC
            )
            return CoverageGap(gap_text, has_mcdc, False, False, False)

        last = coverage_input.tests[-1]
        if last.status == TestStatus.COMPILE_ERROR.value:
            return CoverageGap(text.COMPILE_ERROR, has_mcdc, False, False, False)
        if last.status == TestStatus.FAILED.value:
            return CoverageGap(text.FAILED, has_mcdc, False, False, False)
        if last.status == TestStatus.UNKNOWN.value:
            return CoverageGap(text.UNKNOWN, has_mcdc, False, False, False)

        mcdc_done = not has_mcdc or metrics.covered_mcdc_pairs >= metrics.total_mcdc_pairs
        cum_stmts = (
            coverage_input.cumulative_unvisited_statements
            if metrics.total_statements == 0
            or len(coverage_input.cumulative_unvisited_statements) <= metrics.total_statements
            else []
        )
        cum_branches = (
            coverage_input.cumulative_unvisited_branches
            if metrics.total_branches == 0
            or len(coverage_input.cumulative_unvisited_branches) <= metrics.total_branches
            else []
        )
        stmt_done = metrics.total_statements == 0 or metrics.covered_statements >= metrics.total_statements
        branch_done = metrics.total_branches == 0 or metrics.covered_branches >= metrics.total_branches

        if mcdc_done and stmt_done and branch_done:
            gap_text = text.SUCCESS_PREFIX + (", MC/DC" if has_mcdc else "") + text.SUCCESS_SUFFIX
            return CoverageGap(gap_text, has_mcdc, mcdc_done, stmt_done, branch_done)

        sections: list[str] = []
        if has_mcdc and not mcdc_done:
            sections.append(self._render_mcdc(coverage_input))
        elif has_mcdc and mcdc_done:
            sections.append(f"MC/DC: 100% covered ({metrics.total_mcdc_pairs}/{metrics.total_mcdc_pairs} pairs).")

        if not stmt_done:
            sections.append(self._render_statements(metrics.statement_pct, cum_stmts))
        if not branch_done:
            sections.append(self._render_branches(metrics.branch_pct, cum_branches))

        return CoverageGap("\n\n".join(sections), has_mcdc, mcdc_done, stmt_done, branch_done)

    def _render_mcdc(self, coverage_input: CoverageGapInput) -> str:
        metrics = coverage_input.metrics
        obligations = coverage_input.obligations
        if not obligations:
            remaining = max(metrics.total_mcdc_pairs - metrics.covered_mcdc_pairs, 0)
            raise RuntimeError(
                "Unable to compute uncovered conditions while MC/DC pairs remain. "
                f"remaining={remaining}, all_conditions={coverage_input.all_conditions_count}, "
                f"unique_condition_ids={coverage_input.unique_condition_ids}, "
                f"covered_keys={metrics.covered_mcdc_pairs}, total_mcdc={metrics.total_mcdc_pairs}. "
                "This indicates inconsistent condition identity between static and execution data."
            )

        target_obligations = self._target_obligations(coverage_input)
        suspected = [item for item in target_obligations if self._is_suspected_stuck(item, coverage_input)]
        non_stuck = [item for item in target_obligations if not self._is_suspected_stuck(item, coverage_input)]

        if not non_stuck and suspected:
            return self._render_all_stuck(suspected)
        return self._render_mcdc_obligations(coverage_input, suspected)

    def _target_obligations(self, coverage_input: CoverageGapInput) -> list[tuple[McdcObligation, bool]]:
        pairs: list[tuple[McdcObligation, bool]] = []
        for obligation in coverage_input.obligations:
            if obligation.needs_true:
                pairs.append((obligation, True))
            if obligation.needs_false:
                pairs.append((obligation, False))
        pairs.sort(
            key=lambda pair: (
                self._is_suspected_stuck(pair, coverage_input),
                pair[0].line_in_function is None,
                pair[0].line_in_function if pair[0].line_in_function is not None else float("inf"),
                pair[0].condition_id,
                0 if pair[1] else 1,
            )
        )
        return pairs

    def _is_suspected_stuck(
        self,
        obligation_pair: tuple[McdcObligation, bool],
        coverage_input: CoverageGapInput,
    ) -> bool:
        obligation, polarity = obligation_pair
        if coverage_input.iteration_count < 6:
            return False
        target_seen = coverage_input.observed_polarity_counts.get(ConditionKey(obligation.condition_id, polarity), 0)
        opposite_seen = coverage_input.observed_polarity_counts.get(ConditionKey(obligation.condition_id, not polarity), 0)
        return target_seen == 0 and opposite_seen >= 3

    def _render_all_stuck(self, suspected: list[tuple[McdcObligation, bool]]) -> str:
        lines = [
            text.CONDITION_IDENTITY_NOTE,
            "",
            text.STUCK_ALL_HEADER,
            text.STUCK_ALL_ADVICE,
            text.STUCK_REPORT_HEADER,
        ]
        for obligation, polarity in suspected[: text.MAX_STUCK_LINES]:
            lines.append(self._format_obligation(obligation, polarity))
        lines.append(text.STUCK_RATIONALE)
        return "\n".join(lines)

    def _render_mcdc_obligations(
        self,
        coverage_input: CoverageGapInput,
        suspected: list[tuple[McdcObligation, bool]],
    ) -> str:
        lines = [text.CONDITION_IDENTITY_NOTE, "", text.UNCOVERED_MCDC_HEADER]
        for obligation in coverage_input.obligations:
            missing = []
            if obligation.needs_true:
                missing.append("TRUE branch")
            if obligation.needs_false:
                missing.append("FALSE branch")
            line_tag = self._line_tag(obligation.line_in_function)
            lines.append(f"  • [node:{obligation.condition_id} {line_tag}] {obligation.condition!r} — missing: {', '.join(missing)}")

        if suspected:
            lines.append(f"\n{text.STUCK_PARTIAL_HEADER}")
            for obligation, polarity in suspected[: text.MAX_PARTIAL_STUCK_LINES]:
                lines.append(self._format_obligation(obligation, polarity))
            lines.append(text.STUCK_PARTIAL_ADVICE)

        if coverage_input.consecutive_redundant >= 3:
            lines.append(text.REDUNDANT_WARNING_3)
        elif coverage_input.consecutive_redundant == 2:
            lines.append(text.REDUNDANT_CAUTION_2)

        lines.append(text.TARGETING_ADVICE)
        return "\n".join(lines)

    def _render_statements(self, statement_pct: float, statements) -> str:
        pct = f"{statement_pct * 100:.0f}"
        if not statements:
            return f"Statement coverage: {pct}% — uncovered statement details unavailable from backend payload."
        lines = [f"Statement coverage: {pct}% — {len(statements)} statement(s) not yet executed by any test:"]
        for statement in sorted(statements, key=lambda x: (x.line_in_function is None, x.line_in_function))[: text.MAX_STATEMENT_LINES]:
            lines.append(f"  • [{self._line_tag(statement.line_in_function)}] {statement.statement!r}")
        if len(statements) > text.MAX_STATEMENT_LINES:
            lines.append(f"  ... and {len(statements) - text.MAX_STATEMENT_LINES} more")
        return "\n".join(lines)

    def _render_branches(self, branch_pct: float, branches) -> str:
        pct = f"{branch_pct * 100:.0f}"
        if not branches:
            return f"Branch coverage: {pct}% — uncovered branch details unavailable from backend payload."
        lines = [f"Branch coverage: {pct}% — {len(branches)} branch node(s) with uncovered side(s) across all tests:"]
        for branch in sorted(branches, key=lambda x: (x.line_in_function is None, x.line_in_function))[: text.MAX_BRANCH_LINES]:
            missing_sides = []
            if not branch.true_visited:
                missing_sides.append("TRUE")
            if not branch.false_visited:
                missing_sides.append("FALSE")
            lines.append(f"  • [{self._line_tag(branch.line_in_function)}] {branch.condition!r} — missing: {', '.join(missing_sides)}")
        if len(branches) > text.MAX_BRANCH_LINES:
            lines.append(f"  ... and {len(branches) - text.MAX_BRANCH_LINES} more")
        return "\n".join(lines)

    def _format_obligation(self, obligation: McdcObligation, polarity: bool) -> str:
        need = "TRUE" if polarity else "FALSE"
        return f"  • [node:{obligation.condition_id} {self._line_tag(obligation.line_in_function)}] need {need} for {obligation.condition!r}"

    def _line_tag(self, line_in_function: int | None) -> str:
        return f"line+{line_in_function}" if line_in_function is not None else "line+?"
