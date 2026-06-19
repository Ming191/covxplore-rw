from __future__ import annotations

from dataclasses import dataclass

from covxplore.types.condition_key import ConditionKey
from covxplore.types.coverage_gap_input import CoverageGapInput
from covxplore.types.mcdc_obligation import McdcObligation

from covxplore.coverage._helpers import line_sort_key

@dataclass(frozen=True, slots=True)
class McdcTarget:
    obligation: McdcObligation
    required_value: bool  # True → target TRUE branch

    @property
    def required_label(self) -> str:
        return "TRUE" if self.required_value else "FALSE"

    @property
    def condition_id(self) -> int:
        return self.obligation.condition_id

    @property
    def line_in_function(self) -> int | None:
        return self.obligation.line_in_function

class McdcAnalyzer:
    """Build MC/DC targets and detect stuck obligations."""

    def build_targets(self, ci: CoverageGapInput) -> list[McdcTarget]:
        targets: list[McdcTarget] = []
        for ob in ci.obligations:
            if ob.needs_true:
                targets.append(McdcTarget(ob, True))
            if ob.needs_false:
                targets.append(McdcTarget(ob, False))
        return sorted(targets, key=lambda t: self._target_sort_key(t, ci))

    @staticmethod
    def is_stuck(target: McdcTarget, ci: CoverageGapInput) -> bool:
        if ci.iteration_count < 6:
            return False
        target_key = ConditionKey(target.condition_id, target.required_value)
        opposite_key = ConditionKey(target.condition_id, not target.required_value)
        return (
            ci.observed_polarity_counts.get(target_key, 0) == 0
            and ci.observed_polarity_counts.get(opposite_key, 0) >= 3
        )

    def partition(
        self, targets: list[McdcTarget], ci: CoverageGapInput
    ) -> tuple[list[McdcTarget], list[McdcTarget]]:
        stuck: list[McdcTarget] = []
        reachable: list[McdcTarget] = []
        for t in targets:
            (stuck if self.is_stuck(t, ci) else reachable).append(t)
        return stuck, reachable

    @staticmethod
    def validate(ci: CoverageGapInput) -> None:
        if ci.obligations:
            return
        m = ci.metrics
        remaining = max(m.total_mcdc_pairs - m.covered_mcdc_pairs, 0)
        raise RuntimeError(
            "Unable to compute uncovered conditions while MC/DC pairs remain. "
            f"remaining={remaining}, all_conditions={ci.all_conditions_count}, "
            f"unique_condition_ids={ci.unique_condition_ids}, "
            f"covered_keys={m.covered_mcdc_pairs}, total_mcdc={m.total_mcdc_pairs}. "
            "This indicates inconsistent condition identity between static and execution data."
        )

    @staticmethod
    def check_targets_not_empty(
            targets: list[McdcTarget], ci: CoverageGapInput
    ) -> None:
        if targets:
            return
        m = ci.metrics
        remaining = max(m.total_mcdc_pairs - m.covered_mcdc_pairs, 0)
        raise RuntimeError(
            "MC/DC pairs remain, but no obligation requires TRUE or FALSE. "
            f"remaining={remaining}, covered={m.covered_mcdc_pairs}, "
            f"total={m.total_mcdc_pairs}, obligations={len(ci.obligations)}. "
            "This indicates inconsistent MC/DC obligation data."
        )

    def _target_sort_key(self, t: McdcTarget, ci: CoverageGapInput) -> tuple:
        return (
            self.is_stuck(t, ci),
            *line_sort_key(t.line_in_function),
            t.condition_id,
            0 if t.required_value else 1,
        )
