"""Build agent prompts from YAML sections and live coverage state."""

from __future__ import annotations

from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.prompts.catalog import catalog_text, load_catalog
from covxplore.prompts.config import PromptConfig

_STATIC_SECTIONS = ("role_persona", "cot_reasoning", "few_shot_examples", "output_format")


def _workflow_name(config: PromptConfig) -> str:
    if config.unlimited_batch:
        return "unlimited_batch"
    if not config.require_expected_path:
        return "no_path"
    if not config.search_tools:
        return "no_search"
    return "search"


class PromptBuilder:
    """Build system and task prompts from YAML plus runtime values."""

    def __init__(self, config: PromptConfig):
        self.config = config

    def system_prompt(self) -> str:
        sections = list(self.config.enabled_sections())
        # Keep CoT/format for coverage-only wording when path is off; only swap
        # path-specific output_format text via a soft instruction (do not strip sections).
        parts = [
            catalog_text("sections", section)
            for section in _STATIC_SECTIONS
            if section in sections
        ]
        if not self.config.require_expected_path:
            # Soften path requirements that remain inside shared YAML sections.
            parts.append(
                "Focus on covering uncovered statements and branches. "
                "Do not predict expected_path; leave expected_path empty."
            )
        elif not self.config.path_feedback:
            parts.append(
                "Still submit expected_path with each candidate, but path-match "
                "diagnostics will not be shown after execution — rely on coverage gaps."
            )
        return self._join(parts) if parts else ""

    def task_description(
        self,
        function_path: str,
        suite=None,
        remaining_batches: int = 0,
        static_context_text: str | None = None,
        static_source_text: str | None = None,
        static_branch_catalog_text: str | None = None,
        dynamic_knowledge_text: str | None = None,
    ) -> str:
        from covxplore.types import TestSuite

        batch_hint = (
            "Return exactly 1 focused test candidate"
            if self.config.max_batch_candidates <= 1
            else "Return 3-5 focused test candidates"
        )
        parts = [
            catalog_text("task", "description").format(
                function_path=function_path,
                workflow=catalog_text("workflows", _workflow_name(self.config)),
                path_requirement=(
                    ", each with a distinct non-empty expected_path (ordered branch outcomes to the target)"
                    if self.config.require_expected_path
                    else " (omit expected_path)"
                ),
            )
        ]
        # Override the default "3-5" wording when single-candidate ablation is on.
        if self.config.max_batch_candidates <= 1:
            parts[0] = parts[0].replace(
                "Return 3-5 focused test candidates",
                batch_hint,
            )

        if static_context_text:
            parts.append(
                catalog_text("task", "context").format(content=static_context_text)
            )
        if static_source_text:
            parts.append(
                catalog_text("task", "source").format(content=static_source_text)
            )
        if static_branch_catalog_text and self.config.preload_branch_catalog:
            parts.append(
                catalog_text("task", "branch_catalog").format(
                    content=static_branch_catalog_text
                )
            )
        if dynamic_knowledge_text:
            parts.append(
                "PRIOR DISCOVERY FROM EARLIER SESSIONS:\n"
                + dynamic_knowledge_text
                + "\nUse this as established context. Search and source tools return this same "
                "result for an exact repeated lookup."
            )

        if self.config.coverage_guidance and suite is not None:
            assert isinstance(suite, TestSuite)
            metrics = suite.coverage.metrics(suite.tests)
            gap = GapAnalyzer().analyze(
                suite.coverage.gap_input(suite.tests, suite.batch_count)
            ).text
            parts.append(
                catalog_text("sections", "coverage_guidance").format(
                    coverage_gap=gap,
                    stmt_pct=f"{metrics.statement_pct * 100:.0f}",
                    branch_pct=f"{metrics.branch_pct * 100:.0f}",
                    remaining_batches=remaining_batches,
                )
            )

        if self.config.self_reflection:
            reflection = catalog_text("sections", "self_reflection")
            if not self.config.path_feedback:
                # Drop PATH DIVERGENCE repair instructions when feedback is ablated.
                filtered = [
                    line
                    for line in reflection.splitlines()
                    if "PATH DIVERGENCE" not in line
                    and "expected_path disagreed" not in line
                    and "divergent condition" not in line
                    and "Shorten the path" not in line
                    and "BRANCH NODE CATALOG, replace" not in line
                ]
                reflection = "\n".join(filtered)
            parts.append(reflection)

        return self._join(parts)

    @staticmethod
    def _join(parts: list[str]) -> str:
        return str(load_catalog().get("separator", "\n\n")).join(parts)
