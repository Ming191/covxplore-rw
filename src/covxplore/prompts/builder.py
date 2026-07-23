"""Build agent prompts from YAML sections and live coverage state."""

from __future__ import annotations

from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.prompts.catalog import catalog_text, load_catalog
from covxplore.prompts.config import PromptConfig

_STATIC_SECTIONS = ("role_persona", "cot_reasoning", "few_shot_examples", "output_format")


def _workflow_name(config: PromptConfig) -> str:
    if config.unlimited_batch:
        return "unlimited_batch"
    if not config.search_tools:
        return "no_search"
    return "search"


class PromptBuilder:
    """Build system and task prompts from YAML plus runtime values."""

    def __init__(self, config: PromptConfig):
        self.config = config

    def system_prompt(self) -> str:
        parts = [
            catalog_text("sections", section)
            for section in _STATIC_SECTIONS
            if section in self.config.enabled_sections()
        ]
        return self._join(parts) if parts else ""

    def task_description(
        self,
        function_path: str,
        suite=None,
        remaining_batches: int = 0,
        static_context_text: str | None = None,
        static_source_text: str | None = None,
    ) -> str:
        from covxplore.types import TestSuite

        parts = [
            catalog_text("task", "description").format(
                function_path=function_path,
                workflow=catalog_text("workflows", _workflow_name(self.config)),
            )
        ]

        if static_context_text:
            parts.append(
                catalog_text("task", "context").format(content=static_context_text)
            )
        if static_source_text:
            parts.append(
                catalog_text("task", "source").format(content=static_source_text)
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
            parts.append(catalog_text("sections", "self_reflection"))

        return self._join(parts)

    @staticmethod
    def _join(parts: list[str]) -> str:
        return str(load_catalog().get("separator", "\n\n")).join(parts)
