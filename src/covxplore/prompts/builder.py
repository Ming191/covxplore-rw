"""Build the fixed test-generation prompt with one reasoning treatment."""

from covxplore.coverage.gap_analyzer import GapAnalyzer
from covxplore.prompts.catalog import catalog_text, load_catalog
from covxplore.prompts.config import ReasoningTechnique
from covxplore.types import TestSuite


class PromptBuilder:
    def __init__(self, technique: ReasoningTechnique):
        self.technique = technique

    def system_prompt(self) -> str:
        return self._join([
            catalog_text("sections", "role"),
            catalog_text("sections", "output_format"),
        ])

    def task_description(
        self,
        function_path: str,
        suite: TestSuite,
        remaining_batches: int,
        static_context_text: str,
        static_source_text: str,
        execution_feedback_text: str | None = None,
        **_: object,
    ) -> str:
        metrics = suite.coverage.metrics(suite.tests)
        gap = GapAnalyzer().analyze(
            suite.coverage.gap_input(suite.tests, suite.batch_count)
        ).text
        parts = [
            catalog_text("task", "description").format(
                function_path=function_path,
                workflow=str(load_catalog()["workflow"]),
                path_requirement=" (omit expected_path)",
            ),
            catalog_text("task", "context").format(content=static_context_text),
            catalog_text("task", "source").format(content=static_source_text),
        ]
        if execution_feedback_text:
            parts.append("PRIOR EXECUTION FEEDBACK:\n" + execution_feedback_text)
        parts.append(
            catalog_text("sections", "coverage_guidance").format(
                coverage_gap=gap,
                stmt_pct=f"{metrics.statement_pct * 100:.0f}",
                branch_pct=f"{metrics.branch_pct * 100:.0f}",
                remaining_batches=remaining_batches,
            )
        )
        return self._join(parts)

    @staticmethod
    def _join(parts: list[str]) -> str:
        return str(load_catalog().get("separator", "\n\n")).join(parts)
