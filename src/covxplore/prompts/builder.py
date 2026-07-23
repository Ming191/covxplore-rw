"""PromptBuilder — assembles the final agent prompt from enabled sections.

Dynamic placeholders in section files (``{coverage_gap}``, ``{stmt_pct}``,
``{branch_pct}``, ``{remaining_batches}``) are filled at
build-time from the live TestSuite state.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache

from covxplore.prompts.config import PromptConfig
from covxplore.coverage.gap_analyzer import GapAnalyzer

_SECTIONS_DIR = pathlib.Path(__file__).parent / "sections"

SECTION_SEPARATOR = "\n\n" + "─" * 60 + "\n\n"


@lru_cache(maxsize=32)
def _load_section(name: str) -> str:
    path = _SECTIONS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt section not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _tool_workflow_text(config: PromptConfig) -> str:
    if config.unlimited_batch:
        return (
            "Plan one batch for this session and call execute_testcase_batch exactly once: "
            "one candidate per useful uncovered statement or branch side plus boundary/error cases. "
            "Avoid duplicate path shapes. No search/source tools are available in this variant; use preloaded source and coverage feedback."
        )
    if not config.search_tools:
        return (
            "Plan one batch for this session and call execute_testcase_batch exactly once with 3-5 focused test bodies. "
            "No search/source tools are available in this variant; use only the "
            "preloaded source and coverage feedback."
        )
    return (
        "Plan one batch for this session and call execute_testcase_batch exactly once with 3-5 focused test bodies; "
        "do not search before that execution.\n"
        "If a helper/type is still unclear, use search_nodes then get_node_source "
        "only for that missing symbol before the batch."
    )


class PromptBuilder:
    """Builds agent system prompt and task description from a PromptConfig.

    Usage:

        Builder = PromptBuilder(config)
        system_prompt = builder.system_prompt()
        task_desc = builder.task_description(
            function_path=...,
            suite=...,
            remaining_batches=...)
    """

    def __init__(self, config: PromptConfig):
        self.config = config

    def system_prompt(self) -> str:
        """Assemble static system prompt (no dynamic state needed)."""
        static_sections = [
            "role_persona",
            "cot_reasoning",
            "few_shot_examples",
            "output_format",
        ]
        parts = []
        for section in static_sections:
            if section in self.config.enabled_sections():
                parts.append(_load_section(section))
        return SECTION_SEPARATOR.join(parts) if parts else ""

    def task_description(
        self,
        function_path: str,
        suite=None,  # TestSuite | None
        remaining_batches: int = 0,
        static_context_text: str | None = None,
        static_source_text: str | None = None,
    ) -> str:
        """Assemble the task description, injecting live coverage state."""
        from covxplore.types import TestSuite  # local import to avoid circular

        parts: list[str] = [
            f"Generate test cases to maximise statement and branch coverage for the function at:\n"
            f"  {function_path}\n\n"
            "Workflow:\n"
            "Use preloaded context and source as ground truth.\n"
            + _tool_workflow_text(self.config)
            + "\nDo NOT call static condition/context fetch tools again; they are already provided below.\n"
            "Prefer execute_testcase_batch with 3-5 focused test bodies targeting distinct obligations; use fewer only when fewer useful candidates remain.\n"
            "Stop after the batch result and output DONE. The Flow will relaunch a fresh session with updated coverage feedback if more work remains."
        ]

        if static_context_text:
            parts.append(
                "PRELOADED FUNCTION CONTEXT (one-time snapshot):\n\n"
                f"{static_context_text}"
            )
        if static_source_text:
            parts.append(
                f"PRELOADED FOCAL SOURCE (one-time snapshot):\n\n{static_source_text}"
            )

        # Dynamic coverage guidance
        if self.config.coverage_guidance and suite is not None:
            assert isinstance(suite, TestSuite)
            metrics = suite.coverage.metrics(suite.tests)
            gap = GapAnalyzer().analyze(
                suite.coverage.gap_input(suite.tests, suite.batch_count)
            ).text
            stmt_pct = f"{metrics.statement_pct * 100:.0f}"
            branch_pct = f"{metrics.branch_pct * 100:.0f}"

            section_text = _load_section("coverage_guidance").format(
                coverage_gap=gap,
                stmt_pct=stmt_pct,
                branch_pct=branch_pct,
                remaining_batches=remaining_batches,
            )
            parts.append(section_text)

        # Self-reflection (static text, included in a task, not system prompt)
        if self.config.self_reflection:
            parts.append(_load_section("self_reflection"))

        return SECTION_SEPARATOR.join(parts)
