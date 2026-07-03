"""PromptBuilder — assembles the final agent prompt from enabled sections.

Dynamic placeholders in section files (``{coverage_gap}``, ``{mcdc_pct}``,
``{covered}``, ``{total}``, ``{remaining_iterations}``) are filled at
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


class PromptBuilder:
    """Builds agent system prompt and task description from a PromptConfig.

    Usage:

        Builder = PromptBuilder(config)
        system_prompt = builder.system_prompt()
        task_desc = builder.task_description(
            function_path=...,
            suite=...,
            remaining_iterations=...)
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
        remaining_iterations: int = 0,
        static_conditions_text: str | None = None,
        static_context_text: str | None = None,
        static_source_text: str | None = None,
    ) -> str:
        """Assemble the task description, injecting live coverage state."""
        from covxplore.types import TestSuite  # local import to avoid circular

        has_mcdc = suite is not None and suite.coverage.has_mcdc
        coverage_target = (
            "MC/DC, statement, and branch coverage"
            if has_mcdc
            else "statement and branch coverage"
        )
        parts: list[str] = [
            f"Generate test cases to maximise {coverage_target} for the function at:\n"
            f"  {function_path}\n\n"
            "Workflow:\n"
            "Use the preloaded static data below as ground truth (conditions, context, source).\n"
            "Start by calling execute_testcase_batch with 3-5 focused test bodies; do not search before the first execution.\n"
            "If a helper/type is still unclear, use search_nodes then get_node_source only for that missing symbol.\n"
            "Do NOT call static condition/context fetch tools again; they are already provided below.\n"
            "Prefer execute_testcase_batch with 3-5 focused test bodies targeting distinct obligations; use fewer only when fewer useful candidates remain.\n"
            "After each execution, use coverage feedback to target the next uncovered statements, branches, or conditions."
            + (
                "\nIn each batch, target distinct uncovered nodeId/polarity obligations."
                if has_mcdc
                else ""
            )
        ]

        if static_conditions_text:
            parts.append(
                "PRELOADED STATIC CONDITIONS (one-time snapshot):\n\n"
                f"{static_conditions_text}"
            )
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
                suite.coverage.gap_input(suite.tests, suite.iteration_count)
            ).text
            mcdc_pct = f"{metrics.mcdc_pct * 100:.0f}"
            stmt_pct = f"{metrics.statement_pct * 100:.0f}"
            branch_pct = f"{metrics.branch_pct * 100:.0f}"
            covered = metrics.covered_mcdc_pairs
            total = metrics.total_mcdc_pairs

            section_text = _load_section("coverage_guidance").format(
                coverage_gap=gap,
                mcdc_pct=mcdc_pct,
                stmt_pct=stmt_pct,
                branch_pct=branch_pct,
                covered=covered,
                total=total,
                remaining_iterations=remaining_iterations,
            )
            parts.append(section_text)

        # Self-reflection (static text, included in a task, not system prompt)
        if self.config.self_reflection:
            parts.append(_load_section("self_reflection"))

        return SECTION_SEPARATOR.join(parts)
