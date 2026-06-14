"""PromptBuilder — assembles the final agent prompt from enabled sections.

Dynamic placeholders in section files (``{coverage_gap}``, ``{mcdc_pct}``,
``{covered}``, ``{total}``, ``{stmt_progress}``, ``{branch_progress}``,
``{remaining_iterations}``) are filled at build-time from the live TestSuite
state.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache

from covxplore.prompts.config import PromptConfig

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
        from covxplore.test_suite import TestSuite  # local import to avoid circular

        has_mcdc = suite is not None and suite.total_mcdc_conditions > 0
        coverage_target = (
            "MC/DC, statement, and branch coverage"
            if has_mcdc
            else "statement and branch coverage"
        )
        workflow = (
            f"Generate test cases to maximise {coverage_target} for the function at:\n"
            f"  {function_path}\n\n"
            "Workflow:\n"
            "Use the preloaded static data below as ground truth (conditions, context, source).\n"
            "If a helper/type is still unclear, use search_nodes then get_node_source only for that missing symbol.\n"
            "Do NOT call static condition/context fetch tools again; they are already provided below.\n"
        )
        workflow += "Generate one focused test body and call execute_testcase.\n"
        workflow += "After each execution, use coverage feedback to target the next uncovered statements, branches, or conditions.\n"
        if has_mcdc:
            workflow += "Condition identity is nodeId only. Always cite nodeId when planning/justifying a test.\n"
            workflow += "Target exactly one uncovered obligation per iteration before calling execute_testcase.\n"
        workflow += "Stop when all coverage targets are met or the iteration budget is exhausted."
        parts: list[str] = [workflow]

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
        if self.config.coverage_guidance and suite is not None:
            assert isinstance(suite, TestSuite)
            gap = suite.coverage_gap_prompt_fragment()
            mcdc_pct = f"{suite.mcdc_coverage_pct * 100:.0f}"
            stmt_pct = f"{suite.statement_coverage_pct * 100:.0f}"
            branch_pct = f"{suite.branch_coverage_pct * 100:.0f}"
            covered = len(suite.covered_keys)
            total = suite.total_mcdc_conditions
            stmt_covered = suite.covered_statements
            stmt_total = suite.total_statements
            branch_covered = suite.covered_branches
            branch_total = suite.total_branches
            stmt_progress = (
                f"{stmt_pct}% ({stmt_covered}/{stmt_total} covered)"
                if stmt_total > 0
                else "N/A (no executed testcase yet)"
            )
            branch_progress = (
                f"{branch_pct}% ({branch_covered}/{branch_total} covered)"
                if branch_total > 0
                else "N/A (no executed testcase yet)"
            )

            section_text = _load_section("coverage_guidance").format(
                coverage_gap=gap,
                mcdc_pct=mcdc_pct,
                stmt_progress=stmt_progress,
                branch_progress=branch_progress,
                covered=covered,
                total=total,
                remaining_iterations=remaining_iterations,
            )
            parts.append(section_text)

        # Self-reflection (static text, included in a task, not system prompt)
        if self.config.self_reflection:
            parts.append(_load_section("self_reflection"))

        return SECTION_SEPARATOR.join(parts)
