"""PromptBuilder — assembles the final agent prompt from enabled sections.

Dynamic placeholders in section files (``{coverage_gap}``, ``{mcdc_pct}``,
``{covered}``, ``{total}``, ``{remaining_iterations}``) are filled at
build-time from the live TestSuite state.
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
    ) -> str:
        """Assemble the task description, injecting live coverage state."""
        from covxplore.models import TestSuite  # local import to avoid circular

        parts: list[str] = [f"Generate MC/DC-covering test cases for the function at:\n"
                            f"  {function_path}\n\n"
                            "Workflow:\n"
                            "1. Call get_conditions_static to get the full list of MC/DC conditions "
                            "that must be covered — do this FIRST so you know the coverage target.\n"
                            "2. Call get_function_context to understand the function signature and types.\n"
                            "3. Call get_node_source on the same path to read the function body.\n"
                            "4. If the context references unknown types or helpers, call search_nodes "
                            "then get_node_source to resolve them.\n"
                            "5. Write a test body targeting a specific uncovered condition and call execute_testcase.\n"
                            "6. After each execution, use the coverage feedback to target the next "
                            "uncovered condition.\n"
                            "7. Stop when all MC/DC conditions are covered or the iteration budget is exhausted."]

        # Always include the goal

        # Dynamic coverage guidance
        if self.config.coverage_guidance and suite is not None:
            assert isinstance(suite, TestSuite)
            gap = suite.coverage_gap_prompt_fragment()
            mcdc_pct = f"{suite.mcdc_coverage_pct * 100:.0f}"
            covered = len(suite.covered_keys)
            total = suite.total_mcdc_conditions

            section_text = _load_section("coverage_guidance").format(
                coverage_gap=gap,
                mcdc_pct=mcdc_pct,
                covered=covered,
                total=total,
                remaining_iterations=remaining_iterations,
            )
            parts.append(section_text)

        # Self-reflection (static text, included in a task, not system prompt)
        if self.config.self_reflection:
            parts.append(_load_section("self_reflection"))

        return SECTION_SEPARATOR.join(parts)
