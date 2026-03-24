"""covxplore Crew — single-agent MC/DC test generation.

Follows the canonical @CrewBase pattern:
  • Agent config (role, goal, backstory template, max_iter) lives in config/agents.yaml
  • Task config (description template, expected_output) lives in config/tasks.yaml
  • Dynamic values (coverage gap, prompt sections) are injected via kickoff(inputs={})

Usage::

    crew_inst = CovxploreCrew(tools=tools, step_callback=cb)
    crew_inst.crew().kickoff(inputs={
        "agent_backstory": builder.system_prompt(),
        "task_description": builder.task_description(...),
    })
"""

from __future__ import annotations

from typing import Callable, List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool

from covxplore.config import get_settings


class _StopGeneration(BaseException):
    """Raised by the step callback to halt the crew early."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@CrewBase
class CovxploreCrew:
    """Single-agent crew for MC/DC coverage-driven test generation.

    Parameters passed at construction time (not from YAML):
      tools          — the 4 AkaUT REST-API tools
      step_callback  — called after every agent step; raise _StopGeneration to stop
      llm            — LiteLLM model string (e.g. 'litellm/deepseek/deepseek-chat')
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(
        self,
        tools: list[BaseTool],
        step_callback: Callable | None = None,
        llm: str | None = None,
    ):
        self._tools = tools
        self._step_callback = step_callback
        cfg = get_settings()
        if llm:
            self._llm = llm
        else:
            model = cfg.deepseek_model.strip()
            # Accept either "deepseek-chat" or fully-qualified values like
            # "deepseek/deepseek-chat".
            self._llm = model if "/" in model else f"deepseek/{model}"

    # ------------------------------------------------------------------ #
    # Agent                                                               #
    # ------------------------------------------------------------------ #

    @agent
    def test_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["test_generator"],  # type: ignore[index]
            tools=self._tools,
            llm=self._llm,
        )

    # ------------------------------------------------------------------ #
    # Task                                                                #
    # ------------------------------------------------------------------ #

    @task
    def generate_tests(self) -> Task:
        return Task(
            config=self.tasks_config["generate_tests"],  # type: ignore[index]
        )

    # ------------------------------------------------------------------ #
    # Crew                                                                #
    # ------------------------------------------------------------------ #

    @crew
    def crew(self) -> Crew:
        """Assemble the sequential single-agent crew."""
        return Crew(
            agents=self.agents,  # auto-collected by @agent
            tasks=self.tasks,  # auto-collected by @task
            process=Process.sequential,
            verbose=True,
            step_callback=self._step_callback,
            tracing=True
        )


# ---------------------------------------------------------------------------
# Factory — builds a ready-to-kickoff crew for one experiment run
# ---------------------------------------------------------------------------


def build_crew(
    function_path: str,
    prompt_config,  # PromptConfig
    suite,  # TestSuite
) -> tuple["CovxploreCrew", object]:  # (crew_instance, builder)
    """Create a ``CovxploreCrew`` + ``PromptBuilder`` for one generation run.

    The caller should then::

        inputs = {
            "agent_backstory":  builder.system_prompt(),
            "task_description": builder.task_description(function_path, suite, remaining),
        }
        crew_inst.crew().kickoff(inputs=inputs)

    Returns the crew instance and builder so callers can rebuild the task
    description mid-run if needed.
    """
    from covxplore.prompts import PromptBuilder
    from covxplore.tools import (
        ExecuteTestcaseTool,
        GetFunctionContextTool,
        GetNodeSourceTool,
        SearchNodesTool,
        get_shared_suite,
    )
    from covxplore.config import get_settings

    cfg = get_settings()
    builder = PromptBuilder(prompt_config)

    tools = [
        GetFunctionContextTool(),
        GetNodeSourceTool(),
        SearchNodesTool(),
        ExecuteTestcaseTool(),
    ]

    crew_inst: CovxploreCrew | None = None
    last_prompt_tokens = 0
    last_completion_tokens = 0

    def step_callback(agent_output) -> None:
        nonlocal last_prompt_tokens, last_completion_tokens
        
        current_suite = get_shared_suite()
        if current_suite is None:
            return

        # 1. Token tracking
        if crew_inst is not None:
            try:
                metrics = crew_inst.crew().usage_metrics
                if metrics and getattr(metrics, "prompt_tokens", 0) > 0 and current_suite.tests:
                    latest = current_suite.tests[-1]
                    latest.token_input += (metrics.prompt_tokens - last_prompt_tokens)
                    latest.token_output += (metrics.completion_tokens - last_completion_tokens)
                    last_prompt_tokens = metrics.prompt_tokens
                    last_completion_tokens = metrics.completion_tokens
            except Exception:
                pass

        # 2. Stop conditions
        if current_suite.iteration_count >= cfg.max_iterations:
            raise _StopGeneration(f"Maximum of {cfg.max_iterations} iterations reached. This is an expected early-stop.")
        if current_suite.mcdc_coverage_pct >= cfg.mcdc_target:
            raise _StopGeneration("Target coverage reached. This is an expected early-stop.")
        if current_suite.iteration_count > 0 and not current_suite.unvisited_summary():
            raise _StopGeneration("No unvisited conditions left. This is an expected early-stop.")

    crew_inst = CovxploreCrew(tools, step_callback)
    return crew_inst, builder
