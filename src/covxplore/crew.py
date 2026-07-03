from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool

from covxplore.config import get_settings
from covxplore.llm import build_llm
from covxplore.prompts import PromptBuilder
from covxplore.tools import SearchNodesTool, GetNodeSourceTool, ExecuteTestcaseBatchTool, ExecuteTestcaseTool
from covxplore.tools.execute_testcase import RunContext


def _guard(ctx: RunContext, max_iterations: int, max_forces: int = 3):
    forced = 0

    def check(output):
        nonlocal forced
        suite = ctx.active_suite()
        if suite is not None and suite.iteration_count >= max_iterations:
            return True, output
        if forced >= max_forces:
            return True, output

        forced += 1
        return False, (
            "Final answer emitted before a hard stop. "
            "Do not finish yet; call execute_testcase_batch."
        )

    return check


@CrewBase
class CovxploreCrew:
    """Single-agent crew for MC/DC coverage-driven test generation.

    Parameters passed at construction time (not from YAML):
      tools          — the 4 AkaUT REST-API tools
      llm            — LLM instance or model string; defaults to Settings values
      max_iterations — run-level iteration cap for the agent loop
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(
        self,
        tools: list[BaseTool],
        llm: LLM | str | None = None,
        max_iterations: int | None = None,
        run_context: RunContext | None = None,
    ):
        self._tools = tools
        self._run_context = run_context

        cfg = get_settings()
        resolved_max_iterations = (
            max_iterations if max_iterations is not None else cfg.max_iterations
        )
        if resolved_max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        self._max_iterations = resolved_max_iterations

        if isinstance(llm, LLM):
            self._llm = llm
        else:
            self._llm = build_llm(llm)

    @agent
    def test_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["test_generator"],  # type: ignore[index]
            tools=self._tools,
            llm=self._llm,
            max_iter=self._max_iterations,
        )

    @task
    def generate_tests(self) -> Task:
        guardrail = (
            _guard(self._run_context, self._max_iterations)
            if self._run_context is not None
            else None
        )
        return Task(
            config=self.tasks_config["generate_tests"],  # type: ignore[index]
            guardrail=guardrail,
            guardrail_max_retries=3,
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the sequential single-agent crew."""
        return Crew(
            agents=self.agents,  # auto-collected by @agent
            tasks=self.tasks,  # auto-collected by @task
            process=Process.sequential,
            verbose=True,
            tracing=False,
        )


def build_crew(
    prompt_config,  # PromptConfig
    *,
    max_iterations: int | None = None,
    run_context: RunContext,
) -> tuple["CovxploreCrew", PromptBuilder]:
    builder = PromptBuilder(prompt_config)

    tools = [
        ExecuteTestcaseBatchTool(run_context=run_context),
#         ExecuteTestcaseTool(run_context=run_context),
        GetNodeSourceTool(),
        SearchNodesTool(),
    ]

    crew_inst = CovxploreCrew(
        tools=tools,
        max_iterations=max_iterations,
        run_context=run_context,
    )
    return crew_inst, builder
