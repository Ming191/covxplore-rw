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
    ):
        self._tools = tools

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
        return Task(
            config=self.tasks_config["generate_tests"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the sequential single-agent crew."""
        return Crew(
            agents=self.agents,  # auto-collected by @agent
            tasks=self.tasks,  # auto-collected by @task
            process=Process.sequential,
            verbose=True,
            tracing=True,
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
        ExecuteTestcaseTool(run_context=run_context),
        GetNodeSourceTool(),
        SearchNodesTool(),
    ]

    crew_inst = CovxploreCrew(tools=tools, max_iterations=max_iterations)
    return crew_inst, builder
