from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool

from covxplore.config import get_settings
from covxplore.llm import build_llm
from covxplore.prompts import PromptBuilder
from covxplore.tools import (
    ExecuteTestcaseBatchAnyTool,
    ExecuteTestcaseBatchTool,
    ExecuteTestcaseTool,
    GetNodeSourceTool,
    SearchNodesTool,
)
from covxplore.tools.execute_testcase import RunContext


def _guard(ctx: RunContext, start_batch: int, max_forces: int = 3):
    forced = 0

    def check(output):
        nonlocal forced
        suite = ctx.active_suite()
        if suite is not None and suite.batch_count > start_batch:
            return True, output
        if forced >= max_forces:
            return True, output

        forced += 1
        return False, (
            "Final answer emitted before executing this session's batch. "
            "Do not finish yet; call execute_testcase_batch exactly once."
        )

    return check


@CrewBase
class TestGenerationCrew:
    """One-session crew for MC/DC coverage-driven test generation."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(
        self,
        tools: list[BaseTool],
        llm: LLM | str | None = None,
        agent_max_iter: int = 3,
        start_batch: int = 0,
        run_context: RunContext | None = None,
    ):
        self._tools = tools
        self._run_context = run_context
        self._start_batch = start_batch

        if agent_max_iter <= 0:
            raise ValueError("agent_max_iter must be > 0")
        self._agent_max_iter = agent_max_iter

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
            max_iter=self._agent_max_iter,
        )

    @task
    def generate_tests(self) -> Task:
        guardrail = (
            _guard(self._run_context, self._start_batch)
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
    agent_max_iter: int = 3,
    start_batch: int = 0,
    run_context: RunContext,
) -> tuple["TestGenerationCrew", PromptBuilder]:
    builder = PromptBuilder(prompt_config)

    batch_tool = (
        ExecuteTestcaseBatchAnyTool
        if prompt_config.unlimited_batch
        else ExecuteTestcaseBatchTool
    )
    tools = [batch_tool(run_context=run_context)]
    if prompt_config.search_tools:
        tools.extend([GetNodeSourceTool(), SearchNodesTool()])

    crew_inst = TestGenerationCrew(
        tools=tools,
        agent_max_iter=agent_max_iter,
        start_batch=start_batch,
        run_context=run_context,
    )
    return crew_inst, builder
