from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool

from covxplore.agents.schemas import (
    GenerateTestBatchAction,
    GenerateTestBatchAnyAction,
    GenerateTestBatchOptionalPathAction,
    GenerateTestBatchSingleAction,
)
from covxplore.config import get_settings
from covxplore.llm import build_llm
from covxplore.prompts.builder import PromptBuilder
from covxplore.tools import GetNodeSourceTool, SearchNodesTool
from covxplore.tools.execute_testcase import RunContext


def _validate_batch_json(schema):
    def check(output):
        text = getattr(output, "raw", output)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return False, "Return one JSON object with a candidates array."
        try:
            batch = schema.model_validate_json(text[start : end + 1])
        except Exception as exc:
            return False, f"Invalid candidate batch: {exc}"
        return True, batch.model_dump_json()

    return check


@CrewBase
class TestGenerationCrew:
    """One-session crew that returns a validated batch for Flow execution."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(
        self,
        tools: list[BaseTool],
        batch_schema: type,
        llm: LLM | str | None = None,
        agent_max_iter: int = 3,
        reasoning: bool | None = None,
        max_reasoning_attempts: int | None = None,
    ):
        self._tools = tools
        self._batch_schema = batch_schema

        if agent_max_iter <= 0:
            raise ValueError("agent_max_iter must be > 0")
        self._agent_max_iter = agent_max_iter

        cfg = get_settings()
        self._reasoning = cfg.agent_reasoning if reasoning is None else reasoning
        self._max_reasoning_attempts = (
            cfg.agent_max_reasoning_attempts
            if max_reasoning_attempts is None
            else max_reasoning_attempts
        )

        if llm is not None and not isinstance(llm, str):
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
            reasoning=self._reasoning,
            max_reasoning_attempts=self._max_reasoning_attempts,
        )

    @task
    def generate_tests(self) -> Task:
        return Task(
            config=self.tasks_config["generate_tests"],  # type: ignore[index]
            guardrail=_validate_batch_json(self._batch_schema),
            guardrail_max_retries=1,
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


def batch_types(prompt_config):
    if not prompt_config.require_expected_path:
        return GenerateTestBatchOptionalPathAction
    if prompt_config.unlimited_batch:
        return GenerateTestBatchAnyAction
    if prompt_config.max_batch_candidates <= 1:
        return GenerateTestBatchSingleAction
    return GenerateTestBatchAction


def build_crew(
    prompt_config,  # PromptConfig
    *,
    agent_max_iter: int = 3,
    start_batch: int = 0,
    run_context: RunContext,
    reasoning: bool | None = None,
) -> tuple["TestGenerationCrew", PromptBuilder]:
    del start_batch
    builder = PromptBuilder(prompt_config)
    tools: list[BaseTool] = []
    if prompt_config.search_tools:
        tools.extend([
            GetNodeSourceTool(run_context=run_context),
            SearchNodesTool(run_context=run_context),
        ])

    crew_inst = TestGenerationCrew(
        tools=tools,
        batch_schema=batch_types(prompt_config),
        agent_max_iter=agent_max_iter,
        reasoning=reasoning,
    )
    return crew_inst, builder
