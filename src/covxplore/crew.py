from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool

from covxplore.config import get_settings


@CrewBase
class CovxploreCrew:
    """Single-agent crew for MC/DC coverage-driven test generation.

    Parameters passed at construction time (not from YAML):
      tools          — the 4 AkaUT REST-API tools
      llm            — LLM instance or model string; defaults to Settings values
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(
        self,
        tools: list[BaseTool],
        llm: LLM | str | None = None,
    ):
        self._tools = tools

        cfg = get_settings()
        if isinstance(llm, LLM):
            self._llm = llm
        else:
            model = (llm or cfg.deepseek_model).strip()
            model = model if "/" in model else f"deepseek/{model}"
            self._llm = LLM(
                model=model,
                api_key=cfg.deepseek_api_key,
                base_url=cfg.deepseek_base_url,
            )

    @agent
    def test_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["test_generator"],  # type: ignore[index]
            tools=self._tools,
            llm=self._llm,
            max_iter=get_settings().max_iterations,
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
            tasks=self.tasks,    # auto-collected by @task
            process=Process.sequential,
            verbose=True,
            tracing=True,
        )


def build_crew(
    prompt_config,  # PromptConfig
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
        GetConditionsStaticTool,
        GetFunctionContextTool,
        GetNodeSourceTool,
        SearchNodesTool,
    )

    builder = PromptBuilder(prompt_config)

    tools = [
        GetConditionsStaticTool(),
        GetFunctionContextTool(),
        GetNodeSourceTool(),
        SearchNodesTool(),
        ExecuteTestcaseTool(),
    ]

    crew_inst: CovxploreCrew | None = None
    crew_inst = CovxploreCrew(tools)
    return crew_inst, builder
