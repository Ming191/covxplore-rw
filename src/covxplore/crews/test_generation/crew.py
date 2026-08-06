from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task, TaskOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from covxplore.agents.schemas import GenerateTestBatchAction
from covxplore.llm import build_llm
from covxplore.prompts.builder import PromptBuilder


def _validate_batch(output: TaskOutput):
    try:
        batch = GenerateTestBatchAction.model_validate_json(output.raw)
    except Exception as exc:
        return False, f"Return only one valid candidate-batch JSON object: {exc}"
    return True, batch.model_dump_json()


@CrewBase
class TestGenerationCrew:
    """Single-agent crew returning one typed candidate batch."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(self, llm: LLM | str | None = None):
        self._llm = llm if llm is not None and not isinstance(llm, str) else build_llm(llm)

    @agent
    def test_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["test_generator"],  # type: ignore[index]
            llm=self._llm,
            max_iter=1,
        )

    @task
    def generate_tests(self) -> Task:
        return Task(
            config=self.tasks_config["generate_tests"],  # type: ignore[index]
            guardrail=_validate_batch,
            guardrail_max_retries=1,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=False,
        )


def build_crew(prompt_config) -> tuple[TestGenerationCrew, PromptBuilder]:
    return TestGenerationCrew(), PromptBuilder(prompt_config)
