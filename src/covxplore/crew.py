from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool

from covxplore.config import get_settings
from covxplore.prompts import PromptBuilder
from covxplore.tools import SearchNodesTool, GetNodeSourceTool, ExecuteTestcaseTool


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
        verbose: bool = True,
        tracing: bool = True,
    ):
        self._tools = tools
        self._verbose = verbose
        self._tracing = tracing

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
            model = (llm or cfg.deepseek_model).strip()
            model = model if "/" in model else f"deepseek/{model}"
            self._llm = LLM(
                model=model,
                api_key=cfg.deepseek_api_key,
                base_url=cfg.deepseek_base_url,
                max_tokens=cfg.max_tokens,
                timeout=cfg.llm_timeout_sec,
            )

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
            verbose=self._verbose,
            tracing=self._tracing,
        )


def build_crew(
    prompt_config,  # PromptConfig
    *,
    max_iterations: int | None = None,
    verbose: bool = True,
    tracing: bool = True,
) -> tuple["CovxploreCrew", object]:  # (crew_instance, builder)
    _configure_local_crewai_storage()
    builder = PromptBuilder(prompt_config)

    tools = [
        ExecuteTestcaseTool(),
        GetNodeSourceTool(),
        SearchNodesTool(),
    ]

    crew_inst: CovxploreCrew | None = None
    crew_cls = cast(Any, CovxploreCrew)
    crew_inst = crew_cls(
        tools=tools,
        max_iterations=max_iterations,
        verbose=verbose,
        tracing=tracing,
    )
    return crew_inst, builder


def _configure_local_crewai_storage() -> None:
    storage_path = (Path.cwd() / ".covxplore" / "crewai").resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    def db_storage_path() -> str:
        return str(storage_path)

    module_names = [
        "crewai.utilities.paths",
        "crewai.events.listeners.tracing.utils",
        "crewai.flow.persistence.sqlite",
        "crewai.memory.storage.kickoff_task_outputs_storage",
        "crewai.memory.storage.ltm_sqlite_storage",
        "crewai.memory.storage.rag_storage",
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "db_storage_path"):
            setattr(module, "db_storage_path", db_storage_path)
