from __future__ import annotations

import traceback
from typing import Any, Callable

from crewai.flow.flow import Flow, listen, start
from crewai.flow.persistence import persist
from rich.console import Console

from covxplore.crews.test_generation.crew import build_crew
from covxplore.generation.prompt_context import StaticPromptData, fetch_static_prompt_data
from covxplore.generation.runtime_session import GenerationFlowState, GenerationRuntimeSession, TestSuiteCodec
from covxplore.generation.stop_reasons import StopReason
from covxplore.generation.tokens import totals_from_usage_metrics
from covxplore.observability import init_observability, trace_observation
from covxplore.prompts.registry import get_variant
from covxplore.tools.execute_testcase import HardStop

_console = Console()


@persist()
class GenerationFlow(Flow[GenerationFlowState]):
    def __init__(
        self,
        *,
        crew_builder: Callable = build_crew,
        static_fetcher: Callable[[str], StaticPromptData] = fetch_static_prompt_data,
        console: Console | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._crew_builder = crew_builder
        self._static_fetcher = static_fetcher
        self._console = console or _console

    @start()
    def initialize(self) -> None:
        if not self.state.config:
            raise ValueError("GenerationFlow requires config input")
        config = _config_from_dict(self.state.config)
        session = GenerationRuntimeSession(config)
        suite = session.open(self._console)
        static_prompt = self._static_fetcher(config.function_path)
        self.state.static_prompt = _dump_static_prompt(static_prompt)
        self.state.suite = session.suite_codec.dump_suite(suite)
        self.state.token_ledger = session.token_ledger.to_dict()
        self.state.dynamic_knowledge = {}
        self.state.trace_urls = []
        self.state.stop_reason = "agent_done"
        self.state.error_message = None
        session.cleanup()

    @listen(initialize)
    def run_sessions(self) -> StopReason:
        config = _config_from_dict(self.state.config)
        prompt_config = get_variant(config.prompt_variant)
        stop_reason: StopReason = "agent_done"
        error_message = None

        init_observability()
        while stop_reason == "agent_done":
            session = GenerationRuntimeSession(config)
            session.restore_suite(self.state.suite)
            session.restore_ledger(self.state.token_ledger)
            session.restore_dynamic_knowledge(self.state.dynamic_knowledge)
            session.trace_urls = list(self.state.trace_urls)
            suite = session.active_suite()
            start_batch = suite.batch_count
            crew_inst = None
            crew = None
            tracing_url = None
            try:
                crew_inst, builder = self._crew_builder(
                    prompt_config,
                    agent_max_iter=3,
                    start_batch=start_batch,
                    run_context=session.run_context,
                    reasoning=config.reasoning,
                )
                static_prompt = _load_static_prompt(self.state.static_prompt)
                session.run_context.set_branch_catalog_ids(static_prompt.branch_catalog_ids)
                remaining = max(config.max_batches - suite.batch_count, 0)
                inputs = {
                    "agent_backstory": builder.system_prompt(),
                    "task_description": builder.task_description(
                        function_path=config.function_path,
                        suite=suite,
                        remaining_batches=remaining,
                        static_context_text=(
                            static_prompt.context_text if prompt_config.preload_context else None
                        ),
                        static_source_text=static_prompt.source_text,
                        static_branch_catalog_text=(
                            static_prompt.branch_catalog_text or None
                            if prompt_config.preload_branch_catalog
                            else None
                        ),
                        dynamic_knowledge_text=_knowledge_text(session.run_context.dynamic_knowledge),
                    ),
                }
                session.run_context.path_feedback = prompt_config.path_feedback
                session.run_context.include_exec_detail = prompt_config.include_exec_detail
                session.run_context.max_batch_candidates = prompt_config.max_batch_candidates
                with trace_observation(
                    "covxplore.generate.session",
                    run_id=config.run_id,
                    function_path=config.function_path,
                    prompt_variant=config.prompt_variant,
                    start_batch=start_batch,
                ) as langfuse_url:
                    tracing_url = langfuse_url
                    crew = crew_inst.crew()
                    crew.kickoff(inputs=inputs)
            except HardStop as exc:
                stop_reason = exc.reason  # type: ignore[assignment]
                error_message = None
                self._console.print(f"[green]Hard stop ({exc.reason}): {exc}[/]")
            except BaseException as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                if "Task failed guardrail validation" in str(exc):
                    stop_reason = "guardrail_incomplete"
                    self._console.print(f"[yellow]Stopped: {error_message}[/]")
                else:
                    stop_reason = "error"
                    self._console.print(f"[red]Error: {error_message}[/]")
                    traceback.print_exc()
            finally:
                session.record_crew_run(crew or crew_inst, tracing_url)
                suite = session.active_suite()
                if stop_reason == "agent_done":
                    stop_reason = session.stop_policy.terminal_reason(suite)
                if stop_reason != "agent_done":
                    session.finalize_tokens()
                snapshot = session.snapshot(
                    stop_reason=stop_reason,
                    error_message=error_message,
                    static_prompt=self.state.static_prompt,
                    llm_interactions=self.state.llm_interactions,
                )
                self.state.suite = snapshot.suite
                self.state.token_ledger = snapshot.token_ledger
                self.state.dynamic_knowledge = snapshot.dynamic_knowledge
                self.state.trace_urls = snapshot.trace_urls
                self.state.stop_reason = snapshot.stop_reason
                self.state.error_message = snapshot.error_message
                session.cleanup()
        return stop_reason

    @listen(run_sessions)
    def finish(self, stop_reason: StopReason):
        self.state.stop_reason = stop_reason
        return self.state


class GenerationFlowRunner:
    def __init__(
        self,
        *,
        crew_builder: Callable = build_crew,
        static_fetcher: Callable[[str], StaticPromptData] = fetch_static_prompt_data,
        console: Console | None = None,
    ):
        self._crew_builder = crew_builder
        self._static_fetcher = static_fetcher
        self._console = console or _console

    def run(self, config):
        from covxplore.generator import GenerationResult, _print_result_summary

        assert config.run_id is not None
        self._console.rule(
            f"[bold cyan]Run {config.run_id} | variant={config.prompt_variant!r}"
        )
        flow = GenerationFlow(
            crew_builder=self._crew_builder,
            static_fetcher=self._static_fetcher,
            console=self._console,
            initial_state=GenerationFlowState(config=config.to_dict()),
        )
        final_state = flow.kickoff()
        if isinstance(final_state, GenerationFlowState):
            state = final_state
        else:
            state = flow.state
        suite = TestSuiteCodec().load_suite(state.suite)
        flow_tokens = totals_from_usage_metrics(getattr(flow, "usage_metrics", None))
        tokens = flow_tokens.to_dict() if flow_tokens.total > 0 else state.token_ledger.get("chosen") or {}
        static_prompt = _load_static_prompt(state.static_prompt)
        result = GenerationResult(
            config=config,
            suite=suite,
            stop_reason=state.stop_reason or "agent_done",
            error_message=state.error_message,
            crew_prompt_tokens=tokens.get("prompt"),
            crew_completion_tokens=tokens.get("completion"),
            tracing_url=state.trace_urls[-1] if state.trace_urls else None,
            llm_interactions=state.llm_interactions,
            branch_catalog_ids=set(static_prompt.branch_catalog_ids),
        )
        _print_result_summary(result)
        return result

    def resume(self, state_id: str):
        raise NotImplementedError("Flow resume CLI wiring is not implemented yet")


def _knowledge_text(knowledge: dict[str, str]) -> str | None:
    if not knowledge:
        return None
    entries = [f"[{key}]\n{value}" for key, value in knowledge.items()]
    text = "\n\n".join(entries)
    return text[:12000]


def _dump_static_prompt(data: StaticPromptData) -> dict:
    return {
        "context_text": data.context_text,
        "source_text": data.source_text,
        "branch_catalog_text": data.branch_catalog_text,
        "branch_catalog_ids": list(data.branch_catalog_ids),
    }


def _load_static_prompt(data: dict) -> StaticPromptData:
    return StaticPromptData(
        context_text=data.get("context_text", ""),
        source_text=data.get("source_text", ""),
        branch_catalog_text=data.get("branch_catalog_text", ""),
        branch_catalog_ids=[int(node_id) for node_id in data.get("branch_catalog_ids") or []],
    )


def _config_from_dict(data: dict):
    from covxplore.generator import GenerationConfig

    return GenerationConfig(
        function_path=data["function_path"],
        prompt_variant=data["prompt_variant"],
        max_batches=int(data["max_batches"]),
        redundant_streak_limit=int(data["redundant_streak_limit"]),
        fail_streak_limit=int(data["fail_streak_limit"]),
        reasoning=bool(data.get("reasoning", False)),
        run_id=data["run_id"],
    )
