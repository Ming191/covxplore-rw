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
        if not suite.coverage.has_mcdc:
            self._console.print(
                "[yellow]No MC/DC conditions found — running for statement/branch coverage.[/]"
            )
        static_prompt = self._static_fetcher(config.function_path)
        self.state.static_prompt = _dump_static_prompt(static_prompt)
        self.state.suite = session.suite_codec.dump_suite(suite)
        self.state.token_ledger = session.token_ledger.to_dict()
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
            session.trace_urls = list(self.state.trace_urls)
            suite = session.active_suite()
            start_batch = suite.batch_count
            crew_inst = None
            tracing_url = None
            try:
                crew_inst, builder = self._crew_builder(
                    prompt_config,
                    agent_max_iter=3,
                    start_batch=start_batch,
                    run_context=session.run_context,
                )
                static_prompt = _load_static_prompt(self.state.static_prompt)
                remaining = max(config.max_batches - suite.batch_count, 0)
                inputs = {
                    "agent_backstory": builder.system_prompt(),
                    "task_description": builder.task_description(
                        function_path=config.function_path,
                        suite=suite,
                        remaining_batches=remaining,
                        static_conditions_text=static_prompt.conditions_text,
                        static_context_text=(
                            static_prompt.context_text if prompt_config.preload_context else None
                        ),
                        static_source_text=static_prompt.source_text,
                    ),
                }
                with trace_observation(
                    "covxplore.generate.session",
                    run_id=config.run_id,
                    function_path=config.function_path,
                    prompt_variant=config.prompt_variant,
                    start_batch=start_batch,
                ) as langfuse_url:
                    tracing_url = langfuse_url
                    crew_inst.crew().kickoff(inputs=inputs)
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
                session.record_crew_run(crew_inst, tracing_url)
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
        tokens = state.token_ledger.get("chosen") or {}
        result = GenerationResult(
            config=config,
            suite=suite,
            stop_reason=state.stop_reason or "agent_done",
            error_message=state.error_message,
            crew_prompt_tokens=tokens.get("prompt"),
            crew_completion_tokens=tokens.get("completion"),
            tracing_url=state.trace_urls[-1] if state.trace_urls else None,
            llm_interactions=state.llm_interactions,
        )
        _print_result_summary(result)
        return result

    def resume(self, state_id: str):
        raise NotImplementedError("Flow resume CLI wiring is not implemented yet")


def _dump_static_prompt(data: StaticPromptData) -> dict:
    return {
        "conditions_text": data.conditions_text,
        "context_text": data.context_text,
        "source_text": data.source_text,
    }


def _load_static_prompt(data: dict) -> StaticPromptData:
    return StaticPromptData(
        conditions_text=data.get("conditions_text", ""),
        context_text=data.get("context_text", ""),
        source_text=data.get("source_text", ""),
    )


def _config_from_dict(data: dict):
    from covxplore.generator import GenerationConfig

    return GenerationConfig(
        function_path=data["function_path"],
        prompt_variant=data["prompt_variant"],
        max_batches=int(data["max_batches"]),
        mcdc_target=float(data["mcdc_target"]),
        redundant_streak_limit=int(data["redundant_streak_limit"]),
        fail_streak_limit=int(data["fail_streak_limit"]),
        run_id=data["run_id"],
    )
