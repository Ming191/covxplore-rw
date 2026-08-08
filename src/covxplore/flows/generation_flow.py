from __future__ import annotations

import traceback
from typing import Callable

from crewai.flow.flow import Flow, listen, start
from rich.console import Console

from covxplore.generation.runner import CrewBundle, build_crew
from covxplore.generation.prompt_context import StaticPromptData, fetch_static_prompt_data
from covxplore.generation.runtime_session import GenerationFlowState, GenerationRuntimeSession, StaticPromptSnapshot, TestSuiteCodec
from covxplore.generation.stop_reasons import StopReason
from covxplore.generation.tokens import totals_from_usage_metrics
from covxplore.prompts.registry import get_variant
from covxplore.reasoning.strategies import ReasoningInput
from covxplore.tools.execute_testcase import ExecuteTestcaseBatchTool, HardStop




class GenerationFlow(Flow[GenerationFlowState]):
    def __init__(
        self,
        *,
        crew_builder: Callable[..., CrewBundle] = build_crew,
        static_fetcher: Callable[[str, str], StaticPromptData] = fetch_static_prompt_data,
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
        suite = session.open()
        static_prompt = self._static_fetcher(config.function_path, config.context_version)
        self.state.static_prompt = StaticPromptSnapshot(
            context_text=static_prompt.context_text,
            source_text=static_prompt.source_text,
            total_statements=static_prompt.total_statements,
            total_branches=static_prompt.total_branches,
        )
        if static_prompt.total_statements is not None and static_prompt.total_branches is not None:
            suite.coverage.seed_totals(
                static_prompt.total_statements,
                static_prompt.total_branches,
            )
        self.state.suite = session.suite_codec.dump_suite(suite)
        self.state.token_ledger = session.token_ledger.to_dict()
        self.state.execution_feedback = ""
        self.state.stop_reason = "agent_done"
        self.state.error_message = None
        session.cleanup()

    @listen(initialize)
    def run_sessions(self) -> StopReason:
        config = _config_from_dict(self.state.config)
        technique = get_variant(config.prompt_variant)
        stop_reason: StopReason = "agent_done"
        error_message = None

        while stop_reason == "agent_done":
            session = GenerationRuntimeSession(config)
            session.restore_suite(self.state.suite)
            session.restore_ledger(self.state.token_ledger)
            suite = session.active_suite()
            bundle: CrewBundle | None = None
            try:
                bundle = self._crew_builder(technique)
                static_prompt_snap = self.state.static_prompt
                reasoning_input = ReasoningInput(
                    system_prompt=bundle.builder.system_prompt(),
                    task_prompt=bundle.builder.task_description(
                        function_path=config.function_path,
                        suite=suite,
                        remaining_batches=config.max_batches - suite.batch_count,
                        static_context_text=static_prompt_snap.context_text,
                        static_source_text=static_prompt_snap.source_text,
                        execution_feedback_text=self.state.execution_feedback or None,
                    ),
                )
                batch = bundle.runner.generate(reasoning_input)
                self.state.execution_feedback = ExecuteTestcaseBatchTool(
                    run_context=session.run_context
                )._run(batch.candidates)
            except HardStop as exc:
                stop_reason = exc.reason  # type: ignore[assignment]
                error_message = None
                self._console.print(f"[green]Hard stop ({exc.reason}): {exc}[/]")
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                if "Task failed guardrail validation" in str(exc):
                    stop_reason = "guardrail_incomplete"
                    self._console.print(f"[yellow]Stopped: {error_message}[/]")
                else:
                    stop_reason = "error"
                    self._console.print(f"[red]Error: {error_message}[/]")
                    traceback.print_exc()
            finally:
                session.record_crew_run(bundle.runner if bundle is not None else None)
                suite = session.active_suite()
                if stop_reason == "agent_done":
                    stop_reason = session.stop_policy.terminal_reason(suite)
                if stop_reason != "agent_done":
                    session.finalize_tokens()
                snapshot = session.snapshot(
                    stop_reason=stop_reason,
                    error_message=error_message,
                    static_prompt=self.state.static_prompt,
                    execution_feedback=self.state.execution_feedback,
                )
                self.state.suite = snapshot.suite
                self.state.token_ledger = snapshot.token_ledger
                self.state.execution_feedback = snapshot.execution_feedback
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
        crew_builder: Callable[..., CrewBundle] = build_crew,
        static_fetcher: Callable[[str, str], StaticPromptData] = fetch_static_prompt_data,
        console: Console | None = None,
    ):
        self._crew_builder = crew_builder
        self._static_fetcher = static_fetcher
        self._console = console or Console()

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
        final_state = flow.kickoff(inputs={"config": config.to_dict()})
        state = final_state if isinstance(final_state, GenerationFlowState) else flow.state
        suite = TestSuiteCodec().load_suite(state.suite)
        flow_tokens = totals_from_usage_metrics(getattr(flow, "usage_metrics", None))
        tokens = flow_tokens.to_dict() if flow_tokens.total > 0 else state.token_ledger.get("chosen") or {}
        result = GenerationResult(
            config=config,
            suite=suite,
            stop_reason=state.stop_reason or "agent_done",
            error_message=state.error_message,
            crew_prompt_tokens=tokens.get("prompt"),
            crew_completion_tokens=tokens.get("completion"),
        )
        _print_result_summary(result, self._console)
        return result



def _config_from_dict(data: dict):
    from covxplore.generator import GenerationConfig

    return GenerationConfig(
        function_path=data["function_path"],
        prompt_variant=data["prompt_variant"],
        context_version=data.get("context_version", "v1"),
        max_batches=int(data["max_batches"]),
        redundant_streak_limit=int(data["redundant_streak_limit"]),
        fail_streak_limit=int(data["fail_streak_limit"]),
        run_id=data["run_id"],
    )
