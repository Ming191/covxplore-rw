from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from covxplore.ablation import AblationRunner
from covxplore.events import keys_to_camel, utc_now_iso
from covxplore.experiment import ExperimentConfig

from covxplore_ui.report import build_report_from_run_state, build_report_from_summary
from covxplore_ui.results import find_result, load_result, result_paths, summary_to_run_state


FINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class EventRecord:
    index: int
    type: str
    run_id: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "type": self.type,
            "runId": self.run_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


@dataclass
class ManagedRun:
    run_id: str
    function_path: str
    variant: str
    out_dir: Path
    status: str = "queued"
    metrics: dict[str, Any] = field(default_factory=dict)
    tests: list[dict[str, Any]] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    error: str | None = None
    result_path: str | None = None
    stop_reason: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def to_state(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "status": self.status,
            "functionPath": self.function_path,
            "variant": self.variant,
            "metrics": self.metrics,
            "tests": self.tests,
            "error": self.error,
            "resultPath": self.result_path,
            "stopReason": self.stop_reason,
        }


class RunManager:
    def __init__(self, default_out_dir: Path | str = "results"):
        self.default_out_dir = Path(default_out_dir)
        self._runs: dict[str, ManagedRun] = {}
        self._lock = threading.RLock()

    def start_run(
        self,
        *,
        absolute_path: str,
        variant: str,
        max_iterations: int,
        max_tests: int | None,
        mcdc_target: float,
        out_dir: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            active = [
                run
                for run in self._runs.values()
                if run.status not in FINAL_STATUSES
            ]
            if active:
                raise RuntimeError(f"Run already active: {active[0].run_id}")

            config = ExperimentConfig(
                function_path=absolute_path,
                prompt_variant=variant,
                max_iterations=max_iterations,
                max_tests=max_tests if max_tests is not None else max_iterations,
                mcdc_target=mcdc_target,
            )
            assert config.run_id is not None
            state = ManagedRun(
                run_id=config.run_id,
                function_path=absolute_path,
                variant=variant,
                out_dir=Path(out_dir) if out_dir else self.default_out_dir,
            )
            self._runs[state.run_id] = state

            thread = threading.Thread(
                target=self._run_worker,
                args=(state, config),
                name=f"covxplore-run-{state.run_id}",
                daemon=True,
            )
            thread.start()
            return state.to_state()

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            states = {run_id: run.to_state() for run_id, run in self._runs.items()}

        for path in result_paths(self.default_out_dir):
            try:
                summary = load_result(path)
            except Exception:
                continue
            run_id = summary.get("run_id")
            if run_id and run_id not in states:
                states[run_id] = summary_to_run_state(summary, path)

        return sorted(states.values(), key=lambda item: item.get("runId", ""), reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                return run.to_state()

        found = find_result(self.default_out_dir, run_id)
        if found is None:
            return None
        path, summary = found
        return summary_to_run_state(summary, path)

    def get_result(self, run_id: str) -> dict[str, Any] | None:
        found = find_result(self.default_out_dir, run_id)
        if found is None:
            return None
        return found[1]

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                if run.result_path:
                    result_path = Path(run.result_path)
                    if result_path.exists():
                        return build_report_from_summary(
                            load_result(result_path),
                            result_path,
                        )
                return build_report_from_run_state(
                    run.to_state(),
                    [event.to_dict() for event in run.events],
                )

        found = find_result(self.default_out_dir, run_id)
        if found is None:
            return None
        path, summary = found
        return build_report_from_summary(summary, path)

    def cancel(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            run.cancel_event.set()
            if run.status not in FINAL_STATUSES:
                run.status = "cancelled"
            self._append_event_locked(
                run,
                "log",
                {"level": "warning", "message": "Cancellation requested."},
            )
            return run.to_state()

    def events_since(self, run_id: str, index: int) -> tuple[list[dict[str, Any]], bool]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return [], True
            events = [
                event.to_dict()
                for event in run.events
                if event.index > index
            ]
            done = run.status in FINAL_STATUSES
            return events, done

    def _run_worker(self, state: ManagedRun, config: ExperimentConfig) -> None:
        def sink(event_type: str, payload: dict[str, Any]) -> None:
            self._append_event(state.run_id, event_type, payload)

        try:
            state.out_dir.mkdir(parents=True, exist_ok=True)
            runner = AblationRunner()
            result = runner.run_one(
                config,
                event_sink=sink,
                cancel_checker=state.cancel_event.is_set,
            )
            summary = result.to_summary_dict()
            result_path = state.out_dir / f"gen_{config.run_id}.json"
            result_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            result_state = summary_to_run_state(summary, result_path)
            with self._lock:
                state.metrics = result_state["metrics"]
                state.tests = result_state["tests"]
                state.error = summary.get("error")
                state.result_path = str(result_path)
                state.stop_reason = summary.get("stop_reason")
                if state.cancel_event.is_set():
                    state.status = "cancelled"
                elif summary.get("error"):
                    state.status = "failed"
                else:
                    state.status = "completed"
                self._append_event_locked(
                    state,
                    "result_written",
                    {"resultPath": state.result_path},
                )
        except BaseException as exc:
            with self._lock:
                state.status = "cancelled" if state.cancel_event.is_set() else "failed"
                state.error = f"{type(exc).__name__}: {exc}"
                self._append_event_locked(
                    state,
                    "run_failed",
                    {"runId": state.run_id, "error": state.error},
                )

    def _append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            self._append_event_locked(run, event_type, payload)

    def _append_event_locked(
        self,
        run: ManagedRun,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type == "run_started":
            run.status = "running"
        elif event_type == "test_completed":
            test = payload.copy()
            test.pop("suite", None)
            run.tests.append(test)
            suite = payload.get("suite") or {}
            if suite:
                run.metrics = keys_to_camel(suite)
        elif event_type == "coverage_updated":
            run.metrics = keys_to_camel(payload)
        elif event_type == "run_completed":
            run.status = "completed"
            run.metrics = keys_to_camel(payload.get("metrics") or run.metrics)
            run.stop_reason = payload.get("stopReason")
        elif event_type == "run_failed":
            run.status = "failed"
            run.error = payload.get("error")
            run.metrics = keys_to_camel(payload.get("metrics") or run.metrics)

        event = EventRecord(
            index=len(run.events),
            type=event_type,
            run_id=run.run_id,
            timestamp=utc_now_iso(),
            payload=payload,
        )
        run.events.append(event)
