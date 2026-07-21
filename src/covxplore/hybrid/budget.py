from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RunBudget:
    wall_time_sec: float = 30 * 60
    max_llm_calls: int = 15
    max_symbolic_attempts: int = 30
    max_test_executions: int = 30
    started_at: float = field(default_factory=time.monotonic)
    llm_calls: int = 0
    symbolic_attempts: int = 0
    test_executions: int = 0
    finished_at: float | None = None

    @property
    def elapsed_sec(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def finish(self) -> None:
        if self.finished_at is None:
            self.finished_at = time.monotonic()

    @property
    def wall_time_exhausted(self) -> bool:
        return self.elapsed_sec >= self.wall_time_sec

    def can_call_llm(self) -> bool:
        return not self.wall_time_exhausted and self.llm_calls < self.max_llm_calls

    def can_attempt_symbolic(self) -> bool:
        return (
            not self.wall_time_exhausted
            and self.symbolic_attempts < self.max_symbolic_attempts
        )

    def can_execute(self) -> bool:
        return (
            not self.wall_time_exhausted
            and self.test_executions < self.max_test_executions
        )

    def consume_llm(self) -> None:
        if not self.can_call_llm():
            raise RuntimeError("LLM call budget exhausted")
        self.llm_calls += 1

    def record_additional_llm_calls(self, count: int) -> None:
        """Account for provider calls observed inside one completion request."""
        if count > 0:
            self.llm_calls += count

    def consume_symbolic(self) -> None:
        if not self.can_attempt_symbolic():
            raise RuntimeError("symbolic attempt budget exhausted")
        self.symbolic_attempts += 1

    def consume_execution(self) -> None:
        if not self.can_execute():
            raise RuntimeError("test execution budget exhausted")
        self.test_executions += 1

    def stop_reason(self) -> str | None:
        if self.wall_time_exhausted:
            return "wall_time_budget"
        if self.test_executions >= self.max_test_executions:
            return "execution_cap"
        return None

    def remaining_fractions(self) -> dict[str, float]:
        def fraction(used: int, maximum: int) -> float:
            return 0.0 if maximum <= 0 else max(0.0, (maximum - used) / maximum)

        return {
            "remaining_wall_time": max(
                0.0, (self.wall_time_sec - self.elapsed_sec) / self.wall_time_sec
            )
            if self.wall_time_sec > 0
            else 0.0,
            "remaining_llm_calls": fraction(self.llm_calls, self.max_llm_calls),
            "remaining_symbolic_attempts": fraction(
                self.symbolic_attempts, self.max_symbolic_attempts
            ),
            "remaining_test_executions": fraction(
                self.test_executions, self.max_test_executions
            ),
        }
