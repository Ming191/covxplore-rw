from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from covxplore.status import TestStatus


@dataclass
class TokenTotals:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def add(self, other: "TokenTotals") -> None:
        self.prompt += other.prompt
        self.completion += other.completion

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "total": self.total,
        }


@dataclass
class TokenLedger:
    crew: TokenTotals = field(default_factory=TokenTotals)
    trace: TokenTotals = field(default_factory=TokenTotals)
    chosen: TokenTotals = field(default_factory=TokenTotals)
    source: Literal["none", "crew", "trace"] = "none"

    def record_crew(self, crew_inst) -> None:
        self.crew.add(totals_from_crew(crew_inst))

    def record_trace(self, trace_id: str | None) -> None:
        from covxplore.observability import fetch_trace_token_totals

        self.trace.add(fetch_trace_token_totals(trace_id))

    def finalize_suite(self, suite) -> TokenTotals:
        if self.crew.total > 0:
            self.chosen = TokenTotals(self.crew.prompt, self.crew.completion)
            self.source = "crew"
        elif self.trace.total > 0:
            self.chosen = TokenTotals(self.trace.prompt, self.trace.completion)
            self.source = "trace"
        else:
            self.chosen = TokenTotals()
            self.source = "none"
        reconcile_suite_tokens(suite, self.chosen)
        return self.chosen

    def to_dict(self) -> dict:
        return {
            "crew": self.crew.to_dict(),
            "trace": self.trace.to_dict(),
            "chosen": self.chosen.to_dict(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "TokenLedger":
        data = data or {}
        ledger = cls()
        ledger.crew = _totals_from_dict(data.get("crew"))
        ledger.trace = _totals_from_dict(data.get("trace"))
        ledger.chosen = _totals_from_dict(data.get("chosen"))
        ledger.source = data.get("source") or "none"
        return ledger


def _totals_from_dict(data: dict | None) -> TokenTotals:
    data = data or {}
    return TokenTotals(
        prompt=int(data.get("prompt") or 0),
        completion=int(data.get("completion") or 0),
    )


def totals_from_crew(crew_inst) -> TokenTotals:
    try:
        if crew_inst is None:
            return TokenTotals()
        totals = totals_from_usage_metrics(getattr(crew_inst, "usage_metrics", None))
        if totals.total > 0:
            return totals
        if hasattr(crew_inst, "calculate_usage_metrics"):
            totals = totals_from_usage_metrics(crew_inst.calculate_usage_metrics())
            if totals.total > 0:
                return totals
        if hasattr(crew_inst, "crew"):
            return totals_from_usage_metrics(crew_inst.crew().usage_metrics)
        return TokenTotals()
    except Exception:
        return TokenTotals()


def totals_from_usage_metrics(metrics) -> TokenTotals:
    if not metrics:
        return TokenTotals()
    return TokenTotals(
        prompt=int(getattr(metrics, "prompt_tokens", 0) or 0),
        completion=int(getattr(metrics, "completion_tokens", 0) or 0),
    )


def reconcile_suite_tokens(suite, totals: TokenTotals) -> None:
    if suite.total_input_tokens > 0 or suite.total_output_tokens > 0:
        return
    if totals.total == 0:
        return
    eligible = [
        t
        for t in suite.tests
        if t.status in {TestStatus.PASSED.value, TestStatus.RUNTIME_ERROR.value}
    ]
    if not eligible:
        return

    n = len(eligible)
    base_in, rem_in = divmod(totals.prompt, n)
    base_out, rem_out = divmod(totals.completion, n)
    for i, t in enumerate(eligible):
        t.token_input = base_in + (rem_in if i == n - 1 else 0)
        t.token_output = base_out + (rem_out if i == n - 1 else 0)
