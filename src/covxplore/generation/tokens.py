from __future__ import annotations

from dataclasses import dataclass

from covxplore.status import TestStatus


@dataclass
class TokenTotals:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


def totals_from_crew(crew_inst) -> TokenTotals:
    try:
        metrics = crew_inst.crew().usage_metrics if crew_inst is not None else None
        if not metrics:
            return TokenTotals()
        return TokenTotals(
            prompt=int(getattr(metrics, "prompt_tokens", 0) or 0),
            completion=int(getattr(metrics, "completion_tokens", 0) or 0),
        )
    except Exception:
        return TokenTotals()


def choose_run_token_totals(crew_inst) -> TokenTotals:
    return totals_from_crew(crew_inst)


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
