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


def totals_from_llm_logger(llm_logger) -> TokenTotals:
    prompt_tokens = 0
    completion_tokens = 0
    for interaction in getattr(llm_logger, "interactions", []) or []:
        usage = interaction.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
    return TokenTotals(prompt=prompt_tokens, completion=completion_tokens)


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


def choose_run_token_totals(crew_inst, llm_logger) -> TokenTotals:
    crew_totals = totals_from_crew(crew_inst)
    if crew_totals.total > 0:
        return crew_totals
    return totals_from_llm_logger(llm_logger)


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
