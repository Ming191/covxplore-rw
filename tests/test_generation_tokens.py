from types import SimpleNamespace

from covxplore.generation.tokens import (
    TokenLedger,
    TokenTotals,
    reconcile_suite_tokens,
    totals_from_crew,
)
from covxplore.types import TestResult, TestSuite


class _CrewInst:
    def __init__(self, prompt=0, completion=0):
        self._crew = SimpleNamespace(
            usage_metrics=SimpleNamespace(
                prompt_tokens=prompt, completion_tokens=completion
            )
        )

    def crew(self):
        return self._crew


def _result(name, status="PASSED", in_tokens=0, out_tokens=0):
    return TestResult(
        test_name=name,
        test_body="void test() {}",
        status=status,
        token_input=in_tokens,
        token_output=out_tokens,
    )


def test_token_totals_use_crew_usage_metrics():
    totals = totals_from_crew(_CrewInst(10, 20))
    assert totals == TokenTotals(prompt=10, completion=20)


def test_token_totals_zero_when_crew_usage_metrics_zero():
    totals = totals_from_crew(_CrewInst(0, 0))
    assert totals == TokenTotals()


def test_token_ledger_accumulates_crew_totals():
    ledger = TokenLedger()
    ledger.record_crew(_CrewInst(10, 20))
    ledger.record_crew(_CrewInst(3, 4))

    assert ledger.crew == TokenTotals(prompt=13, completion=24)


def test_token_ledger_crew_beats_trace(monkeypatch):
    monkeypatch.setattr(
        "covxplore.observability.fetch_trace_token_totals",
        lambda trace_id: TokenTotals(prompt=100, completion=100),
    )
    suite = TestSuite(function_path="f")
    suite.tests = [_result("pass")]
    ledger = TokenLedger()
    ledger.record_crew(_CrewInst(1, 2))
    ledger.record_trace("trace")

    totals = ledger.finalize_suite(suite)

    assert totals == TokenTotals(prompt=1, completion=2)
    assert ledger.source == "crew"
    assert (suite.tests[0].token_input, suite.tests[0].token_output) == (1, 2)


def test_token_ledger_trace_fallback(monkeypatch):
    monkeypatch.setattr(
        "covxplore.observability.fetch_trace_token_totals",
        lambda trace_id: TokenTotals(prompt=5, completion=7),
    )
    suite = TestSuite(function_path="f")
    suite.tests = [_result("pass")]
    ledger = TokenLedger()
    ledger.record_trace("trace")

    totals = ledger.finalize_suite(suite)

    assert totals == TokenTotals(prompt=5, completion=7)
    assert ledger.source == "trace"


def test_reconcile_suite_tokens_distributes_to_eligible_statuses_with_remainder():
    suite = TestSuite(function_path="f")
    suite.tests = [
        _result("pass", "PASSED"),
        _result("compile", "COMPILE_ERROR"),
        _result("runtime", "RUNTIME_ERROR"),
    ]

    reconcile_suite_tokens(suite, TokenTotals(prompt=5, completion=7))

    assert [(t.token_input, t.token_output) for t in suite.tests] == [
        (2, 3),
        (0, 0),
        (3, 4),
    ]


def test_reconcile_suite_tokens_does_not_overwrite_existing_totals():
    suite = TestSuite(function_path="f")
    suite.tests = [_result("pass", in_tokens=1, out_tokens=0)]

    reconcile_suite_tokens(suite, TokenTotals(prompt=10, completion=20))

    assert suite.tests[0].token_input == 1
    assert suite.tests[0].token_output == 0
