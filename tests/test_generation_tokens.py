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
            usage_metrics=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
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


def test_token_ledger_accumulates_crew_totals():
    ledger = TokenLedger()
    ledger.record_crew(_CrewInst(10, 20))
    ledger.record_crew(_CrewInst(3, 4))
    assert ledger.crew == TokenTotals(prompt=13, completion=24)


def test_token_totals_calculate_usage_after_interrupted_kickoff():
    class Crew:
        usage_metrics = None

        def calculate_usage_metrics(self):
            return SimpleNamespace(prompt_tokens=11, completion_tokens=22)

    assert totals_from_crew(Crew()) == TokenTotals(prompt=11, completion=22)


def test_finalize_uses_crew_usage():
    suite = TestSuite(function_path="f")
    suite.tests = [_result("pass")]
    ledger = TokenLedger()
    ledger.record_crew(_CrewInst(1, 2))
    assert ledger.finalize_suite(suite) == TokenTotals(prompt=1, completion=2)
    assert ledger.source == "crew"
    assert (suite.tests[0].token_input, suite.tests[0].token_output) == (1, 2)


def test_reconcile_suite_tokens_distributes_to_eligible_statuses_with_remainder():
    suite = TestSuite(function_path="f")
    suite.tests = [
        _result("pass", "PASSED"),
        _result("compile", "COMPILE_ERROR"),
        _result("runtime", "RUNTIME_ERROR"),
    ]
    reconcile_suite_tokens(suite, TokenTotals(prompt=5, completion=7))
    assert [(test.token_input, test.token_output) for test in suite.tests] == [
        (2, 3),
        (0, 0),
        (3, 4),
    ]


def test_reconcile_does_not_overwrite_existing_totals():
    suite = TestSuite(function_path="f")
    suite.tests = [_result("pass", in_tokens=1)]
    reconcile_suite_tokens(suite, TokenTotals(prompt=10, completion=20))
    assert suite.tests[0].token_input == 1
