from types import SimpleNamespace

from covxplore.generation.tokens import (
    TokenTotals,
    choose_run_token_totals,
    reconcile_suite_tokens,
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


def _logger(prompt=0, completion=0):
    return SimpleNamespace(
        interactions=[{"usage": {"prompt_tokens": prompt, "completion_tokens": completion}}]
    )


def _result(name, status="PASSED", in_tokens=0, out_tokens=0):
    return TestResult(
        test_name=name,
        test_body="void test() {}",
        status=status,
        token_input=in_tokens,
        token_output=out_tokens,
    )


def test_token_totals_crew_precedence_over_logger():
    totals = choose_run_token_totals(_CrewInst(10, 20), _logger(100, 200))
    assert totals == TokenTotals(prompt=10, completion=20)


def test_token_totals_fallback_to_logger_when_crew_zero():
    totals = choose_run_token_totals(_CrewInst(0, 0), _logger(100, 200))
    assert totals == TokenTotals(prompt=100, completion=200)


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
