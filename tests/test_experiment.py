"""Tests for covxplore.experiment — flat_row projection, no network, no secrets.

Caveat: importing covxplore.generator triggers imports of crewai, config, etc.
If import is too heavy in CI without uv sync, skip this module.
"""

import pytest

from covxplore.experiment import flat_row
from covxplore.generator import GenerationConfig, GenerationResult, StopReason
from covxplore.types import (
    ConditionKey,
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TestSuite,
)
from covxplore.status import TestStatus


def _make_result(
    test_name: str,
    status: str = "PASSED",
    is_redundant: bool = False,
    node_id: int = 1,
    polarity: bool = True,
) -> TestResult:
    return TestResult(
        test_name=test_name,
        test_body="void test() {}",
        status=status,
        is_redundant=is_redundant,
        condition_trace=[
            ConditionTraceEntry(
                node_id=node_id,
                condition=f"c{node_id}",
                true_branch_visited=polarity,
                false_branch_visited=not polarity,
            ),
        ],
        mcdc_coverage=CoverageDetail(visited=1, total=2, progress=0.5),
        statement_coverage=CoverageDetail(visited=3, total=5, progress=0.6),
        branch_coverage=CoverageDetail(visited=2, total=4, progress=0.5),
        token_input=100,
        token_output=50,
        elapsed_ms=150.0,
    )


class TestFlatRow:
    def test_empty_suite(self):
        """flat_row on a suite with no tests."""
        config = GenerationConfig(
            function_path="/a/b.cpp::foo(int)",
            prompt_variant="full",
            max_batches=10,
            mcdc_target=1.0,
        )
        suite = TestSuite(function_path=config.function_path)
        result = GenerationResult(
            config=config, suite=suite, stop_reason="agent_done",
        )

        row = flat_row(result)

        assert row["run_id"] is not None
        assert row["function_path"] == "/a/b.cpp::foo(int)"
        assert row["prompt_variant"] == "full"
        assert row["stop_reason"] == "agent_done"
        assert row["accepted_test_count"] == 0
        assert row["passing_test_count"] == 0
        assert row["rejected_candidate_count"] == 0
        assert row["candidate_count"] == 0
        assert row["batches_used"] == 0
        assert row["error"] == ""
        assert row["total_tokens"] == 0

    def test_with_passing_and_failing_tests(self):
        """flat_row counts passing and candidates correctly."""
        config = GenerationConfig(
            function_path="/x/y.cpp::bar()",
            prompt_variant="baseline",
            max_batches=5,
            mcdc_target=1.0,
        )
        suite = TestSuite(function_path=config.function_path)
        t1 = _make_result("t1", status="PASSED", is_redundant=False)
        t1.accepted_order = 1
        t2 = _make_result("t2", status="PASSED", is_redundant=True)
        t2.accepted_order = 2
        t3 = _make_result("t3", status="FAILED", is_redundant=False)
        t3.accepted_order = 3
        t4 = _make_result("t4", status="RUNTIME_ERROR", is_redundant=False)
        t4.accepted_order = 4
        suite.tests = [t1, t2, t3, t4]
        suite.rejected_tests = [_make_result("reject")]
        suite.batch_count = 2

        result = GenerationResult(
            config=config, suite=suite, stop_reason="max_batches",
        )
        row = flat_row(result)

        assert row["accepted_test_count"] == 4
        assert row["passing_test_count"] == 2  # t1, t2 are PASSED
        assert row["rejected_candidate_count"] == 1
        assert row["candidate_count"] == 5
        assert row["batches_used"] == 2
        assert row["stop_reason"] == "max_batches"

    def test_error_message_propagated(self):
        config = GenerationConfig(
            function_path="/e/f.cpp::baz()",
            prompt_variant="full",
        )
        suite = TestSuite(function_path=config.function_path)
        result = GenerationResult(
            config=config,
            suite=suite,
            stop_reason="error",
            error_message="Something went wrong",
        )
        row = flat_row(result)
        assert row["stop_reason"] == "error"
        assert row["error"] == "Something went wrong"

    def test_none_error_becomes_empty_string(self):
        config = GenerationConfig(
            function_path="/g/h.cpp::qux()",
            prompt_variant="no_cot",
        )
        suite = TestSuite(function_path=config.function_path)
        result = GenerationResult(
            config=config, suite=suite, stop_reason="coverage_target", error_message=None,
        )
        row = flat_row(result)
        assert row["error"] == ""

    def test_run_id_auto_generated(self):
        """GenerationConfig.__post_init__ generates a run_id."""
        config = GenerationConfig(
            function_path="/p/q.cpp::myFunc(int)",
            prompt_variant="full",
            max_batches=3,
            mcdc_target=0.8,
        )
        assert config.run_id is not None
        assert "_myFunc" in config.run_id

    def test_coverage_metrics_in_flat_row(self):
        config = GenerationConfig(
            function_path="/r/s.cpp::coverMe(bool)",
            prompt_variant="full",
        )
        suite = TestSuite(function_path=config.function_path)
        suite.coverage.total_mcdc_pairs = 2
        suite.coverage._covered_keys = {ConditionKey(1, True), ConditionKey(1, False)}
        suite.coverage._total_statements = 5
        suite.coverage._total_branches = 4
        result = GenerationResult(
            config=config, suite=suite, stop_reason="coverage_target",
        )
        row = flat_row(result)

        assert row["mcdc_coverage_pct"] == 1.0
        assert row["covered_mcdc_pairs"] == 2
        assert row["total_mcdc_pairs"] == 2
        assert row["total_statements"] == 5
        assert row["total_branches"] == 4

    @pytest.mark.parametrize("stop_reason", [
        "max_batches",
        "coverage_target",
        "redundant_streak",
        "agent_done",
        "guardrail_incomplete",
        "error",
    ])
    def test_all_stop_reasons_in_row(self, stop_reason):
        config = GenerationConfig(
            function_path="/z.cpp::f()",
            prompt_variant="baseline",
        )
        suite = TestSuite(function_path=config.function_path)
        result = GenerationResult(
            config=config, suite=suite, stop_reason=stop_reason,
        )
        row = flat_row(result)
        assert row["stop_reason"] == stop_reason

    def test_tracing_url_in_flat_row(self):
        config = GenerationConfig(
            function_path="/trace.cpp::f()",
            prompt_variant="full",
        )
        suite = TestSuite(function_path=config.function_path)
        result = GenerationResult(
            config=config,
            suite=suite,
            stop_reason="coverage_target",
            tracing_url="http://localhost:3000/project/covxplore/traces/abc",
        )

        row = flat_row(result)

        assert row["tracing_url"] == "http://localhost:3000/project/covxplore/traces/abc"
