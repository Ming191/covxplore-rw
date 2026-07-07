import json
from dataclasses import asdict

import pytest

from covxplore.generation.batch import merge_batch_results
from covxplore.generation.runtime_session import TestSuiteCodec
from covxplore.generation.stop_reasons import StopPolicy
from covxplore.types import (
    ConditionKey,
    ConditionTraceEntry,
    CoverageDetail,
    TestResult,
    TestSuite,
    UnvisitedBranch,
    UnvisitedStatement,
)


def _result(name: str, node_id: int, polarity: bool) -> TestResult:
    return TestResult(
        test_name=name,
        test_body=f"{name}();",
        status="PASSED",
        statement_coverage=CoverageDetail(visited=1, total=2, progress=0.5),
        branch_coverage=CoverageDetail(visited=1, total=2, progress=0.5),
        condition_trace=[
            ConditionTraceEntry(
                node_id=node_id,
                condition=f"c{node_id}",
                true_branch_visited=polarity,
                false_branch_visited=not polarity,
                line_in_function=node_id,
            )
        ],
        unvisited_statements=[
            UnvisitedStatement(node_id=20, statement="return 0;", line_in_function=4)
        ],
        unvisited_branches=[
            UnvisitedBranch(
                node_id=30,
                condition="x",
                true_visited=False,
                false_visited=True,
                line_in_function=5,
            )
        ],
        accepted_order=1,
    )


def _suite() -> TestSuite:
    suite = TestSuite(function_path="/f.cpp::f()")
    suite.add_result(_result("first", 1, True), min_suite_size=0)
    suite.rejected_tests.append(_result("reject", 2, False))
    suite.record_batch(False)
    suite.fail_streak = 1
    suite.coverage.total_mcdc_pairs = 4
    suite.coverage.consecutive_redundant = 2
    return suite


def test_codec_payload_is_json_safe():
    payload = TestSuiteCodec().dump_suite(_suite())

    json.dumps(payload)


def test_codec_round_trip_preserves_full_state():
    codec = TestSuiteCodec()
    suite = _suite()

    restored = codec.load_suite(codec.dump_suite(suite))

    assert restored.function_path == suite.function_path
    assert restored.batch_count == suite.batch_count
    assert restored.fail_streak == suite.fail_streak
    assert [t.model_dump(mode="json") for t in restored.tests] == [
        t.model_dump(mode="json") for t in suite.tests
    ]
    assert [t.model_dump(mode="json") for t in restored.rejected_tests] == [
        t.model_dump(mode="json") for t in suite.rejected_tests
    ]
    assert restored.coverage._covered_keys == suite.coverage._covered_keys
    assert restored.coverage._condition_id_to_text == suite.coverage._condition_id_to_text
    assert restored.coverage._condition_id_to_line == suite.coverage._condition_id_to_line
    assert restored.coverage._all_conditions == suite.coverage._all_conditions
    assert restored.coverage._cumulative_uncovered_stmt_ids == suite.coverage._cumulative_uncovered_stmt_ids
    assert restored.coverage._cumulative_uncovered_branch_keys == suite.coverage._cumulative_uncovered_branch_keys
    assert restored.coverage._total_statements == suite.coverage._total_statements
    assert restored.coverage._total_branches == suite.coverage._total_branches
    assert restored.coverage.total_mcdc_pairs == suite.coverage.total_mcdc_pairs
    assert restored.coverage.consecutive_redundant == suite.coverage.consecutive_redundant


def test_codec_round_trip_preserves_metrics_gap_input_and_unvisited_summary():
    codec = TestSuiteCodec()
    suite = _suite()
    restored = codec.load_suite(codec.dump_suite(suite))

    assert asdict(restored.coverage.metrics(restored.tests)) == asdict(suite.coverage.metrics(suite.tests))
    assert restored.coverage.unvisited_summary() == suite.coverage.unvisited_summary()
    assert asdict(restored.coverage.gap_input(restored.tests, restored.batch_count)) == asdict(suite.coverage.gap_input(suite.tests, suite.batch_count))


def test_codec_restore_preserves_merge_batch_results_behavior():
    codec = TestSuiteCodec()
    suite = _suite()
    restored = codec.load_suite(codec.dump_suite(suite))
    dupe = _result("dupe", 1, True)

    summary = merge_batch_results(restored, [dupe], min_suite_size=0)

    assert summary.accepted == []
    assert [r.test_name for r in summary.rejected_redundant] == ["dupe"]


def test_codec_restore_preserves_stop_reason_inputs():
    policy = StopPolicy(max_batches=10, mcdc_target=1.0, redundant_streak_limit=2, fail_streak_limit=3)
    restored = TestSuiteCodec().load_suite(TestSuiteCodec().dump_suite(_suite()))

    assert policy.terminal_reason(restored) == "redundant_streak"


def test_codec_preserves_none_for_cumulative_uncovered_sets():
    suite = TestSuite(function_path="/empty.cpp::f()")
    restored = TestSuiteCodec().load_suite(TestSuiteCodec().dump_suite(suite))

    assert restored.coverage._cumulative_uncovered_stmt_ids is None
    assert restored.coverage._cumulative_uncovered_branch_keys is None


def test_codec_rejects_invalid_version():
    payload = TestSuiteCodec().dump_suite(_suite())
    payload["version"] = 999

    with pytest.raises(ValueError, match="Unsupported"):
        TestSuiteCodec().load_suite(payload)


def test_codec_rejects_invalid_private_map_entries():
    payload = TestSuiteCodec().dump_suite(_suite())
    payload["coverage"]["_stmt_node_info"] = [{"statement": "x"}]

    with pytest.raises(ValueError, match="node_id"):
        TestSuiteCodec().load_suite(payload)
