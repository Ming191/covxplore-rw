import json

import pytest

from covxplore.generation.runtime_session import TestSuiteCodec
from covxplore.types import CoverageDetail, TestResult, TestSuite, UnvisitedBranch, UnvisitedStatement


def _suite():
    suite = TestSuite(function_path="/f.cpp::f()")
    suite.add_result(TestResult(
        test_name="first", test_body="f();", status="PASSED",
        statement_coverage=CoverageDetail(visited=1, total=2),
        branch_coverage=CoverageDetail(visited=1, total=2),
        unvisited_statements=[UnvisitedStatement(node_id=2, statement="return 0;")],
        unvisited_branches=[UnvisitedBranch(node_id=3, condition="x", true_visited=False, false_visited=True)],
    ))
    suite.record_batch(False)
    return suite


def test_codec_payload_is_json_safe_and_round_trips_structural_state():
    codec = TestSuiteCodec()
    payload = codec.dump_suite(_suite())
    json.dumps(payload)
    restored = codec.load_suite(payload)
    assert restored.function_path == "/f.cpp::f()"
    assert restored.coverage._cumulative_uncovered_stmt_ids == {2}
    assert restored.coverage._cumulative_uncovered_branch_keys == {(3, True)}


def test_codec_rejects_invalid_version():
    payload = TestSuiteCodec().dump_suite(_suite())
    payload["version"] = 999
    with pytest.raises(ValueError, match="Unsupported"):
        TestSuiteCodec().load_suite(payload)
