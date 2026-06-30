from covxplore.generation.tokens import TokenTotals
from covxplore.observability import extract_trace_id, token_totals_from_trace


def test_extract_trace_id_from_langfuse_url():
    trace_id = "fd09de7a96da11d51f477d4aa31dcb18"

    assert (
        extract_trace_id(f"http://localhost:3000/project/covxplore/traces/{trace_id}")
        == trace_id
    )


def test_extract_trace_id_returns_none_for_missing_url():
    assert extract_trace_id(None) is None
    assert extract_trace_id("http://localhost:3000/project/covxplore/traces/not-a-trace") is None


def test_token_totals_from_trace_sums_observation_usage_shapes():
    trace = {
        "observations": [
            {"usage": {"input": 10, "output": 20}},
            {"usageDetails": {"input": 3, "output": 4}},
            {"inputUsage": 5, "outputUsage": 6},
            {"usage": {"promptTokens": 7, "completionTokens": 8}},
        ]
    }

    assert token_totals_from_trace(trace) == TokenTotals(prompt=25, completion=38)
