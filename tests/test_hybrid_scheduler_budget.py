from __future__ import annotations

from pathlib import Path

import pytest

from covxplore.hybrid.budget import RunBudget
from covxplore.hybrid.models import Route
from covxplore.hybrid.runner import HybridGenerationConfig
from covxplore.hybrid.scheduler import LinUCBPolicy, RuleScheduler, calculate_reward


def test_budget_enforces_independent_hard_caps() -> None:
    budget = RunBudget(
        wall_time_sec=60,
        max_llm_calls=1,
        max_symbolic_attempts=1,
        max_test_executions=1,
    )
    budget.consume_llm()
    budget.consume_symbolic()
    budget.consume_execution()
    assert not budget.can_call_llm()
    assert not budget.can_attempt_symbolic()
    assert budget.stop_reason() == "execution_cap"
    with pytest.raises(RuntimeError):
        budget.consume_execution()


def test_budget_records_provider_calls_observed_inside_completion() -> None:
    budget = RunBudget(max_llm_calls=2)
    budget.consume_llm()
    budget.record_additional_llm_calls(2)
    assert budget.llm_calls == 3
    assert not budget.can_call_llm()


def test_finished_budget_elapsed_time_is_stable() -> None:
    budget = RunBudget(wall_time_sec=60)
    budget.finish()
    elapsed = budget.elapsed_sec
    assert budget.elapsed_sec == elapsed


def test_rule_scheduler_routes_complex_cpp_to_hybrid() -> None:
    scheduler = RuleScheduler()
    assert scheduler.choose({"pointer_count": 1}, set(Route)) == Route.HYBRID
    assert scheduler.choose({}, set(Route)) == Route.SYMBOLIC


def test_reward_uses_coverage_time_token_and_invalid_penalties() -> None:
    reward = calculate_reward(
        delta_statement_fraction=0.5,
        delta_branch_fraction=0.5,
        route_seconds=60,
        route_tokens=8000,
        invalid_test=False,
    )
    assert reward == pytest.approx(0.4)
    invalid = calculate_reward(
        delta_statement_fraction=0,
        delta_branch_fraction=0,
        route_seconds=0,
        route_tokens=0,
        invalid_test=True,
    )
    assert invalid == pytest.approx(-0.2)


def test_linucb_policy_round_trips_and_frozen_policy_does_not_update(tmp_path: Path) -> None:
    policy = LinUCBPolicy(alpha=1, ridge=1)
    features = {"bias": 1.0, "pointer_count": 1.0}
    policy.update(Route.HYBRID, features, 0.8)
    path = tmp_path / "policy.json"
    policy.save(path)

    loaded = LinUCBPolicy.load(path, frozen=True)
    before = loaded.to_dict()
    loaded.update(Route.LLM, features, 1.0)
    assert loaded.to_dict() == before
    assert loaded.choose(features, set(Route)) in set(Route)


def test_untrained_linucb_uses_deterministic_rule_fallback() -> None:
    policy = LinUCBPolicy()
    assert policy.choose({"pointer_count": 1.0}, set(Route)) == Route.HYBRID
    assert policy.choose({}, set(Route)) == Route.SYMBOLIC


def test_policy_freeze_writes_immutable_artifact(tmp_path: Path) -> None:
    collected = LinUCBPolicy()
    collected.update(Route.SYMBOLIC, {"bias": 1.0}, 0.5)
    path = tmp_path / "frozen-policy.json"
    collected.freeze_to(path)

    frozen = LinUCBPolicy.load(path)
    assert frozen.frozen
    before = frozen.to_dict()
    frozen.update(Route.LLM, {"bias": 1.0}, 1.0)
    assert frozen.to_dict() == before


def test_frozen_runner_configuration_rejects_missing_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        HybridGenerationConfig(
            function_path="D:/src/f.cpp/f()",
            scheduler_mode="frozen",
            policy_path=tmp_path / "missing.json",
        )


def test_collection_requires_a_persisted_policy_path() -> None:
    with pytest.raises(ValueError, match="collect scheduler requires policy_path"):
        HybridGenerationConfig(
            function_path="D:/project/f.cpp/f()",
            scheduler_mode="collect",
        )
