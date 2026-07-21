from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from covxplore.api_client import AkaUTClient, AkaUTError
from covxplore.hybrid.budget import RunBudget
from covxplore.hybrid.coverage import ExactCoverageState
from covxplore.hybrid.llm_completion import IntentCompleter, IntentCompletionError
from covxplore.hybrid.models import (
    CONTRACT_VERSION,
    CoverageModel,
    Route,
    RouteDecision,
    SolverStatus,
    SymbolicAttemptRequest,
    SymbolicAttemptResponse,
    Target,
    TestAttemptRecord,
    TestIntent,
    ExecutionResult,
)
from covxplore.hybrid.scheduler import (
    LinUCBPolicy,
    RuleScheduler,
    calculate_reward,
)
from covxplore.hybrid.selector import SuccessfulSeed, TargetSelection, TargetSelector


Strategy = Literal["llm", "symbolic", "hybrid"]
SchedulerMode = Literal["rule", "collect", "frozen"]


@dataclass
class HybridGenerationConfig:
    function_path: str
    strategy: Strategy = "hybrid"
    statement_target: float = 1.0
    branch_target: float = 1.0
    scheduler_mode: SchedulerMode = "rule"
    policy_path: Path | None = None
    wall_time_minutes: float = 30.0
    max_llm_calls: int = 15
    max_symbolic_attempts: int = 30
    max_test_executions: int = 30
    random_seed: int = 42
    use_nearest_seed: bool = True
    use_divergence_repair: bool = True
    use_symbolic_partial: bool = True
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        if isinstance(self.policy_path, str):
            self.policy_path = Path(self.policy_path)
        if self.strategy not in {"llm", "symbolic", "hybrid"}:
            raise ValueError("strategy must be llm, symbolic, or hybrid")
        if self.scheduler_mode not in {"rule", "collect", "frozen"}:
            raise ValueError("scheduler_mode must be rule, collect, or frozen")
        for name, value in (
            ("statement_target", self.statement_target),
            ("branch_target", self.branch_target),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.scheduler_mode == "frozen" and self.policy_path is None:
            raise ValueError("frozen scheduler requires policy_path")
        if self.scheduler_mode == "collect" and self.policy_path is None:
            raise ValueError("collect scheduler requires policy_path")
        if (
            self.scheduler_mode == "frozen"
            and self.policy_path is not None
            and not self.policy_path.is_file()
        ):
            raise ValueError(f"frozen policy does not exist: {self.policy_path}")
        if self.wall_time_minutes <= 0:
            raise ValueError("wall_time_minutes must be positive")
        if self.max_test_executions <= 0:
            raise ValueError("max_test_executions must be positive")
        if self.max_llm_calls < 0 or self.max_symbolic_attempts < 0:
            raise ValueError("route budgets must not be negative")


@dataclass
class HybridGenerationResult:
    config: HybridGenerationConfig
    coverage_model: CoverageModel
    state: ExactCoverageState
    stop_reason: str
    attempts: list[TestAttemptRecord]
    route_decisions: list[RouteDecision]
    llm_interactions: list[dict]
    budget: RunBudget
    error: str | None = None
    policy_metadata: dict | None = None

    def to_summary_dict(self) -> dict:
        metrics = self.state.summary()
        passing = sum(
            1
            for item in self.attempts
            if item.execution is not None and item.execution.passed
        )
        executed = sum(item.execution is not None for item in self.attempts)
        redundant = sum(1 for item in self.attempts if item.redundant)
        input_tokens = sum(item.input_tokens for item in self.attempts)
        output_tokens = sum(item.output_tokens for item in self.attempts)
        statement_auc, branch_auc = self._coverage_time_auc()
        symbolic_metrics = self._symbolic_metrics()
        return {
            "contract_version": CONTRACT_VERSION,
            "run_id": self.config.run_id,
            "function_path": self.config.function_path,
            "strategy": self.config.strategy,
            "scheduler": self.config.scheduler_mode,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "coverage_model_hash": self.coverage_model.model_hash,
            "metrics": {
                **metrics,
                "redundancy_rate": redundant / executed if executed else 0.0,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "elapsed_sec": round(self.budget.elapsed_sec, 3),
                "iteration_used": len(self.attempts),
                "num_testcases": executed,
                "num_passing": passing,
                "num_redundant": redundant,
                "llm_calls": self.budget.llm_calls,
                "symbolic_attempts": self.budget.symbolic_attempts,
                "test_executions": self.budget.test_executions,
                "statement_coverage_time_auc": statement_auc,
                "branch_coverage_time_auc": branch_auc,
                **symbolic_metrics,
            },
            "targets": {
                "statement": self.config.statement_target,
                "branch": self.config.branch_target,
            },
            "random_seed": self.config.random_seed,
            "ablation": {
                "use_nearest_seed": self.config.use_nearest_seed,
                "use_divergence_repair": self.config.use_divergence_repair,
                "use_symbolic_partial": self.config.use_symbolic_partial,
            },
            "budget": {
                "wall_time_minutes": self.config.wall_time_minutes,
                "max_llm_calls": self.config.max_llm_calls,
                "max_symbolic_attempts": self.config.max_symbolic_attempts,
                "max_test_executions": self.config.max_test_executions,
            },
            "coverage_model": self.coverage_model.wire_dict(),
            "attempts": [item.wire_dict() for item in self.attempts],
            "route_decisions": [item.wire_dict() for item in self.route_decisions],
            "policy": self.policy_metadata,
            "llm_interactions": self.llm_interactions,
        }

    def _symbolic_metrics(self) -> dict[str, float | int]:
        status_counts = {
            "SOLVED": 0,
            "UNSAT": 0,
            "UNSUPPORTED": 0,
            "TIMEOUT": 0,
            "ERROR": 0,
        }
        z3_calls = 0
        z3_bindings = 0
        z3_elapsed_ms = 0.0
        symbolic_elapsed_ms = 0.0
        partial_targets: set[str] = set()
        assisted_passing_targets: set[str] = set()
        assisted_tests = 0
        assisted_passing = 0
        assisted_useful = 0
        target_reached = 0
        assisted_new_statements = 0
        assisted_new_branches = 0
        symbolic_new_statements = 0
        symbolic_new_branches = 0
        hybrid_new_statements = 0
        hybrid_new_branches = 0

        for attempt in self.attempts:
            symbolic = attempt.symbolic
            if symbolic is not None:
                symbolic_elapsed_ms += max(0.0, symbolic.elapsed_ms)
                z3_elapsed_ms += max(0.0, symbolic.solver_elapsed_ms)
                z3_calls += max(0, symbolic.solver_calls)
                z3_bindings += max(0, symbolic.model_binding_count)
                counts = symbolic.solver_status_counts
                if counts:
                    for status in status_counts:
                        status_counts[status] += max(0, int(counts.get(status, 0)))
                elif symbolic.solver_status is not None:
                    status = symbolic.solver_status.value
                    if status in status_counts:
                        status_counts[status] += 1
                        if symbolic.solver_calls == 0:
                            z3_calls += 1
                if symbolic.status == SolverStatus.PARTIAL:
                    partial_targets.add(attempt.target.key)

            if not self._is_symbolic_assisted(attempt):
                continue
            execution = attempt.execution
            if execution is None:
                continue
            assisted_tests += 1
            if execution.passed:
                assisted_passing += 1
                assisted_passing_targets.add(attempt.target.key)
            if execution.target_reached:
                target_reached += 1
            useful = bool(attempt.new_statement_keys or attempt.new_branch_keys)
            if useful:
                assisted_useful += 1
            assisted_new_statements += len(attempt.new_statement_keys)
            assisted_new_branches += len(attempt.new_branch_keys)
            if attempt.route == Route.SYMBOLIC:
                symbolic_new_statements += len(attempt.new_statement_keys)
                symbolic_new_branches += len(attempt.new_branch_keys)
            elif attempt.route == Route.HYBRID:
                hybrid_new_statements += len(attempt.new_statement_keys)
                hybrid_new_branches += len(attempt.new_branch_keys)

        partial_to_passing = len(partial_targets & assisted_passing_targets)
        return {
            "z3_calls": z3_calls,
            "z3_solved": status_counts["SOLVED"],
            "z3_unsat": status_counts["UNSAT"],
            "z3_unsupported": status_counts["UNSUPPORTED"],
            "z3_timeout": status_counts["TIMEOUT"],
            "z3_error": status_counts["ERROR"],
            "z3_solve_rate": (
                status_counts["SOLVED"] / z3_calls if z3_calls else 0.0
            ),
            "z3_time_used": round(z3_elapsed_ms / 1000.0, 3),
            "symbolic_time_used": round(symbolic_elapsed_ms / 1000.0, 3),
            "z3_binding_count": z3_bindings,
            "symbolic_assisted_tests": assisted_tests,
            "symbolic_assisted_passing": assisted_passing,
            "symbolic_target_reached": target_reached,
            "symbolic_useful_attempts": assisted_useful,
            "symbolic_useful_rate": (
                assisted_useful / assisted_tests if assisted_tests else 0.0
            ),
            "symbolic_assisted_new_statements": assisted_new_statements,
            "symbolic_assisted_new_branches": assisted_new_branches,
            "symbolic_new_statements": symbolic_new_statements,
            "symbolic_new_branches": symbolic_new_branches,
            "hybrid_assisted_new_statements": hybrid_new_statements,
            "hybrid_assisted_new_branches": hybrid_new_branches,
            "partial_to_passing_rate": (
                partial_to_passing / len(partial_targets)
                if partial_targets
                else 0.0
            ),
        }

    @staticmethod
    def _is_symbolic_assisted(attempt: TestAttemptRecord) -> bool:
        if attempt.symbolic is not None:
            return True
        if attempt.intent is None:
            return False
        return any(
            provenance == "SYMBOLIC"
            or provenance.startswith("SYMBOLIC_")
            or provenance.startswith("SYMBOLIC_MODEL:")
            for provenance in attempt.intent.provenance
        )

    def _coverage_time_auc(self) -> tuple[float, float]:
        total_ms = max(self.budget.elapsed_sec * 1000, 0.0)
        if total_ms <= 0:
            return (self.state.statement_fraction, self.state.branch_fraction)
        statement_area = 0.0
        branch_area = 0.0
        previous_ms = 0.0
        previous_statement = 0.0
        previous_branch = 0.0
        for attempt in self.attempts:
            current_ms = min(max(attempt.run_elapsed_ms, previous_ms), total_ms)
            duration = current_ms - previous_ms
            statement_area += previous_statement * duration
            branch_area += previous_branch * duration
            previous_ms = current_ms
            previous_statement = attempt.cumulative_statement_cov
            previous_branch = attempt.cumulative_branch_cov
        tail = max(0.0, total_ms - previous_ms)
        statement_area += previous_statement * tail
        branch_area += previous_branch * tail
        return (statement_area / total_ms, branch_area / total_ms)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class HybridGenerationRunner:
    _TARGET_FAILURES_BEFORE_DEFERRAL = 3
    _POST_LLM_SYMBOLIC_MISS_LIMIT = 3
    _MAX_ROUTE_ATTEMPTS = {
        Route.SYMBOLIC: 4,
        Route.HYBRID: 3,
        Route.LLM: 3,
    }
    _LOOP_BOUNDS = (3, 5, 7, 10)

    def __init__(
        self,
        *,
        client: AkaUTClient | None = None,
        completer: IntentCompleter | None = None,
    ) -> None:
        self.client = client or AkaUTClient()
        self.completer = completer or IntentCompleter()
        self.selector = TargetSelector()

    def run(
        self,
        config: HybridGenerationConfig,
        *,
        initial_statement_keys: set[str] | None = None,
        initial_branch_keys: set[str] | None = None,
        initial_seeds: list[SuccessfulSeed] | None = None,
    ) -> HybridGenerationResult:
        model = self.client.get_coverage_model(config.function_path)
        state = ExactCoverageState(
            model,
            covered_statement_keys=set(initial_statement_keys or set()),
            covered_branch_keys=set(initial_branch_keys or set()),
        )
        state.covered_statement_keys.intersection_update(model.statement_keys)
        state.covered_branch_keys.intersection_update(model.branch_keys)
        budget = RunBudget(
            wall_time_sec=config.wall_time_minutes * 60,
            max_llm_calls=config.max_llm_calls,
            max_symbolic_attempts=config.max_symbolic_attempts,
            max_test_executions=config.max_test_executions,
        )
        scheduler = self._scheduler(config)
        attempts: list[TestAttemptRecord] = []
        decisions: list[RouteDecision] = []
        seeds: list[SuccessfulSeed] = list(initial_seeds or [])
        route_counts: dict[tuple[str, Route], int] = {}
        route_successes = {route: 0 for route in Route}
        route_uses = {route: 0 for route in Route}
        excluded_targets: set[str] = set()
        deferred_targets: set[str] = set()
        target_failures: dict[str, int] = {}
        partials: dict[str, SymbolicAttemptResponse] = {}
        previous_execution: dict[str, ExecutionResult] = {}
        previous_intent: dict[str, TestIntent] = {}
        post_llm_symbolic_misses = 0
        stop_reason = "routes_exhausted"
        error: str | None = None

        try:
            function_source = model.source or self.client.get_node_source(
                config.function_path
            ).source
            function_context = self.client.get_function_context(
                config.function_path
            ).context

            while True:
                if state.reached(config.statement_target, config.branch_target):
                    stop_reason = "coverage_target"
                    break
                budget_reason = budget.stop_reason()
                if budget_reason:
                    stop_reason = budget_reason
                    break
                if (
                    config.strategy == "hybrid"
                    and not budget.can_call_llm()
                    and post_llm_symbolic_misses
                    >= self._POST_LLM_SYMBOLIC_MISS_LIMIT
                ):
                    stop_reason = "routes_exhausted"
                    break

                include_branches = state.branch_fraction < config.branch_target
                include_statements = (
                    state.statement_fraction < config.statement_target
                )

                selection = self.selector.select(
                    state,
                    seeds if config.use_nearest_seed else [],
                    excluded_keys=excluded_targets | deferred_targets,
                    include_branches=include_branches,
                    include_statements=include_statements,
                )
                if selection is None and deferred_targets:
                    deferred_targets.clear()
                    selection = self.selector.select(
                        state,
                        seeds if config.use_nearest_seed else [],
                        excluded_keys=excluded_targets,
                        include_branches=include_branches,
                        include_statements=include_statements,
                    )
                if selection is None:
                    stop_reason = "routes_exhausted"
                    break

                available = self._available_routes(
                    config, budget, selection.target, route_counts, partials
                )
                if not available:
                    excluded_targets.add(selection.target.key)
                    continue

                features = self._features(
                    model,
                    state,
                    budget,
                    selection,
                    route_successes,
                    route_uses,
                    partials.get(selection.target.key),
                )
                route = scheduler.choose(features, available)
                iteration = len(attempts) + 1
                decision = RouteDecision(
                    iteration=iteration,
                    target=selection.target,
                    route=route,
                    features=features,
                    reason="configured strategy"
                    if config.strategy != "hybrid"
                    else "scheduler selection",
                )
                decisions.append(decision)
                route_uses[route] += 1
                key = (selection.target.key, route)
                route_counts[key] = route_counts.get(key, 0) + 1

                started = time.monotonic()
                record = TestAttemptRecord(
                    iteration=iteration,
                    route=route,
                    target=selection.target,
                )
                before_statement = state.statement_fraction
                before_branch = state.branch_fraction

                try:
                    symbolic = None
                    if route in {Route.SYMBOLIC, Route.HYBRID}:
                        symbolic = partials.get(selection.target.key)
                        if symbolic is None or route == Route.SYMBOLIC:
                            budget.consume_symbolic()
                            attempt_number = route_counts[key] - 1
                            loop_bound = self._LOOP_BOUNDS[
                                min(attempt_number, len(self._LOOP_BOUNDS) - 1)
                            ]
                            symbolic = self.client.create_symbolic_attempt(
                                SymbolicAttemptRequest(
                                    function_path=config.function_path,
                                    target=selection.target,
                                    seed=selection.seed,
                                    loop_bound=loop_bound,
                                )
                            )
                            record.symbolic = symbolic
                            if symbolic.status in {
                                SolverStatus.PARTIAL,
                                SolverStatus.UNSUPPORTED,
                            }:
                                partials[selection.target.key] = symbolic

                    symbolic_for_intent = symbolic
                    if (
                        symbolic is not None
                        and symbolic.status == SolverStatus.PARTIAL
                        and not config.use_symbolic_partial
                    ):
                        symbolic_for_intent = symbolic.model_copy(
                            deep=True, update={"intent": None}
                        )
                    intent = self._intent_for_route(
                        route, config.function_path, selection, symbolic_for_intent
                    )
                    previous = previous_execution.get(selection.target.key)
                    repair = previous_intent.get(selection.target.key)
                    is_repair = (
                        repair is not None
                        and previous is not None
                        and (not previous.passed or not previous.target_reached)
                        and config.use_divergence_repair
                        and route in {Route.HYBRID, Route.LLM}
                    )
                    if is_repair:
                        intent = repair.model_copy(deep=True)

                    needs_llm = route == Route.LLM or (
                        route == Route.HYBRID
                        and (
                            symbolic is None
                            or symbolic.status != SolverStatus.SOLVED
                            or bool(intent.unresolved_requirements)
                            or not intent.is_executable
                            or is_repair
                        )
                    )
                    if needs_llm:
                        budget.consume_llm()
                        completed, usage = self.completer.complete(
                            intent,
                            function_source=function_source,
                            function_context=function_context,
                            previous_execution=(
                                previous if config.use_divergence_repair else None
                            ),
                        )
                        intent = completed
                        record.input_tokens = usage.input_tokens
                        record.output_tokens = usage.output_tokens
                        budget.record_additional_llm_calls(
                            max(0, len(usage.interactions) - 1)
                        )

                    intent.validate_executable()
                    record.intent = intent
                    budget.consume_execution()
                    execution = self.client.execute_intent(
                        intent,
                        test_name=f"cov127_{config.run_id}_{iteration:03d}",
                    )
                    record.execution = execution
                    previous_execution[selection.target.key] = execution
                    previous_intent[selection.target.key] = intent.model_copy(deep=True)
                    delta = state.add(execution)
                    record.new_statement_keys = sorted(delta.new_statement_keys)
                    record.new_branch_keys = sorted(delta.new_branch_keys)
                    record.redundant = delta.redundant
                    if execution.passed:
                        seeds.append(
                            SuccessfulSeed(
                                intent_id=intent.intent_id,
                                bindings=tuple(intent.bindings),
                                result=execution,
                            )
                        )
                    if delta.new_statement_keys or delta.new_branch_keys:
                        route_successes[route] += 1
                        excluded_targets.discard(selection.target.key)
                        deferred_targets.clear()
                        target_failures[selection.target.key] = 0

                    reward = calculate_reward(
                        delta_statement_fraction=state.statement_fraction
                        - before_statement,
                        delta_branch_fraction=state.branch_fraction - before_branch,
                        route_seconds=time.monotonic() - started,
                        route_tokens=record.input_tokens + record.output_tokens,
                        invalid_test=not execution.passed,
                    )
                    decision.reward = reward
                    if config.scheduler_mode == "collect":
                        scheduler.update(route, features, reward)
                except AkaUTError as exc:
                    if exc.status_code not in {400, 422}:
                        raise
                    record.error = f"{type(exc).__name__}: {exc}"
                    decision.reward = calculate_reward(
                        delta_statement_fraction=0.0,
                        delta_branch_fraction=0.0,
                        route_seconds=time.monotonic() - started,
                        route_tokens=record.input_tokens + record.output_tokens,
                        invalid_test=True,
                    )
                    if config.scheduler_mode == "collect":
                        scheduler.update(route, features, decision.reward)
                except IntentCompletionError as exc:
                    budget.record_additional_llm_calls(
                        max(0, len(exc.usage.interactions) - 1)
                    )
                    record.input_tokens = exc.usage.input_tokens
                    record.output_tokens = exc.usage.output_tokens
                    record.error = f"{type(exc).__name__}: {exc}"
                    decision.reward = calculate_reward(
                        delta_statement_fraction=0.0,
                        delta_branch_fraction=0.0,
                        route_seconds=time.monotonic() - started,
                        route_tokens=record.input_tokens + record.output_tokens,
                        invalid_test=True,
                    )
                    if config.scheduler_mode == "collect":
                        scheduler.update(route, features, decision.reward)
                except Exception as exc:
                    record.error = f"{type(exc).__name__}: {exc}"
                    decision.reward = calculate_reward(
                        delta_statement_fraction=0.0,
                        delta_branch_fraction=0.0,
                        route_seconds=time.monotonic() - started,
                        route_tokens=record.input_tokens + record.output_tokens,
                        invalid_test=True,
                    )
                    if config.scheduler_mode == "collect":
                        scheduler.update(route, features, decision.reward)
                finally:
                    record.elapsed_ms = (time.monotonic() - started) * 1000
                    record.run_elapsed_ms = budget.elapsed_sec * 1000
                    record.cumulative_statement_cov = state.statement_fraction
                    record.cumulative_branch_cov = state.branch_fraction
                    attempts.append(record)
                    made_progress = bool(
                        record.new_statement_keys or record.new_branch_keys
                    )
                    if made_progress:
                        post_llm_symbolic_misses = 0
                    elif (
                        config.strategy == "hybrid"
                        and route == Route.SYMBOLIC
                        and not budget.can_call_llm()
                    ):
                        post_llm_symbolic_misses += 1
                    if not made_progress:
                        target_key = selection.target.key
                        failures = target_failures.get(target_key, 0) + 1
                        target_failures[target_key] = failures
                        if failures >= self._TARGET_FAILURES_BEFORE_DEFERRAL:
                            deferred_targets.add(target_key)
                            target_failures[target_key] = 0

        except AkaUTError as exc:
            stop_reason = "infra_error"
            error = str(exc)
        except Exception as exc:
            stop_reason = "error"
            error = f"{type(exc).__name__}: {exc}"

        budget.finish()
        policy_metadata = None
        if isinstance(scheduler, LinUCBPolicy):
            if config.scheduler_mode == "collect" and config.policy_path is not None:
                scheduler.save(config.policy_path)
            policy_json = json.dumps(scheduler.to_dict(), sort_keys=True).encode()
            policy_metadata = {
                "version": "1.0",
                "algorithm": "LinUCB",
                "frozen": scheduler.frozen,
                "sha256": hashlib.sha256(policy_json).hexdigest(),
                "observations": {
                    route.value: count
                    for route, count in scheduler.observations.items()
                },
            }

        return HybridGenerationResult(
            config=config,
            coverage_model=model,
            state=state,
            stop_reason=stop_reason,
            attempts=attempts,
            route_decisions=decisions,
            llm_interactions=self.completer.interactions,
            budget=budget,
            error=error,
            policy_metadata=policy_metadata,
        )

    @staticmethod
    def _scheduler(config: HybridGenerationConfig):
        if config.scheduler_mode == "rule":
            return RuleScheduler()
        if config.scheduler_mode == "frozen":
            assert config.policy_path is not None
            return LinUCBPolicy.load(config.policy_path, frozen=True)
        if config.policy_path is not None and config.policy_path.exists():
            return LinUCBPolicy.load(config.policy_path, frozen=False)
        return LinUCBPolicy(alpha=1.0, ridge=1.0, frozen=False)

    def _available_routes(
        self,
        config: HybridGenerationConfig,
        budget: RunBudget,
        target: Target,
        counts: dict[tuple[str, Route], int],
        partials: dict[str, SymbolicAttemptResponse],
    ) -> set[Route]:
        if config.strategy == "llm":
            candidates = {Route.LLM}
        elif config.strategy == "symbolic":
            candidates = {Route.SYMBOLIC}
        else:
            candidates = set(Route)

        available: set[Route] = set()
        for route in candidates:
            if counts.get((target.key, route), 0) >= self._MAX_ROUTE_ATTEMPTS[route]:
                continue
            if (
                route == Route.SYMBOLIC
                and target.key not in partials
                and budget.can_attempt_symbolic()
                and budget.can_execute()
            ):
                available.add(route)
            elif route == Route.LLM and budget.can_call_llm() and budget.can_execute():
                available.add(route)
            elif route == Route.HYBRID and budget.can_call_llm() and budget.can_execute():
                if target.key in partials or budget.can_attempt_symbolic():
                    available.add(route)
        return available

    @staticmethod
    def _intent_for_route(
        route: Route,
        function_path: str,
        selection: TargetSelection,
        symbolic: SymbolicAttemptResponse | None,
    ) -> TestIntent:
        if symbolic is not None and symbolic.intent is not None:
            if route == Route.SYMBOLIC and symbolic.status != SolverStatus.SOLVED:
                raise ValueError(
                    f"symbolic route requires SOLVED status; got {symbolic.status.value}"
                )
            intent = symbolic.intent.model_copy(deep=True)
            if selection.seed is not None and intent.seed is None:
                intent.seed = selection.seed
            return intent
        if route == Route.SYMBOLIC:
            status = symbolic.status.value if symbolic else "missing"
            raise ValueError(f"symbolic route did not produce an intent ({status})")
        return TestIntent(
            intent_id=str(uuid.uuid4()),
            function_path=function_path,
            target=selection.target,
            seed=selection.seed,
            expected_trace=list(selection.expected_path),
            unresolved_requirements=[
                "Create valid C++ setup and invocation for the selected coverage target"
            ],
            provenance=[route.value],
        )

    @staticmethod
    def _features(
        model: CoverageModel,
        state: ExactCoverageState,
        budget: RunBudget,
        selection: TargetSelection,
        successes: dict[Route, int],
        uses: dict[Route, int],
        symbolic: SymbolicAttemptResponse | None,
    ) -> dict[str, float]:
        def rate(route: Route) -> float:
            return successes[route] / uses[route] if uses[route] else 0.0

        static = model.features
        values = {
            "bias": 1.0,
            "path_depth": min(selection.path_depth / 50.0, 1.0),
            "loop_depth": min(selection.loop_depth / 10.0, 1.0),
            "constraint_count": min(
                (
                    len(symbolic.constraints)
                    if symbolic
                    else static.constraint_count
                )
                / 50.0,
                1.0,
            ),
            "is_branch": float(selection.target.kind.value == "BRANCH_EDGE"),
            "scalar_count": min(static.scalar_count / 20.0, 1.0),
            "string_count": min(static.string_count / 10.0, 1.0),
            "pointer_count": min(static.pointer_count / 10.0, 1.0),
            "container_count": min(static.container_count / 10.0, 1.0),
            "object_count": min(static.object_count / 10.0, 1.0),
            "external_call_count": min(static.external_call_count / 20.0, 1.0),
            "nonlinear_constraint_count": min(
                static.nonlinear_constraint_count / 10.0, 1.0
            ),
            "statement_coverage": state.statement_fraction,
            "branch_coverage": state.branch_fraction,
            "symbolic_success_rate": rate(Route.SYMBOLIC),
            "hybrid_success_rate": rate(Route.HYBRID),
            "llm_success_rate": rate(Route.LLM),
        }
        values.update(budget.remaining_fractions())
        return values
