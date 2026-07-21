# COV127: C++ Neuro-Symbolic Unit Test Generation

## Objective

Build a bidirectional symbolic/LLM unit-test generator for C++ that targets exact
uncovered statements and branch edges. AkaUT owns CDT analysis, CFG construction,
symbolic execution, instrumentation, compilation, and execution. Covxplore owns
target selection, route scheduling, LLM completion/repair, budgets, and experiment
artifacts.

Contract version `1.0` is shared by both repositories.

## Non-Negotiable Scope

- The subject programs and generated tests are C/C++.
- Statement and branch coverage reported by AkaUT are authoritative.
- MC/DC is not part of the `cov127` public contracts, stop conditions, prompts,
  reports, or evaluation.
- Assertion-oracle quality, external gcov/LLVM coverage, UI, authentication, and
  multi-user execution are out of scope.
- Only passing tests contribute suite coverage. Failed tests remain available as
  repair feedback.

## AkaUT Responsibilities

- Expose a stable coverage model for one focal method: statements, branch edges,
  CFG relationships, source ranges, successor/case identity, and downstream
  statements.
- Identify statements and edges with stable keys derived from normalized source
  path and source offsets, never transient CFG node IDs.
- Provide targeted symbolic attempts with loop bound 3, adaptively increasing to
  10, a 60-second solver timeout, and status values `SOLVED`, `PARTIAL`, `UNSAT`,
  `UNSUPPORTED`, `TIMEOUT`, or `ERROR`.
- Materialize versioned `TestIntent` objects into C++ test bodies, compile and run
  them, and return exact visited statement/edge keys plus an ordered focal trace.
- Limit ordered focal traces to 5,000 steps and expose `traceTruncated`.
- Serialize execution through a global lock; concurrent execution returns HTTP
  `409 BUSY`.

Required REST API:

```text
GET  /api/v2/functions/coverage-model?absolutePath=...
POST /api/v2/symbolic/attempts
POST /api/v2/test-intents/execute
```

Existing search, context, and source endpoints remain available.

## Shared TestIntent 1.0

```json
{
  "version": "1.0",
  "intentId": "...",
  "functionPath": "...",
  "target": {
    "kind": "BRANCH_EDGE",
    "key": "...",
    "desiredOutcome": true
  },
  "seed": {},
  "actions": [],
  "bindings": [],
  "expectedTrace": [],
  "unresolvedRequirements": [],
  "provenance": []
}
```

- Target kinds: `STATEMENT`, `BRANCH_EDGE`.
- Action kinds: `DECLARE`, `CONSTRUCT`, `ASSIGN`, `CALL`, `STUB`, `RAW_CPP`,
  `INVOKE`.
- Every executable intent contains exactly one `INVOKE` action.
- A binding carries `cppLvalue`, `cppType`, `value`, `origin`, and `locked`.
  LLM processing must preserve locked symbolic values.
- Materialization topologically orders action dependencies before emitting C++.

## Covxplore Responsibilities

- Replace autonomous tool-loop execution with an explicit
  `HybridGenerationRunner` state machine.
- Maintain exact set-union coverage over stable statement and branch-edge keys.
  A passing test is redundant when it adds neither kind of key.
- Select targets by downstream uncovered-statement potential, nearest seed by
  longest common trace prefix, lower path depth, then stable key.
- Target remaining statements after branch targets are exhausted.
- Routes are `SYMBOLIC`, `HYBRID`, and `LLM`. Symbolic partial models are completed
  by the LLM without changing locked bindings. Execution divergence triggers a
  minimal repair of the same intent.
- Successful LLM-generated tests become seeds for nearby symbolic exploration.

## Scheduler And Budgets

The final scheduler is LinUCB with a deterministic rule fallback. Context features
include path/loop depth, constraint count, C++ value categories, external/nonlinear
operations, route history, current coverage, and remaining budget.

```text
reward = 0.4 * delta_statement_fraction
       + 0.6 * delta_branch_fraction
       - 0.05 * min(route_seconds / 60, 1)
       - 0.05 * min(route_tokens / 8000, 1)
       - 0.20 * invalid_test
```

Defaults:

- LinUCB alpha `1`, ridge `1`, random seed `42`.
- 30 minutes per function.
- 15 LLM calls, 30 symbolic attempts, and 30 test executions per function.
- No default LLM output-token cap. Tokens are measured for reward/reporting.
- Train policy on training projects, select it on validation projects, and freeze
  it before test-project evaluation.

CLI additions:

```text
--strategy llm|symbolic|hybrid
--statement-target <fraction>
--branch-target <fraction>
--scheduler rule|collect|frozen
--policy <path>
--wall-time-minutes <number>
--max-llm-calls <number>
--max-symbolic-attempts <number>
--max-test-executions <number>
```

Stop reasons are `coverage_target`, `wall_time_budget`, `execution_cap`,
`routes_exhausted`, `infra_error`, and `error`.

## Artifacts And Evaluation

- JSON stores coverage-model hash, targets, TestIntents, generated C++, traces,
  symbolic attempts, route decisions/features/reward, token counts, timings, and
  policy version/hash.
- CSV uses `statement_cov` and `branch_cov`; do not relabel statement coverage as
  line coverage.
- Pilot functions: Hjson `_next`, `_parseLoop`, `_readString`, and `_quoteName`.
- Evaluation uses project-level train/validation/test splits.
- Baselines: AkaUT symbolic, Covxplore LLM, sequential symbolic-then-LLM, result
  union, rule hybrid, and frozen-bandit hybrid.
- Ablations: no nearest seed, no first-divergence repair, no partial model, and rule
  scheduler versus frozen LinUCB.
- Metrics: statement/branch coverage, coverage-time AUC, coverage/token, wall time,
  route counts, pass/compile/runtime rates, and redundancy.

## Delivery Order

1. Establish branches, this contract, Java 17/Maven baseline, and Python baseline.
2. Implement contract models and stable keys on both sides.
3. Implement AkaUT statement/branch coverage model and ordered trace.
4. Extract targeted symbolic services and TestIntent materialization.
5. Expose REST v2 and add Java contract/API tests.
6. Implement Covxplore coverage state, target/seed selector, budgets, and runner.
7. Add structured LLM completion and first-divergence repair.
8. Add rule/collection/frozen LinUCB scheduler modes.
9. Complete CLI, artifacts, baselines, ablations, and Hjson integration checks.

Definition of done requires Java/Python tests to pass, synchronized contract `1.0`,
enforced budgets, no public MC/DC dependency, immutable frozen evaluation policy,
and reproducible C++ experiment artifacts.
