# Covxplore COV127

Covxplore COV127 generates unit-test driver bodies for C++ focal methods by
coordinating AkaUT targeted symbolic execution and an LLM. Coverage objectives
are exact focal-method statement keys and branch-edge keys. MC/DC is not part of
the COV127 runtime contract, stop condition, or report.

## Architecture

```text
C++ project loaded in AkaUT
  -> CDT/CFG + targeted symbolic execution/Z3
  -> TestIntent 1.0
  -> Covxplore target selector and route scheduler
  -> optional LLM completion/repair
  -> AkaUT materialize/compile/execute
  -> exact ordered statement/branch trace
```

AkaUT remains the authoritative C++ analyzer and executor. Covxplore owns the
hybrid state machine, exact suite-level coverage union, budgets, route policy,
and experiment artifacts.

## Prerequisites

- Python 3.10-3.13.
- AkaUT branch `cov127`, built with Java 17.
- AkaUT GUI running with a C++ environment loaded and successfully built.
- Z3 configured in AkaUT for symbolic routes.
- A configured LLM key for `llm` or `hybrid` routes.

Install Covxplore in the Windows environment:

```powershell
cd D:\covxplore-rw
.\.venv-win\Scripts\python.exe -m pip install -e .
```

Copy `.env.example` to `.env`, then set `AKAUT_BASE_URL`, `LLM_PROVIDER`, and the
selected provider credentials. Covxplore does not impose a default output-token
cap.

## AkaUT API V2

```text
GET  /api/v2/functions/coverage-model?absolutePath=...
POST /api/v2/symbolic/attempts
POST /api/v2/test-intents/execute
```

The existing `/api/search`, `/api/context`, and `/api/node/source` endpoints are
used for discovery and context. The COV127 branch does not expose the old MC/DC
conditions or legacy one-shot execution endpoints.

## Single Function

Get an exact function path from AkaUT search, then run:

```powershell
$path = 'D:\project\src\file.cpp\Namespace\function(int)'

.\.venv-win\Scripts\covxplore-gen.exe `
  --path $path `
  --strategy hybrid `
  --statement-target 1.0 `
  --branch-target 1.0 `
  --scheduler rule `
  --wall-time-minutes 30 `
  --max-llm-calls 15 `
  --max-symbolic-attempts 30 `
  --max-test-executions 30 `
  --out results\cov127
```

Strategies are `llm`, `symbolic`, and `hybrid`. Scheduler modes are `rule`,
`collect`, and `frozen`. A frozen run requires `--policy`.

## Batch

AkaUT has process-global execution state, so batch execution is deliberately
sequential.

```powershell
.\.venv-win\Scripts\covxplore-batch.exe `
  --source-files hjson_decode.cpp hjson_encode.cpp `
  --strategy hybrid `
  --scheduler rule `
  --out results\hjson_cov127
```

Alternatively pass `--paths-file functions.txt`. Resume is enabled by default
and uses a hash of strategy, targets, budgets, ablation flags, and policy content.
Use `--no-resume` to rerun matching configurations.

## Policy Lifecycle

Collect a LinUCB policy on training projects:

```powershell
.\.venv-win\Scripts\covxplore-batch.exe `
  --paths-file train.txt `
  --scheduler collect `
  --policy results\policy\collected.json `
  --out results\train
```

After model selection on validation projects, create an immutable evaluation
artifact:

```powershell
.\.venv-win\Scripts\covxplore-policy.exe freeze `
  --input results\policy\collected.json `
  --out results\policy\frozen.json
```

Evaluation uses `--scheduler frozen --policy results\policy\frozen.json`.

## Experiments

`covxplore-ablate` supports:

- `llm`
- `symbolic`
- `symbolic_then_llm`
- `symbolic_llm_union`
- `hybrid_rule`
- `hybrid_frozen`
- `no_nearest_seed`
- `no_first_divergence_repair`
- `no_symbolic_partial`

```powershell
.\.venv-win\Scripts\covxplore-ablate.exe `
  --path $path `
  --variants llm symbolic symbolic_then_llm hybrid_rule `
  --repeat 3 `
  --out results\ablation
```

## Artifacts

Each run writes enriched JSON, one-row CSV, and a cumulative CSV. Artifacts
include coverage-model hash, TestIntents, C++ bodies, ordered traces, solver
status, scheduler features/reward, statement/branch coverage, token use, route
time, total time, and policy hash. Only `PASSED` tests contribute coverage.

The cumulative CSV keeps final coverage, coverage-time AUC, test outcomes,
tokens, route counts, and compact Z3 telemetry. In particular,
`z3_solved/unsat/unsupported/timeout/error` describe actual solver calls;
`symbolic_useful_rate` is the fraction of symbolic-assisted executions that
passed and added new suite coverage; and the `symbolic_new_*` versus
`hybrid_assisted_new_*` columns separate direct symbolic gains from gains that
required LLM completion. `z3_solve_rate` alone is not a coverage metric.
Detailed path candidates, per-attempt constraints, timings, traces, model and
policy hashes remain in JSON. When a stopped batch is resumed, an older CSV
header is migrated in place and existing rows are preserved; telemetry that
was not recorded by the older runtime remains blank. Close the CSV in Excel
before resuming because Excel locks the file against schema migration/appends.

## Hybrid Behavior And Limitations

- A target that produces no new passing coverage for three attempts is deferred
  so another uncovered target can be explored. Any later coverage gain reopens
  deferred targets with the enlarged seed pool.
- Once either coverage target is reached, selection focuses only on the metric
  still below target. After the LLM budget is exhausted, three consecutive
  symbolic-only misses stop the hybrid run instead of scanning every remaining
  edge without a materialization route.
- Scalar constraints that AkaUT can solve remain locked in `TestIntent`.
  Constructors, pointers, containers, object state, and unsupported C++
  expressions are reported as partial requirements for the LLM to complete.
- Compile and runtime failures are retained for minimal repair through the
  ordered trace and first divergence, but their visited keys never enter suite
  coverage.
- Coverage and runtime vary across LLM samples, especially for stateful focal
  methods. Evaluation should therefore use repeated runs and project-level
  train/validation/test splits rather than a single favorable run.
- Rebuilding AkaUT does not update an already running JVM. Restart the GUI and
  reload the C++ environment after Java changes before an integration run.

## Verification

```powershell
# Covxplore
.\.venv-win\Scripts\python.exe -m pytest -q

# COV127 AkaUT tests
cd D:\akautauto
.\mvnw.cmd -q "-Dtest=com.dse.api.v2.**" test
```

The full historical AkaUT test suite contains GUI- and fixture-dependent tests
that cannot run in a headless terminal. COV127 tests are isolated under
`com.dse.api.v2`.
