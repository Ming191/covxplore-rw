# CLAUDE.md

## Project overview

Covxplore is a Python 3.10+ statement/branch coverage-driven C/C++ test-generation experiment. It uses one CrewAI agent to generate typed candidate batches, executes them through AkaUT, and feeds coverage gaps and execution feedback into the next batch.

Generation requires AkaUT's Java API process, defaulting to `http://localhost:8080`.

## Common commands

```bash
uv sync
uv run pytest
uv build

# One function and one reasoning treatment
uv run covxplore-gen --path "<absolute-function-node-path>" --variant none --out results

# All five reasoning treatments for one function
uv run covxplore-ablate --path "<absolute-function-node-path>" --repeat 3 --out results

# Many functions listed one per line
uv run covxplore-pipeline --paths-file functions.txt --workers 3 --repeat 3 --out results
```

No dedicated lint, format, or type-check tool is configured.

## Runtime configuration

Settings come from environment variables or `.env` via `covxplore.config.Settings`.

- `AKAUT_BASE_URL` — AkaUT API URL, default `http://localhost:8080`.
- `LLM_PROVIDER` — `deepseek` or `local`.
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`.
- `LOCAL_API_KEY`, `LOCAL_BASE_URL`, `LOCAL_MODEL`.
- `MAX_BATCHES`, `REDUNDANT_STREAK_LIMIT`, `FAIL_STREAK_LIMIT`, `REQUEST_TIMEOUT_SEC`.
- `LLM_TEMPERATURE`, `LLM_SEED`, `LLM_THINKING`, `MAX_TOKENS`.

## Architecture notes

- `covxplore.main` owns `covxplore-gen`, `covxplore-ablate`, and `covxplore-pipeline`.
- `covxplore.flows.generation_flow` runs the fixed loop: static context and source, coverage gaps and prior execution feedback, typed LLM batch, AkaUT execution, repeat.
- `covxplore.generation.runner` exposes `StrategyRunner` (wraps one canonical `ReasoningStrategy`) and `build_crew` for direct, CoT, and CFG path-guided generation.
- `covxplore.tools.execute_testcase.ExecuteTestcaseBatchTool` validates untrusted test bodies, executes them, merges structural coverage, and raises deterministic stop conditions.
- `covxplore.prompts` keeps infrastructure fixed and varies only `none`, `cot`, or `path_guided`.
- `covxplore.generator.GenerationResult` freezes experiment metadata and emits JSON; `covxplore.experiment.flat_row` emits router-training CSV rows.
- `covxplore.ablation` and `covxplore.pipeline` run repeated treatment matrices.

## Development cautions

- Check current CrewAI docs with Context7 before changing CrewAI agents, tasks, flows, LLM execution, or structured output.
- Keep generated-test validation in `covxplore.driver` and `covxplore.tools.execute_testcase`; LLM-generated C++ is untrusted.
- Keep all non-reasoning infrastructure identical across treatments.
- Tests must not require a live AkaUT server.
- Exclude `.venv` and generated result directories from broad searches.
