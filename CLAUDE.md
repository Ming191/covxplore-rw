# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Covxplore is a Python 3.10+ MC/DC coverage-driven C/C++ test-generation agent. It uses CrewAI to drive an LLM agent that calls AkaUT's REST API, generates C++ test driver bodies, executes them, tracks statement/branch/MC/DC coverage, and exports generation/ablation results.

This package is not standalone: generation runs require the AkaUT Java desktop/API process to be running and serving REST endpoints, defaulting to `http://localhost:8080`.

## Common commands

```bash
# Install/sync dependencies from uv.lock and pyproject.toml
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_driver_executor.py

# Run a single test
uv run pytest tests/test_driver_executor.py::test_akaut_executor_calls_client

# Build the package from pyproject.toml
uv build

# Run one generation for an absolute AkaUT function node path
uv run covxplore-gen --path "<absolute-function-node-path>" --variant full --out results

# Run an ablation matrix for one function
uv run covxplore-ablate --path "<absolute-function-node-path>" --leave-one-out --repeat 3 --out results

# Run ablations for many functions listed one per line
uv run covxplore-pipeline --paths-file functions.txt --workers 3 --leave-one-out --repeat 3 --out results
```

No dedicated lint/format/type-check tool is configured in `pyproject.toml`.

## Runtime configuration

Settings are loaded from environment variables or `.env` via `covxplore.config.Settings`.

Important variables:

- `AKAUT_BASE_URL` — AkaUT API base URL, default `http://localhost:8080`.
- `LLM_PROVIDER` — `deepseek` by default; `kimchi` is also supported.
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`.
- `KIMCHI_API_KEY`, `KIMCHI_BASE_URL`, `KIMCHI_MODEL`.
- `MAX_BATCHES`, `MCDC_TARGET`, `REDUNDANT_STREAK_LIMIT`, `FAIL_STREAK_LIMIT`, `REQUEST_TIMEOUT_SEC`.

## Architecture notes

- `covxplore.main` owns CLI entry points: `covxplore-gen`, `covxplore-ablate`, and `covxplore-pipeline`.
- `covxplore.generator.generate` is the core single-run flow. It seeds static AkaUT context/conditions, builds the CrewAI crew, runs generation, reconciles token usage, deduces stop reasons, and returns a serializable `GenerationResult`.
- `covxplore.crew` builds a sequential single-agent CrewAI crew using YAML in `src/covxplore/config/agents.yaml` and `tasks.yaml`. The agent gets AkaUT tools for search, source, single-test execution, and batch execution.
- `covxplore.tools.execute_testcase` contains the main tool loop boundary. It validates agent actions, executes generated test bodies through the driver executor, updates per-run `TestSuite` state, and raises `HardStop` for coverage target, redundant streak, or fail streak termination.
- `covxplore.api_client` is the thin HTTP client for AkaUT endpoints: `/api/search`, `/api/context`, `/api/node/source`, `/api/node/conditions`, and `/api/testcase/execute`.
- `covxplore.driver` separates the test-body contract validator from the executor that talks to AkaUT. Keep validation at this boundary; generated test bodies are untrusted LLM output.
- `covxplore.coverage` accumulates coverage state and turns execution responses into gap guidance for the next prompt.
- `covxplore.prompts` defines prompt sections and named ablation variants. `registry.py` is the source of valid variant names such as `baseline`, `full`, `full_shots`, and `no_*` leave-one-out variants.
- `covxplore.ablation` runs repeated variant matrices for one function and exports JSON/CSV. `covxplore.pipeline` runs ablations for multiple functions in separate processes and writes a combined summary CSV.
- DTOs live under `covxplore.types`; prefer extending these models over passing raw API dictionaries through generation logic.

## Development cautions

- Before implementing any new feature, check current CrewAI docs with Context7 if the change touches CrewAI, agents, tools, tasks, prompts, orchestration, or LLM execution; reuse CrewAI capabilities instead of duplicating them.
- Keep generated-test validation in `covxplore.driver` / `covxplore.tools.execute_testcase`; do not trust LLM-produced C++ bodies.
- Generation tests should avoid requiring a live AkaUT server; existing tests use fakes/monkeypatching around clients and executors.
- Avoid broad recursive searches that include `.venv`, batch outputs, or result directories; this repo contains local environment and generated experiment data.
