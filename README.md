# Covxplore

Covxplore generates C/C++ test drivers with one CrewAI agent and compares direct, Chain-of-Thought, and CFG path-guided treatments under fixed context, coverage-gap, execution-feedback, and batch infrastructure.

## Installation

Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest
```

Generation also requires the AkaUT API, defaulting to `http://localhost:8080`.

## Configuration

Set `LLM_PROVIDER=deepseek` with `DEEPSEEK_API_KEY`, or use `LLM_PROVIDER=local` with an OpenAI-compatible `LOCAL_BASE_URL` and `LOCAL_MODEL`.

## CLI

```bash
# One function, one treatment
uv run covxplore-gen \
  --path "/project/src/foo.cpp::MyNS::bar(int)" \
  --variant none \
  --out results

# All treatments, repeated
uv run covxplore-ablate \
  --path "/project/src/foo.cpp::MyNS::bar(int)" \
  --repeat 3 \
  --out results

# Selected treatments
uv run covxplore-ablate \
  --path "/project/src/foo.cpp::MyNS::bar(int)" \
  --variants none cot path_guided

# Multiple functions, one absolute path per line
uv run covxplore-pipeline \
  --paths-file functions.txt \
  --workers 3 \
  --repeat 3 \
  --out results
```

Available treatments: `none`, `cot`, and `path_guided`.
