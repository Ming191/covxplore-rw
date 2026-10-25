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

### Kaggle Qwen endpoint

`LOCAL_BASE_URL` can point at an SSH tunnel to a Qwen server on Kaggle. Start script on Kaggle after SSH login:

```bash
bash scripts/kaggle-qwen.sh check
bash scripts/kaggle-qwen.sh setup
```

`check` validates existing `server.py` configuration; it will not rewrite unknown source. `setup` starts Uvicorn and checks `/health` and `/v1/models`. If script is only on local machine, run this from repo root to stream it over SSH and set up Kaggle:

```bash
export KAGGLE_SSH_TARGET='user@host'
export KAGGLE_SSH_PORT=22
export KAGGLE_SSH_KEY="$HOME/.ssh/id_ed25519" # optional
scripts/kaggle-qwen.sh remote-setup
```

For local tunnel, use same SSH settings:

```bash
export KAGGLE_SSH_TARGET='user@host'
export KAGGLE_SSH_PORT=22
export KAGGLE_SSH_KEY="$HOME/.ssh/id_ed25519" # optional; SSH agent/config also works
bash scripts/kaggle-qwen.sh tunnel
```

Configure Covxplore in another local shell:

```bash
export LLM_PROVIDER=local
export LOCAL_BASE_URL=http://127.0.0.1:18000/v1
export LOCAL_MODEL=Qwen3-4B-Instruct-2507
export LOCAL_API_KEY=
```

The script expects reviewed server file at `/kaggle/working/covxplore-qwen/server.py` and model files at `/kaggle/working/mi_cpp/models/Qwen3-4B-Instruct-2507`. Override paths and ports with `KAGGLE_QWEN_DIR`, `KAGGLE_QWEN_MODEL_PATH`, `KAGGLE_QWEN_PORT`, and `KAGGLE_QWEN_LOCAL_PORT`. Server binds loopback only.

**Caution:** Source patch automation is intentionally not included; repo does not contain canonical Kaggle `server.py`. Keep current validated settings (`MAX_CONTEXT=16384`, `prefill_chunk_size=1024`, dynamic cache, 1024 output cap, prompt and elapsed logs) in that remote source before running `check`/`setup`.

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
