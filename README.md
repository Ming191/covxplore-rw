# Covxplore

Covxplore is a single-agent system for automated C/C++ test case generation,
targeting improved MC/DC coverage. It drives the AkaUT REST API and runs an
iterative generate-execute-refine loop until coverage targets are met.

> Experimental tooling (prompt ablation matrix, parallel pipeline, run
> logging) lives on the `playground` branch.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling.

First, if you haven't already, install uv:

```bash
pip install uv
```

Make sure the project dependencies are installed (e.g., via `uv sync` or `uv pip install -e .` depending on your setup).

## Configuration

**Add your LLM API key (like `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`) into the`.env` file** depending on the configuration you are currently using.

## CLI Commands

Covxplore provides a single CLI entry point:

### `covxplore-gen`

Run a single test generation for one function with one prompt variant.

**Example:**
```bash
covxplore-gen \
    --path "/project/src/foo.cpp\MyNS::bar(int)" \
    --variant full \
    --out results/
```

**Options:**
- `--path`, `-p`: Absolute path of the function node (as returned by `/api/search`). *(Required)*
- `--variant`, `-v`: Prompt variant name (default: value from `Settings.default_prompt_variant`).
- `--out`, `-o`: Directory to write the summary JSON. Defaults to current directory.
- `--max-iter`: Override max iterations setting.
- `--mcdc-target`: Override MC/DC target 0.0–1.0 setting.
