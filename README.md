# Covxplore

Covxplore is a multi-agent system for automated C/C++ test case generation, targeting statement and branch coverage. It leverages an iterative loop to perform ablation studies on prompts, identifying the most effective prompting techniques and components.

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

Covxplore provides the following main CLI entry points:

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
- `--max-batches`: Override max batch count setting.

### `covxplore-ablate`

Run the full ablation matrix (all or selected variants, N repeats).

**Example:**
```bash
covxplore-ablate \
    --path "/project/src/foo.cpp\MyNS::bar(int)" \
    --variants full no_cot no_coverage baseline \
    --repeat 3 \
    --out results/
```

**Options:**
- `--path`, `-p`: Absolute path of the function node. *(Required)*
- `--variants`: Variant names to include. Defaults to all variants.
- `--repeat`, `-r`: Number of times to repeat the evaluations.
- `--out`, `-o`: Directory to write the results. Defaults to `results`.

*Note: The legacy `crewai run`, `train`, `replay`, and `test` commands are deprecated and unsupported in Covxplore v0.2. Please use the CLI endpoints defined above.*
