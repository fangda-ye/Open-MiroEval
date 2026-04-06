# Factual Eval

Active fact-checking powered by the [MiroFlow](https://github.com/MiroMindAI/MiroFlow) agent framework. Automatically extracts and verifies key factual statements in reports via search engines.

## How It Works

1. **Report Segmentation**: Splits the model-generated report into logical segments
2. **Per-segment Fact-checking**: Deploys an agent for each segment to gather evidence via web search
3. **Verdict**: Labels each factual statement as `Right` (correct) / `Wrong` (incorrect) / `Unknown` (unverifiable). For multimodal evaluation, an additional `Conflict` label is used when web sources and attachment content provide contradictory evidence.

## Directory Structure

```
factual_eval/
├── config/                    # Hydra configuration files
│   ├── benchmark/             # Base benchmark config (factual-eval.yaml)
│   ├── llm/                   # LLM model configs
│   ├── tool/                  # Tool configs (search, browsing, etc.)
│   ├── prompts/               # Prompt templates
│   ├── benchmark_factual-eval_text.yaml        # Canonical config for text-only models
│   └── benchmark_factual-eval_multimodal.yaml  # Canonical config for multimodal models
├── miroflow/                  # MiroFlow core framework
│   ├── agents/                # Agent implementations (iterative + rollback)
│   ├── benchmark/             # Evaluation runners and verifiers
│   ├── llm/                   # Multi-provider LLM support
│   ├── tool/                  # MCP server tool integration
│   ├── io_processor/          # I/O processors (segmentation, summarization, etc.)
│   ├── logging/               # Task tracing and logging decorators
│   ├── skill/                 # Skill manager and definitions
│   └── utils/                 # Utility functions
├── utils/
│   ├── convert_to_factual_eval.py  # Convert method_results JSON array → per-item files
│   └── check_progress_factual_eval.py
├── scripts/
│   ├── run_factual_eval.sh    # Main run script
│   └── run_single_factual_eval_task.py  # Single-task runner
├── .env.template              # Environment variables template
└── pyproject.toml             # Dependencies (Python >= 3.11)
```

## Data Loading

Factual eval reads per-item JSON files from `data/factual-eval/<model-dir>/` inside `factual_eval/`.
The base data directory is configured via the `DATA_DIR` environment variable. The shell script defaults to `./data` (relative to `factual_eval/`).

**Step 1 — Convert raw results to per-item files** (one-time, skip if already done):

```bash
cd factual_eval

# Convert a method_results JSON array → individual files in factual_eval/data/factual-eval/
python utils/convert_to_factual_eval.py \
    --input ../data/method_results/mirothinker_v17_text_demo.json \
    --output-dir data/factual-eval/mirothinker-v17-text-demo
```

The output format is one JSON file per item (same schema as the source), named `<model-name>_<id>.json`.

The loader also supports reading directly from a JSON array file via `--source-file` (see Usage below).

For multimodal queries, attachment files are stored in `data/input_queries/multimodal-attachments/<query_id>/` and referenced via the `files` field.

## Setup

Recommended: run `uv sync` at the **repo root** to create a shared `.venv` for all modules.

```bash
# From repo root
uv sync
cp .env.template .env   # edit with your API keys
```

The standalone setup is also supported:

```bash
cd factual_eval
uv sync
cp .env.template .env
```

## Usage

```bash
cd factual_eval

# Evaluate a specific model (from pre-converted per-item files)
bash scripts/run_factual_eval.sh --model-dir mirothinker-v17-text

# Evaluate directly from a JSON array file (no pre-conversion needed)
bash scripts/run_factual_eval.sh --source-file mirothinker_v17_text_100.json

# Multimodal evaluation
bash scripts/run_factual_eval.sh \
    --config config/benchmark_factual-eval_multimodal.yaml \
    --model-dir mirothinker-v17-multimodal

# Limit number of tasks (for testing)
bash scripts/run_factual_eval.sh --model-dir chatgpt-text-only --max-tasks 5

# Control concurrency
bash scripts/run_factual_eval.sh --model-dir mirothinker-v17-text \
    --max-concurrent 5 --max-concurrent-chunks 5

# Resume a previous run
bash scripts/run_factual_eval.sh --result-dir logs/factual-eval/prev_run
```

## Output Format

Each query produces a JSON result containing a `core_state` list:

```json
{
  "core_state": [
    {
      "statement": "The statement being verified",
      "verification": "Right | Wrong | Unknown | Conflict",
      "evidence": [
        { "source": "Evidence source URL", "excerpt": "Quoted key text from source" }
      ],
      "reasoning": "Explanation of the verification reasoning and process"
    }
  ]
}
```

**Key Metric:** Correct Statement Ratio = Right / (Right + Wrong + Unknown + Conflict)
