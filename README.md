<div align="center">
  <img src="static/miromind_logo.png" width="45%" alt="MiroMind" />

  <h3>MiroEval: Benchmarking Multimodal Deep Research Agents in Process and Outcome</h3>

[![Paper](https://img.shields.io/badge/Paper-arXiv%202603.28407-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.28407)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://miroeval-ai.github.io/blog)
[![WEBSITE](https://img.shields.io/badge/Website-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://miroeval-ai.github.io/website)
[![GITHUB](https://img.shields.io/badge/Github-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MiroMindAI/MiroEval)

</div>

MiroEval is a comprehensive evaluation framework for Deep Research systems, providing automated **task generation** and assessment across three complementary dimensions: **Factual** correctness, **Point**-wise quality, and **Process** quality.

<div align="center">
  <img src="static/text_70_results.png" width="90%" alt="Text-only 70-query evaluation results across three dimensions" />
  <p><i>Text-only evaluation results (70 queries) across Synthesis Quality, Factual Accuracy, and Process Quality.</i></p>
</div>

All three evaluation modules share a unified `data/` directory as their input data source. Each sub-project manages its own `.env` file for API keys (see `.env.template` in each sub-project).

## Architecture

```
MiroEval/
├── task_generation/           # Evaluation task generation pipeline
├── data/                      # Shared data directory
│   ├── input_queries/         # Evaluation query sets + multimodal attachments
│   └── detail_results/        # Per-task per-model intermediate scores
├── factual_eval/              # Factual evaluation (MiroFlow-based fact-checking agent)
├── point_quality/             # Quality evaluation (adaptive point-wise scoring)
└── process_eval/              # Process evaluation (intrinsic process quality + report alignment)
```

> **Note:** Model result files (one JSON array per model) should be placed in user-created directories such as `data/method_results/` (text-only) and `data/method_multimodal_results/` (multimodal). These directories are not included in the repository and must be created before running evaluations.

---

## Task Generation

Automated pipeline for generating high-quality deep-research evaluation queries. Combines anonymized seed patterns from real user queries, real-time web trends, LLM generation, and multi-stage filtering (search validation, deep-research necessity, quality gating) to produce challenging evaluation tasks.

See [`task_generation/README.md`](task_generation/README.md) for full details.

---

## Data Format

### Input Queries (`data/input_queries/`)


| File / Directory            | Description                                                                                                                                  | Count |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `mirobench_text.json`       | Text-only query set                                                                                                                          | 70    |
| `mirobench_multimodal.json` | Multimodal query set (with image/document attachments)                                                                                       | 30    |
| `multimodal-attachments/`   | Attachment files referenced by multimodal queries, organized by query ID (e.g., `72/`, `93/`). Contains images, PDFs, and other documents.   | —     |


**Query Schema (text-only):**

```json
{
  "id": 1,
  "chat_id": "uuid",
  "rewritten_query": "Expanded/rewritten query",
  "annotation": {
    "category": "text",
    "language": "zh | en",
    "origin_id": 2,
    "pattern": "T1 | T2 | T5 | T6",
    "domain": "tech | finance | medical | ...",
    "topic": "...",
    "persona": "...",
    "...": "additional fields omitted"
  }
}
```

**Query Schema (multimodal):**

```json
{
  "id": 71,
  "chat_id": "uuid",
  "rewritten_query": "Expanded/rewritten query",
  "files": [
    { "filename": "attachment_71_01.jpg", "type": "image", "dir": "multimodal-attachments/71/attachment_71_01.jpg", "size": "1.5 MB" }
  ],
  "annotation": {
    "category": "image | doc | multi_doc",
    "language": "zh | en",
    "origin_id": 102
  }
}
```

> **Note:** Text queries do not have a `files` field. Multimodal queries do not have `pattern` or `domain` fields.

**Pattern Taxonomy:** (applies to text queries; ~50% of text queries carry a pattern label)

- T1: Landscape Survey
- T2: Comparative Evaluation
- T5: Decision Analysis
- T6: Scheme Design

**Domain Distribution:** tech, finance, medical, engineering, business, humanities, science, lifestyle, cybersecurity, education, energy, geopolitics, health, legal, policy, trade, other

### Model Results (`data/method_results/`, `data/method_multimodal_results/`)

One JSON file per model, containing a JSON array of complete query-response pairs. Place your model's output file (e.g., `<model_name>_text.json`) in the appropriate directory (these directories must be created by the user).

**Result Schema:**

```json
{
  "id": 1,
  "chat_id": "uuid",
  "rewritten_query": "Rewritten query",
  "annotation": { "..." },
  "response": "Model-generated research report",
  "process": "String of research process"
}
```

The `response` field contains the model's final report output. The `process` field contains the model's intermediate research process trace (format varies by model). Multimodal entries additionally contain a `files` field (see Query Schema above).

---

## 1. Factual Eval

Active fact-checking powered by the [MiroFlow](https://github.com/MiroMindAI/MiroFlow) agent framework. Automatically extracts and verifies key factual statements in reports via search engines.

### How It Works

1. **Report Segmentation**: Splits the model-generated report into logical segments
2. **Per-segment Fact-checking**: Deploys an agent for each segment to gather evidence via web search
3. **Verdict**: Labels each factual statement as `Right` (correct) / `Wrong` (incorrect) / `Unknown` (unverifiable). For multimodal evaluation, an additional `Conflict` label is used when web sources and attachment content provide contradictory evidence.

### Directory Structure

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

### Data Loading

Factual eval reads per-item JSON files from `data/factual-eval/<model-dir>/` inside `factual_eval/`.
The base data directory is configured via the `DATA_DIR` environment variable. The shell script defaults to `./data` (relative to `factual_eval/`), while the Hydra config falls back to `../../miroflow/data`. It is recommended to set `DATA_DIR` explicitly in your `.env` file.

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

### Setup

```bash
cd factual_eval

# Install dependencies
uv sync

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your API keys (OPENAI_API_KEY, SERPER_API_KEY, etc.)
# Optionally set DATA_DIR to the absolute path of miroflow/data/
```

### Usage

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

### Output Format

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

---

## 2. Comprehensive Adaptive Point-wise Quality Evaluation

Comprehensive Adaptive Point-wise Quality Evaluation that dynamically generates evaluation dimensions, criteria, and weights for each query task, enabling fine-grained quality assessment.

### How It Works

The evaluation pipeline consists of 5 stages:

1. **Dimension Generation**: LLM generates 1-3 task-specific additional dimensions (supplementing 4 fixed dimensions)
2. **Weight Assignment**: Assigns normalized weights to all dimensions (summing to 1.0)
3. **Criteria Generation**: Generates 1-10 specific evaluation criteria per dimension
4. **Per-criteria Scoring**: Scores the report against each criterion (0-10)
5. **Hierarchical Aggregation**: Criteria scores -> dimension scores -> total weighted score

**4 Fixed Dimensions:**


| Dimension             | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| Coverage              | Breadth, depth, and relevance of coverage                  |
| Insight               | Depth, originality, logic, and analytical value            |
| Instruction Following | Accuracy in meeting all query requirements                 |
| Clarity               | Readability, fluency, structure, and ease of understanding |


### Directory Structure

```
point_quality/
├── deepresearcharena/         # Core evaluation framework
│   ├── evaluator/             # Evaluator implementations
│   │   ├── base_evaluator.py
│   │   ├── pointwise_core.py
│   │   └── pointwise_evaluator.py
│   ├── prompts/               # Prompt templates
│   ├── cache/                 # Caching system
│   ├── config/                # YAML configuration files
│   └── utils/                 # LLM calls, config loading
├── outputs/                   # Auto-created on first run; stores results and logs
├── run_batch_eval.py          # Entry point script
```

### Data Loading

Takes a model result JSON file from the shared `data/` directory as input via the `--input` flag:

```bash
cd point_quality
python run_batch_eval.py --input ../data/method_results/mirothinker_v17_text_demo.json --model_name mirothinker_v17
```

The input file is a JSON array of entries, each containing `rewritten_query`, `response`, and optional `files` for attachments. Attachment file paths in the `dir` field (e.g., `multimodal-attachments/72/attachment_72_01.jpg`) are resolved relative to `{data_dir}/input_queries/multimodal/`.

### Setup

```bash
cd point_quality

pip install openai python-dotenv pyyaml

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your OPENAI_API_KEY (or OPENROUTER_API_KEY)
```

### Usage

```bash
# Text-only evaluation
python run_batch_eval.py --input ../data/method_results/mirothinker_v17_text_demo.json --model_name mirothinker_v17

# Multimodal evaluation (attachments resolved automatically from data/input_queries/multimodal/)
python run_batch_eval.py --input ../data/method_multimodal_results/mirothinker_v17_multimodal_demo.json --model_name mirothinker_v17

# Specify evaluator model and query count
python run_batch_eval.py --input ../data/method_results/claude_text.json --model_name claude \
    --evaluator_model gpt-5.1 --max_queries 50

# Reuse criteria from a previous run (only re-score)
python run_batch_eval.py --input ../data/method_results/gemini_text.json --model_name gemini \
    --criteria_file outputs/mirothinker_v17_results.json
```

### Configuration

Configuration file located at `deepresearcharena/config/pointwise.yaml`. Key fields:

```yaml
evaluator_model:
  name: "gpt-5.1"             # Judge LLM
  api_type: "auto"             # auto (detect by model name), openai, or openrouter
  temperature: 0.1

evaluation:
  max_workers: 20              # Parallel workers
  scoring:
    score_range: [0, 10]
    decimal_places: 2
```

### Output Format

```json
{
  "summary": {
    "models": {
      "mirothinker": {
        "average_total_score": 8.807,
        "total_queries": 70,
        "dimension_averages": {
          "coverage_score": 8.5,
          "insight_score": 8.6,
          "instruction_following_score": 9.48,
          "clarity_score": 9.36
        }
      }
    }
  },
  "query_results": { ... }
}
```

---

## 3. Process Eval

Evaluates the quality of a model's research process (intermediate reasoning, search strategies, etc.) and the alignment between the process and the final report.

### How It Works

The evaluation consists of two phases:

**Phase 1 - Structuring:**

- Auto-detects different models' process trace formats (JSON array, block tags, step tags, plain text, etc.)
- Uses LLM to unify heterogeneous formats into a structured JSON schema (step list + global findings)

**Phase 2 - Evaluation:**

- **Intrinsic Evaluation**: 5 dimensions assessing the research process quality itself
- **Alignment Evaluation**: 3 dimensions assessing consistency between process and report

**8 Evaluation Dimensions:**


| Type      | Dimension              | Description                                                    |
| --------- | ---------------------- | -------------------------------------------------------------- |
| Intrinsic | search_breadth         | Diversity of sources and angles explored                       |
| Intrinsic | analytical_depth       | Depth of analysis and insight                                  |
| Intrinsic | progressive_refinement | Ability to iteratively deepen investigation                    |
| Intrinsic | critical_thinking      | Cross-verification and critical reasoning                      |
| Intrinsic | efficiency             | Conciseness and effectiveness of steps                         |
| Alignment | findings_to_report     | Fraction of process findings covered in the report             |
| Alignment | report_to_process      | Whether report claims can be traced back to the process        |
| Alignment | contradiction          | Consistency between process and report (10 = fully consistent) |


### Directory Structure

```
process_eval/
├── run_pipeline.py            # Entry point script
├── config/
│   ├── process_eval.yaml      # Text-only evaluation config
│   └── process_eval_multimodal.yaml  # Multimodal evaluation config
├── process_evaluator/         # Core package
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── data_loader.py         # Data loading
│   ├── preprocessors/         # Multi-format preprocessors (auto-detection)
│   ├── structuring/           # LLM-based structuring
│   ├── evaluation/            # Intrinsic + alignment evaluators
│   ├── cache/                 # Thread-safe JSON caching
│   └── utils/                 # LLM client, config loading
├── .env.template              # Environment variables template
└── requirements.txt
```

### Data Loading

Reads model result files directly from the shared data directory. The data loader auto-discovers files by matching `{model_name}_{data_type}*.json` pattern:

```yaml
# config/process_eval.yaml
data:
  data_dir: "../data/method_results"           # Text-only
# config/process_eval_multimodal.yaml
data:
  data_dir: "../data/method_multimodal_results" # Multimodal
```

The `process` field in each model result file contains the research process trace; the `response` field contains the final report.

### Setup

```bash
cd process_eval

pip install -r requirements.txt   # openai, pyyaml, tqdm, python-dotenv

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your OPENAI_API_KEY (or OPENROUTER_API_KEY)
```

### Usage

Supports both **text-only** and **multimodal** evaluation, selected via config file:

- Text-only (default): `config/process_eval.yaml` — reads from `data/method_results/`
- Multimodal: `config/process_eval_multimodal.yaml` — reads from `data/method_multimodal_results/`

```bash
# Text-only evaluation (default)
python run_pipeline.py

# Multimodal evaluation
python run_pipeline.py --config config/process_eval_multimodal.yaml

# Run structuring phase only
python run_pipeline.py --phase phase1

# Run evaluation phase only (requires phase1 to be completed first)
python run_pipeline.py --phase phase2

# Specify models and entry count
python run_pipeline.py --models claude gemini --max-entries 10

# Evaluate specific entry IDs with custom parallelism
python run_pipeline.py --entry-ids 1 2 3 --max-workers 4

# Clear cache and re-run
python run_pipeline.py --clear-cache
```

### Output Format

```json
{
  "summary": {
    "mirothinker": {
      "search_breadth": { "avg": 8.2, "count": 70 },
      "analytical_depth": { "avg": 7.8, "count": 70 },
      "progressive_refinement": { "avg": 8.1, "count": 70 },
      "critical_thinking": { "avg": 7.5, "count": 70 },
      "efficiency": { "avg": 7.9, "count": 70 },
      "findings_to_report": { "avg": 8.3, "count": 70 },
      "report_to_process": { "avg": 7.6, "count": 70 },
      "contradiction": { "avg": 8.8, "count": 70 },
      "intrinsic_avg": 8.1,
      "alignment_avg": 8.23,
      "overall_avg": 8.17
    }
  },
  "entry_results": { ... }
}
```

---

## Module Comparison


| Aspect            | Factual Eval                    | Point Quality               | Process Eval              |
| ----------------- | ------------------------------- | --------------------------- | ------------------------- |
| **Goal**          | Report factual correctness      | Report content quality      | Research process quality  |
| **Method**        | Agent + web search verification | LLM multi-dimension scoring | LLM structuring + scoring |
| **Data Input**    | response (report text)          | response (report text)      | process + response        |
| **Scoring Scale** | Right / Wrong / Unknown         | 0-10 continuous             | 1-10 integer              |
| **Judge LLM**     | GPT-5-mini (default)            | GPT-5.1 (default)           | GPT-5.2 (default)         |
| **Parallelism**   | Async + semaphore               | ThreadPoolExecutor          | ThreadPoolExecutor        |
| **Caching**       | None (agent state)              | Multi-level JSON cache      | Three-level JSON cache    |
| **Python**        | >= 3.11 (uv)                    | >= 3.10 (pip)               | >= 3.10 (pip)             |


## Citation

```bibtex
@misc{ye2026miroevalbenchmarkingmultimodaldeep,
      title={MiroEval: Benchmarking Multimodal Deep Research Agents in Process and Outcome},
      author={Fangda Ye and Yuxin Hu and Pengxiang Zhu and Yibo Li and Ziqi Jin and Yao Xiao and Yibo Wang and Lei Wang and Zhen Zhang and Lu Wang and Yue Deng and Bin Wang and Yifan Zhang and Liangcai Su and Xinyu Wang and He Zhao and Chen Wei and Qiang Ren and Bryan Hooi and An Bo and Shuicheng Yan and Lidong Bing},
      year={2026},
      eprint={2603.28407},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.28407},
}
```

## License

Apache-2.0
