# Process Eval

Evaluates the quality of a model's research process (intermediate reasoning, search strategies, etc.) and the alignment between the process and the final report.

## How It Works

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

## Directory Structure

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

## Setup

If you have already run `uv sync` at the repo root, the `.venv` environment includes all dependencies needed here. You can use it directly:

```bash
.venv/bin/python process_eval/run_pipeline.py --help
```

Alternatively, install standalone:

```bash
cd process_eval
pip install -r requirements.txt   # openai, pyyaml, tqdm, python-dotenv

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your OPENAI_API_KEY (or OPENROUTER_API_KEY)
```

## Usage

Supports both **text-only** and **multimodal** evaluation, selected via config file:

- Text-only (default): `config/process_eval.yaml` -- reads from `data/method_results/`
- Multimodal: `config/process_eval_multimodal.yaml` -- reads from `data/method_multimodal_results/`

```bash
cd process_eval

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

## Output Format

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
