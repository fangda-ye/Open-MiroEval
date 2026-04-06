# Point-wise Quality Evaluation

Comprehensive Adaptive Point-wise Quality Evaluation that dynamically generates evaluation dimensions, criteria, and weights for each query task, enabling fine-grained quality assessment.

## How It Works

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

## Directory Structure

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

## Setup

If you have already run `uv sync` at the repo root, the `.venv` environment includes all dependencies needed here. You can use it directly:

```bash
.venv/bin/python point_quality/run_batch_eval.py --help
```

Alternatively, install standalone:

```bash
cd point_quality
pip install openai python-dotenv pyyaml

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your OPENAI_API_KEY (or OPENROUTER_API_KEY)
```

## Usage

```bash
cd point_quality

# Text-only evaluation
python run_batch_eval.py --input ../data/method_results/mirothinker_v17_text.json --model_name mirothinker_v17

# Multimodal evaluation (attachments resolved automatically from data/input_queries/multimodal/)
python run_batch_eval.py --input ../data/method_multimodal_results/mirothinker_v17_multimodal.json --model_name mirothinker_v17

# Specify evaluator model and query count
python run_batch_eval.py --input ../data/method_results/claude_text.json --model_name claude \
    --evaluator_model gpt-5.1 --max_queries 50

# Reuse criteria from a previous run (only re-score)
python run_batch_eval.py --input ../data/method_results/gemini_text.json --model_name gemini \
    --criteria_file outputs/mirothinker_v17_results.json
```

## Configuration

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

## Output Format

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
