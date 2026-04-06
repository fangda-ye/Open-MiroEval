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
  <img src="static/benchmark_results.png" width="100%" alt="Benchmark results across Text-Only and Multimodal evaluations" />
</div>

---

## Quick Start

### 1. Setup

All three evaluation modules share a single Python environment managed by [uv](https://docs.astral.sh/uv/) at the repo root:

```bash
uv sync
```

> If you use `run_eval.sh`, this step is done automatically on first run.

Then configure your API keys:

```bash
cp .env.template .env   # edit with your own keys
```

Required keys in `.env`:

| Variable | Used By |
|----------|---------|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | All modules (judge LLM) |
| `SERPER_API_KEY` | Factual eval (web search) |
| `JINA_API_KEY` | Factual eval (web reading) |

### 2. Prepare Input

Place your model's results as a JSON array file. Each entry should follow the schema:

```json
{
  "id": 1,
  "rewritten_query": "The evaluation query",
  "response": "Model-generated research report",
  "process": "Research process trace (for process eval)",
  "files": []
}
```

The first 70 entries (text-only, `files: []`) and last 30 entries (multimodal, with `files`) are routed automatically — text entries use the text factual-eval config, multimodal entries use the multimodal config.

### 3. Run Evaluation

```bash
# Run all three dimensions (auto-creates venv on first run)
bash run_eval.sh --input data/method_results/my_model.json --model_name my_model

# Run specific dimensions only
bash run_eval.sh --input results.json --model_name test --evaluations factual_eval point_quality

# Or call directly with the venv python
.venv/bin/python run_eval.py --input data/method_results/my_model.json --model_name my_model
```

### 4. Results

Combined results are saved to `outputs/<model_name>_<timestamp>/results.json`:

```json
{
  "model_name": "my_model",
  "entries_count": 100,
  "factual_eval": {
    "avg_right_ratio": 0.825,
    "total_statements": 1500,
    "right": 1200, "wrong": 150, "unknown": 100, "conflict": 50,
    "per_entry": { ... }
  },
  "point_quality": {
    "average_total_score": 8.5,
    "dimension_averages": { "coverage_score": 8.5, "insight_score": 8.6, ... },
    "per_entry": { ... }
  },
  "process_eval": {
    "overall_avg": 8.17,
    "intrinsic_avg": 8.1,
    "alignment_avg": 8.23,
    "dimensions": { ... },
    "per_entry": { ... }
  }
}
```

---

## Architecture

```
MiroEval/
├── pyproject.toml             # Root uv project (manages .venv for all modules)
├── .env                       # API keys (single configuration point)
├── run_eval.py                # Unified entry point (all three dimensions)
├── run_eval.sh                # Shell wrapper (auto-creates venv)
├── eval/                      # Evaluation orchestration layer
│   ├── config.py              # Path constants, env loading
│   └── adapters/              # Per-module adapters
├── data/                      # Shared data directory
│   ├── input_queries/         # Evaluation query sets + multimodal attachments
│   └── detail_results/        # Per-task per-model intermediate scores
├── factual_eval/              # Factual evaluation (MiroFlow-based fact-checking agent)
├── point_quality/             # Quality evaluation (adaptive point-wise scoring)
├── process_eval/              # Process evaluation (intrinsic process quality + report alignment)
└── task_generation/           # Evaluation task generation pipeline
```

---

## Evaluation Dimensions

| Dimension | Goal | Method | Key Metric | Details |
|-----------|------|--------|------------|---------|
| **Factual Eval** | Report factual correctness | Agent + web search verification | Right Ratio | [factual_eval/README.md](factual_eval/README.md) |
| **Point Quality** | Report content quality | LLM multi-dimension scoring (0-10) | Weighted Total Score | [point_quality/README.md](point_quality/README.md) |
| **Process Eval** | Research process quality | LLM structuring + scoring (1-10) | Overall Avg (intrinsic + alignment) | [process_eval/README.md](process_eval/README.md) |

For fine-grained single-dimension evaluation, see each module's README.

---

## Data Format

### Input Queries (`data/input_queries/`)

| File / Directory            | Description                                                                                                                                  | Count |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `mirobench_text.json`       | Text-only query set                                                                                                                          | 70    |
| `mirobench_multimodal.json` | Multimodal query set (with image/document attachments)                                                                                       | 30    |
| `multimodal-attachments/`   | Attachment files referenced by multimodal queries, organized by query ID (e.g., `72/`, `93/`). Contains images, PDFs, and other documents.   | ---     |

**Text query schema:**

```json
{
  "id": 1,
  "chat_id": "uuid",
  "rewritten_query": "Expanded/rewritten query",
  "annotation": {
    "category": "text",
    "language": "zh | en",
    "pattern": "T1 | T2 | T5 | T6",
    "domain": "tech | finance | medical | ..."
  }
}
```

**Multimodal query schema:**

```json
{
  "id": 71,
  "chat_id": "uuid",
  "rewritten_query": "Expanded/rewritten query",
  "files": [
    { "filename": "attachment_71_01.jpg", "type": "image", "dir": "multimodal-attachments/71/attachment_71_01.jpg", "size": "1.5 MB" }
  ],
  "annotation": { "category": "image | doc | multi_doc", "language": "zh | en" }
}
```

### Model Results

One JSON file per model, containing a JSON array of complete query-response pairs. Place your model's output file in `data/method_results/` (text-only) or `data/method_multimodal_results/` (multimodal).

**Result entry schema:**

```json
{
  "id": 1,
  "rewritten_query": "Rewritten query",
  "response": "Model-generated research report",
  "process": "Research process trace",
  "files": [],
  "annotation": { ... }
}
```

The `response` field contains the model's final report. The `process` field contains the intermediate research process trace (needed for process eval). Multimodal entries additionally contain a `files` field.

---

## Task Generation

Automated pipeline for generating high-quality deep-research evaluation queries. See [`task_generation/README.md`](task_generation/README.md) for full details.

---

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
