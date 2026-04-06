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

```bash
# Install (conda recommended)
conda create -n miroeval python=3.11
conda activate miroeval
pip install -e .                    # quality + process eval
pip install -e ".[factual]"         # + factual eval (heavier deps)

# Configure API keys
cp .env.template .env               # edit with your keys
```

Required keys in `.env`:

| Variable | Used By |
|----------|---------|
| `OPENAI_API_KEY` | All modules (judge LLM) |
| `SERPER_API_KEY` | Factual eval (web search) |
| `JINA_API_KEY` | Factual eval (web reading) |

### 2. Run Evaluation

```bash
# Run all three dimensions (incremental — skips already-completed entries)
python -m miroeval run --input data/method_results/my_model.json --model my_model

# Run only specific dimensions
python -m miroeval run --input data/method_results/my_model.json --model my_model --eval quality process

# Retry failed entries
python -m miroeval retry --model my_model

# Check evaluation progress across all models
python -m miroeval status

# Re-aggregate results without re-evaluating
python -m miroeval aggregate --model my_model
```

### 3. Results

Results are saved incrementally to `outputs/<model_name>/`:

```
outputs/my_model/
├── manifest.json           # Per-entry status tracking (pending/completed/failed)
├── factual/
│   ├── entry_1.json        # Per-entry factual verdicts
│   └── summary.json        # Aggregated factual scores
├── quality/
│   ├── entry_1.json        # Per-entry quality scores
│   └── summary.json
├── process/
│   ├── entry_1.json        # Per-entry process scores
│   └── summary.json
└── results.json            # Combined 3-dimension summary
```

The evaluation is **incremental** — if a run is interrupted, re-running the same command automatically picks up where it left off. Only pending or failed entries are evaluated.

---

## Architecture

```
MiroEval/
├── pyproject.toml                 # Dependencies (miroflow optional for factual)
├── .env.template                  # Single environment config for all modules
│
├── miroeval/                      # Main evaluation package
│   ├── core/                      # Shared infrastructure
│   │   ├── llm.py                 # Unified LLM client (OpenAI/OpenRouter)
│   │   ├── cache.py               # Thread-safe file-backed cache
│   │   ├── config.py              # Centralized configuration
│   │   ├── models.py              # TypedDict data models
│   │   └── utils.py               # JSON extraction, data loading
│   ├── factual/                   # Factual eval (wraps MiroFlow)
│   ├── quality/                   # Report quality eval (5-stage pipeline)
│   ├── process/                   # Process quality eval (8 dimensions)
│   ├── runner.py                  # Incremental orchestrator with manifest tracking
│   └── cli.py                     # CLI entry point
│
├── factual_eval/                  # MiroFlow agent framework (standalone SDK)
├── task_generation/               # Evaluation task generation pipeline
├── data/                          # Shared data directory
│   ├── input_queries/             # Evaluation query sets + attachments
│   └── detail_results/            # Benchmark result scores
└── outputs/                       # Per-model evaluation outputs (auto-created)
```

---

## Evaluation Dimensions

| Dimension | Goal | Method | Key Metric |
|-----------|------|--------|------------|
| **Factual** | Report factual correctness | Agent + web search verification | Right Ratio |
| **Quality** | Report content quality | LLM adaptive point-wise scoring (0-10) | Weighted Total Score |
| **Process** | Research process quality | LLM structuring + 8-dimension scoring (1-10) | Overall Avg |

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
