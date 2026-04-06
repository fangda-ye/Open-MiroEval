# MiroEval Refactoring Plan

## 1. Problem Statement

MiroEval currently consists of three independently developed evaluation modules bolted together with an adapter layer. This creates several issues:

- **Duplicated infrastructure** — 3 LLM clients, 2 cache implementations, 5 `.env` files, 3 config systems
- **MiroFlow embedded** — a 20K-line generic agent framework lives inside `factual_eval/`, bloating the project
- **Deeply nested packages** — `deepresearcharena/evaluator/`, `process_evaluator/evaluation/` add indirection without value
- **No incremental evaluation** — can't resume, can't check status, can't re-run a single failed dimension for one entry
- **Scattered results** — outputs land in 3 different directories with 3 different formats

## 2. Target Architecture

```
Open-MiroEval/
├── pyproject.toml                     # Single dependency manager (uv)
├── .env.template                      # Single env config (all API keys)
│
├── miroeval/                          # Main package
│   ├── __init__.py
│   │
│   ├── core/                          # Shared infrastructure (one copy)
│   │   ├── __init__.py
│   │   ├── models.py                  # TypedDict: EvalEntry, EvalResult, FileAttachment
│   │   ├── llm.py                     # Unified sync LLM client (OpenAI/OpenRouter, retry, cost)
│   │   ├── cache.py                   # Thread-safe FileCache + CacheManager
│   │   ├── config.py                  # Env loading + path constants + defaults
│   │   └── utils.py                   # extract_json, attachment helpers
│   │
│   ├── factual/                       # Factual correctness evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Wraps MiroFlow: segment → verify → merge verdicts
│   │   ├── config/                    # Hydra YAML configs (text + multimodal)
│   │   └── prompts/                   # Prompt configs for MiroFlow agents
│   │
│   ├── quality/                       # Report quality evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py              # 5-stage pipeline (dims → weights → criteria → score → aggregate)
│   │   └── prompts.py                 # 9 prompt templates
│   │
│   ├── process/                       # Research process evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Intrinsic (5 dims) + alignment (3 dims) scoring
│   │   ├── structurer.py             # Process trace → structured JSON
│   │   ├── preprocessors.py          # Multi-format detection + chunking
│   │   └── prompts.py                 # Structuring + evaluation prompts
│   │
│   ├── runner.py                      # Unified orchestrator with incremental support
│   └── cli.py                         # CLI entry point (python -m miroeval ...)
│
├── task_generation/                   # Task generation pipeline (standalone)
│   ├── pipeline.py
│   └── config.py
│
├── data/                              # Input data (unchanged)
├── outputs/                           # Unified output root (NEW)
│   └── {model_name}/                  # Per-model output directory
│       ├── manifest.json              # Incremental state tracker
│       ├── factual/                   # Per-entry factual results
│       ├── quality/                   # Per-entry quality results
│       ├── process/                   # Per-entry process results
│       └── results.json               # Aggregated 3-dimension summary
│
├── scripts/
│   └── run_eval.sh                    # Shell wrapper
│
└── static/                            # Docs assets (unchanged)
```

### What goes where

| Concern | Location | Notes |
|---------|----------|-------|
| LLM client (sync) | `miroeval/core/llm.py` | Merged from point_quality + process_eval. MiroFlow keeps its own async client internally. |
| Cache | `miroeval/core/cache.py` | Thread-safe FileCache from process_eval + CacheManager from point_quality |
| Config + env | `miroeval/core/config.py` | Single `.env`, all defaults overridable via env vars |
| Data types | `miroeval/core/models.py` | TypedDict for EvalEntry, results, attachments |
| Factual eval logic | `miroeval/factual/evaluator.py` | Thin wrapper: calls MiroFlow's `run_factual_eval_tasks`, parses verdicts |
| Quality eval logic | `miroeval/quality/evaluator.py` | Core 5-stage pipeline + hierarchical scoring (from pointwise_evaluator + pointwise_core) |
| Quality prompts | `miroeval/quality/prompts.py` | 9 templates defining evaluation semantics |
| Process eval logic | `miroeval/process/evaluator.py` | Intrinsic 5-dim + alignment 3-dim scoring |
| Process structuring | `miroeval/process/structurer.py` | Process trace → unified JSON schema |
| Process prompts | `miroeval/process/prompts.py` | Structuring + evaluation prompts |
| Orchestration | `miroeval/runner.py` | Incremental-aware runner with manifest tracking |
| CLI | `miroeval/cli.py` | `python -m miroeval run/status/retry` |

### What gets removed

| Old location | Action |
|---|---|
| `factual_eval/miroflow/` (97 files, 20K lines) | Extracted to standalone `miroflow` package (separate repo or submodule) |
| `point_quality/deepresearcharena/` | Core logic moves to `miroeval/quality/`, package deleted |
| `process_eval/process_evaluator/` | Core logic moves to `miroeval/process/`, package deleted |
| `eval/` (adapter layer) | Absorbed into `miroeval/runner.py` |
| `run_eval.py`, `run_eval.sh` | Replaced by `miroeval/cli.py` and `scripts/run_eval.sh` |
| 4 module-level `.env.template` files | Replaced by single root `.env.template` |
| `process_eval/requirements.txt` | Managed by root `pyproject.toml` |

## 3. MiroFlow Separation

MiroFlow is a generic agent framework (agents, LLM providers, MCP tools, skills, tracing). It happens to be used for factual verification, but is not MiroEval-specific.

**Separation strategy:**

1. Move `factual_eval/miroflow/` to a standalone repo (e.g., `MiroMindAI/MiroFlow`, already referenced in README)
2. In `pyproject.toml`, declare `miroflow` as a normal dependency (published package or git URL)
3. `miroeval/factual/` becomes a thin integration layer:
   - Imports from `miroflow.benchmark.factual_eval_task_runner`
   - Manages Hydra config, temp file bridging, verdict parsing
   - No agent logic lives here

**What stays in `miroeval/factual/`:**
- Hydra config files (text + multimodal variants)
- The `evaluate()` function that: writes temp data → calls MiroFlow → parses verdicts
- Verdict aggregation logic (right/wrong/conflict/unknown counting)

**What moves to MiroFlow SDK:**
- All agent implementations (iterative, sequential, rollback)
- All LLM providers (GPT, Claude, OpenRouter, MiroThinker)
- MCP server integrations (search, browse)
- Task runner infrastructure
- I/O processors (segmentation, summarization)
- Logging/tracing framework

## 4. Incremental Evaluation System

### 4.1 Manifest Design

Each model gets a `manifest.json` in `outputs/{model_name}/`:

```json
{
  "model_name": "qwen3",
  "input_file": "data/method_results/qwen3_text.json",
  "created_at": "2026-04-06T14:30:00",
  "updated_at": "2026-04-06T15:45:00",
  "config": {
    "factual": {"max_concurrent": 10, "max_chunks": 10},
    "quality": {"judge_model": "gpt-5.1", "api_type": "openai"},
    "process": {"judge_model": "gpt-5.2", "api_type": "openai"}
  },
  "entries": {
    "1":  {"factual": "completed", "quality": "completed", "process": "completed"},
    "2":  {"factual": "completed", "quality": "completed", "process": "failed"},
    "3":  {"factual": "running",   "quality": "pending",   "process": "pending"},
    "71": {"factual": "completed", "quality": "pending",   "process": "pending"}
  },
  "summary": {
    "total_entries": 70,
    "factual":  {"completed": 68, "failed": 1, "running": 1, "pending": 0},
    "quality":  {"completed": 68, "failed": 0, "running": 0, "pending": 2},
    "process":  {"completed": 67, "failed": 1, "running": 0, "pending": 2}
  }
}
```

### 4.2 Per-Entry Result Files

Each evaluation dimension writes one JSON per entry:

```
outputs/qwen3/
├── manifest.json
├── factual/
│   ├── entry_1.json    # {"statements": [...], "right": 12, "wrong": 1, ...}
│   ├── entry_2.json
│   └── summary.json    # Aggregated across all entries
├── quality/
│   ├── entry_1.json    # {"total_weighted_score": 8.5, "dimensions": {...}, ...}
│   ├── entry_2.json
│   └── summary.json
├── process/
│   ├── entry_1.json    # {"intrinsic_scores": {...}, "alignment_scores": {...}}
│   ├── entry_2.json
│   └── summary.json
└── results.json        # Combined 3-dimension final results
```

### 4.3 Evaluation Flow

```
python -m miroeval run --input data/method_results/qwen3_text.json --model qwen3
```

1. **Load entries** from input file
2. **Load or create manifest** for this model
3. **Diff** — for each (entry_id, dimension) pair:
   - `completed` → skip
   - `failed` or `pending` → add to work queue
   - New entry not in manifest → add as `pending`
4. **Execute** — run pending evaluations (parallel across dimensions, incremental within)
5. **Write results** — per-entry JSON + update manifest atomically
6. **Aggregate** — rebuild `summary.json` for each dimension + combined `results.json`

### 4.4 CLI Commands

```bash
# Full evaluation (incremental — skips completed entries)
python -m miroeval run --input data/method_results/qwen3_text.json --model qwen3

# Run only one dimension
python -m miroeval run --input data/method_results/qwen3_text.json --model qwen3 --eval quality

# Retry failed entries only
python -m miroeval retry --model qwen3
python -m miroeval retry --model qwen3 --eval factual  # only factual failures

# Check status of all models
python -m miroeval status
# Output:
#   Model        Factual     Quality     Process     Overall
#   qwen3        68/70       70/70       67/70       partial
#   claude       70/70       70/70       70/70       complete
#   gemini       0/70        0/70        0/70        pending

# Check status of one model
python -m miroeval status --model qwen3
# Output: detailed per-dimension breakdown with failed entry IDs

# Force re-run (ignore cache for specific entries)
python -m miroeval run --model qwen3 --eval quality --entries 1,2,3 --force

# Batch: run all models in a directory
python -m miroeval run --input data/method_results/ --eval quality

# Re-aggregate without re-evaluating (after fixing a scoring bug)
python -m miroeval aggregate --model qwen3
```

### 4.5 Incremental Guarantees

| Scenario | Behavior |
|----------|----------|
| Process killed mid-run | Completed entries have per-entry JSONs. Re-run picks up where it left off. |
| factual_eval fails for entry 5 | Manifest shows `"5": {"factual": "failed"}`. `retry --model qwen3` re-runs it. |
| Add a new model | New manifest created, all entries `pending`. |
| Add 30 multimodal entries to existing model | Manifest detects new entry IDs, adds as `pending`, existing entries stay `completed`. |
| Change judge model | `--force` flag or delete the dimension's summary to trigger re-eval. |
| Aggregate-only (no re-eval) | `miroeval aggregate --model qwen3` regenerates summaries from per-entry JSONs. |

## 5. Shared Infrastructure Details

### 5.1 Unified LLM Client (`miroeval/core/llm.py`)

Merges the nearly identical clients from point_quality and process_eval:

```python
class LLMClient:
    """Sync LLM client supporting OpenAI and OpenRouter APIs."""

    def __init__(self, model: str, api_type: str = "auto", **kwargs):
        # auto-detect: "/" in model name → openrouter, else openai
        ...

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate with retry + exponential backoff."""
        ...

    def generate_json(self, messages: list[dict], **kwargs) -> dict:
        """Generate and parse JSON from response (with bracket-matching extraction)."""
        ...

    @property
    def total_cost(self) -> float: ...

    @classmethod
    def global_cost(cls) -> float: ...
```

**NOT unified:** MiroFlow's async `UnifiedOpenAIClient` stays inside MiroFlow. It has entirely different needs (tool calls, message history, context windows).

### 5.2 Unified Cache (`miroeval/core/cache.py`)

```python
class FileCache:
    """Thread-safe JSON-backed cache with per-key persistence."""

    def __init__(self, cache_dir: str, name: str): ...
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...
    def has(self, key: str) -> bool: ...
    def batch_set(self, items: dict[str, Any]) -> None: ...
    def clear(self) -> None: ...

class CacheManager:
    """Manages multiple named caches under one directory."""

    def __init__(self, base_dir: str): ...
    def get_cache(self, name: str) -> FileCache: ...
```

### 5.3 Data Models (`miroeval/core/models.py`)

```python
class FileAttachment(TypedDict, total=False):
    filename: str
    type: str       # "image", "pdf", "doc", "txt"
    dir: str        # relative path from data/input_queries/
    size: str

class EvalEntry(TypedDict, total=False):
    id: int
    chat_id: str
    rewritten_query: str
    response: str
    process: str
    files: list[FileAttachment]
    annotation: dict

class EntryStatus(TypedDict):
    factual: str    # "pending" | "running" | "completed" | "failed"
    quality: str
    process: str

class Manifest(TypedDict):
    model_name: str
    input_file: str
    created_at: str
    updated_at: str
    config: dict
    entries: dict[str, EntryStatus]
    summary: dict
```

## 6. Implementation Phases

### Phase 1: Foundation (no behavioral changes)
1. Create `miroeval/core/` with models, llm, cache, config, utils
2. Add `miroeval/__init__.py` and update `pyproject.toml`
3. Verify: import `miroeval.core.llm` works, existing code untouched

### Phase 2: Migrate quality + process
4. Move point_quality core logic → `miroeval/quality/` (evaluator + prompts)
5. Move process_eval core logic → `miroeval/process/` (evaluator + structurer + preprocessors + prompts)
6. Both modules import from `miroeval.core` for LLM, cache, config
7. Verify: each module's evaluator works standalone

### Phase 3: Wrap factual + MiroFlow separation
8. Extract `factual_eval/miroflow/` → standalone package reference
9. Create `miroeval/factual/evaluator.py` (wraps MiroFlow, manages Hydra/config)
10. Move factual config/prompt files to `miroeval/factual/config/`
11. Verify: factual evaluation works via new wrapper

### Phase 4: Incremental runner
12. Implement manifest tracking (`outputs/{model}/manifest.json`)
13. Implement per-entry result persistence
14. Implement `miroeval/runner.py` with diff-based incremental logic
15. Implement `miroeval/cli.py` (run / status / retry / aggregate)

### Phase 5: Cleanup
16. Remove old directories (`point_quality/`, `process_eval/`, `eval/`, `factual_eval/` except config)
17. Update root README with new usage
18. Update all sub-READMEs
19. Consolidate `.env.template` files
20. Remove `run_eval.py` (replaced by `python -m miroeval`)

## 7. Dependency Strategy

### Root `pyproject.toml`

```toml
[project]
name = "miroeval"
version = "2.0.0"
requires-python = ">=3.11"
dependencies = [
    # Core
    "openai>=1.0",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "tqdm>=4.60",

    # Factual eval (MiroFlow)
    "miroflow>=1.6.0",    # standalone SDK, brings in hydra, mcp, etc.
]
```

**Key change:** `miroflow` is a regular dependency, not an editable local path. This cleanly separates the 87-dep agent framework from the lightweight evaluation code.

### Optional lightweight install

```toml
[project.optional-dependencies]
quality = []       # no extra deps beyond core
process = []       # no extra deps beyond core
factual = ["miroflow>=1.6.0"]  # heavy deps only if needed
```

Users who only need quality + process evaluation can skip the heavy MiroFlow install:
```bash
pip install miroeval              # quality + process only
pip install miroeval[factual]     # all three dimensions
```

## 8. Migration Compatibility

During the transition, both old and new entry points will work:

```bash
# New (recommended)
python -m miroeval run --input data/method_results/qwen3.json --model qwen3

# Old (deprecated, still works during transition)
python run_eval.py --input data/method_results/qwen3.json --model_name qwen3

# Module-specific (still works)
cd point_quality && python run_batch_eval.py --input ...
```

The old module directories (`point_quality/`, `process_eval/`) will be kept as thin shims during Phase 2-3, then removed in Phase 5.
