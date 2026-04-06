"""Adapter for the factual-evaluation module.

Strategy: write API entries as a temp JSON file inside ``factual_eval/``,
``cd`` into that directory (so all relative config/prompt/tool paths resolve
naturally), run the evaluation, collect results, and clean up the temp file.

Automatically selects the text or multimodal config via the *mode* parameter.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from eval.config import (
    FACTUAL_EVAL_DIR,
    FACTUAL_EVAL_MAX_CHUNKS,
    FACTUAL_EVAL_MAX_CONCURRENT,
)

logger = logging.getLogger(__name__)

_TEXT_CONFIG = "benchmark_factual-eval_text"
_MULTIMODAL_CONFIG = "benchmark_factual-eval_multimodal"

# ── Hydra singleton guard ──────────────────────────────────────────────────
_cfg_text: DictConfig | None = None
_cfg_multimodal: DictConfig | None = None


def init_hydra() -> None:
    """One-time Hydra initialisation.  Must be called before ``evaluate()``.

    Loads both text and multimodal config templates so that ``evaluate()``
    can pick the right one per request.
    """
    global _cfg_text, _cfg_multimodal

    if _cfg_text is not None:
        return

    # Add factual_eval/ to sys.path so its internal imports resolve.
    _fe_dir = str(FACTUAL_EVAL_DIR)
    if _fe_dir not in sys.path:
        sys.path.insert(0, _fe_dir)

    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # cd into factual_eval/ so that Hydra resolves relative paths correctly.
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(FACTUAL_EVAL_DIR))
        from config import load_config  # factual_eval/config/__init__.py

        # 1) Load text config (this also calls hydra.initialize_config_dir).
        _cfg_text = load_config(_TEXT_CONFIG)

        # 2) Load multimodal config — Hydra is already initialised, so we
        #    only need hydra.compose (no second initialize_config_dir).
        import hydra

        raw = hydra.compose(config_name=_MULTIMODAL_CONFIG)
        resolved = OmegaConf.to_container(raw, resolve=True)
        _cfg_multimodal = OmegaConf.create(resolved)
        # Ensure output_dir is absolute (same as text config).
        _cfg_multimodal.output_dir = _cfg_text.output_dir
    finally:
        os.chdir(prev_cwd)

    logger.info("factual_eval: Hydra initialised (text + multimodal configs)")


def evaluate(
    entries: list[dict[str, Any]],
    model_name: str,
    *,
    mode: str = "text",
    max_concurrent: int = FACTUAL_EVAL_MAX_CONCURRENT,
    max_concurrent_chunks: int = FACTUAL_EVAL_MAX_CHUNKS,
) -> dict[str, Any]:
    """Run factual evaluation and return aggregated verdict counts.

    *mode* must be ``"text"`` or ``"multimodal"`` — selects the Hydra config.
    """
    if not entries or _cfg_text is None:
        return _empty_result()

    cfg_template = _cfg_multimodal if mode == "multimodal" else _cfg_text
    logger.info("factual_eval: using %s config (%d entries)", mode, len(entries))

    # ── 1. Write temp data file ────────────────────────────────────────────
    tmp_dir = FACTUAL_EVAL_DIR / "data" / "_api_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filename = f"eval_{model_name}_{uuid.uuid4().hex[:8]}.json"
    tmp_path = tmp_dir / tmp_filename
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)

        # ── 2. cd into factual_eval/ ───────────────────────────────────────
        prev_cwd = os.getcwd()
        os.chdir(str(FACTUAL_EVAL_DIR))
        try:
            return _run_in_factual_dir(
                cfg_template,
                tmp_path,
                model_name,
                max_concurrent,
                max_concurrent_chunks,
            )
        finally:
            os.chdir(prev_cwd)
    finally:
        # ── 5. Clean up temp file ──────────────────────────────────────────
        tmp_path.unlink(missing_ok=True)


def _run_in_factual_dir(
    cfg_template: DictConfig,
    source_path: Path,
    model_name: str,
    max_concurrent: int,
    max_concurrent_chunks: int,
) -> dict[str, Any]:
    """Execute factual eval while CWD is ``factual_eval/``."""
    # Lazy imports — they depend on sys.path containing factual_eval/.
    from miroflow.benchmark.eval_utils import Task, TaskResult, STATUS_FAILED
    from miroflow.benchmark.factual_eval_task_runner import run_factual_eval_tasks
    from miroflow.benchmark.run_factual_eval import _load_json_array_tasks
    from miroflow.logging.task_tracer import set_tracer

    # Per-request config copy with unique output dir.
    cfg_dict = OmegaConf.to_container(cfg_template, resolve=True)
    cfg = OmegaConf.create(cfg_dict)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(FACTUAL_EVAL_DIR / "logs" / "factual-eval" / f"run_{ts}")
    cfg.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    set_tracer(output_dir)

    # Load tasks from the temp JSON file.
    tasks = _load_json_array_tasks(source_path)
    if not tasks:
        return _empty_result()

    logger.info(
        "factual_eval: running %d tasks (concurrent=%d, chunks=%d)",
        len(tasks),
        max_concurrent,
        max_concurrent_chunks,
    )

    execution_cfg = cfg.benchmark.execution
    results: list[TaskResult] = run_factual_eval_tasks(
        cfg=cfg,
        tasks=tasks,
        max_concurrent=max_concurrent,
        max_concurrent_chunks=max_concurrent_chunks,
        max_retries=execution_cfg.get("max_retries", 5),
        chunk_timeout=execution_cfg.get("chunk_timeout", None),
        chunk_tracing=execution_cfg.get("chunk_tracing", True),
    )

    return _aggregate(results)


# ── helpers ────────────────────────────────────────────────────────────────


def _aggregate(results: list) -> dict[str, Any]:
    """Count verdicts across all completed tasks."""
    right = wrong = conflict = unknown = total = 0
    task_rrs: list[float] = []
    per_entry: dict[str, Any] = {}

    for r in results:
        task_id = r.task.task_id if hasattr(r, "task") else r.get("task_id", "?")
        entry_id = str(task_id).rsplit("/", 1)[-1]

        model_response = (
            r.model_response
            if hasattr(r, "model_response")
            else r.get("model_response", "")
        )
        status = r.status if hasattr(r, "status") else r.get("status", "")

        if status != "completed" or not model_response:
            per_entry[entry_id] = {
                "status": status,
                "error": getattr(r, "error_message", ""),
            }
            continue

        try:
            core_state = json.loads(model_response)
            if isinstance(core_state, dict):
                core_state = core_state.get("core_state", [])
        except json.JSONDecodeError:
            per_entry[entry_id] = {"status": "parse_error"}
            continue

        entry_counts = {"right": 0, "wrong": 0, "conflict": 0, "unknown": 0}
        for stmt in core_state:
            verdict = stmt.get("verification", "").strip().lower()
            if verdict == "right":
                entry_counts["right"] += 1
            elif verdict == "wrong":
                entry_counts["wrong"] += 1
            elif verdict == "conflict":
                entry_counts["conflict"] += 1
            else:
                entry_counts["unknown"] += 1

        n = sum(entry_counts.values())
        entry_counts["total"] = n
        denom = (
            entry_counts["right"]
            + entry_counts["wrong"]
            + entry_counts["unknown"]
            + entry_counts["conflict"]
        )
        entry_counts["right_ratio"] = (
            entry_counts["right"] / denom if denom > 0 else 0.0
        )
        per_entry[entry_id] = entry_counts

        right += entry_counts["right"]
        wrong += entry_counts["wrong"]
        conflict += entry_counts["conflict"]
        unknown += entry_counts["unknown"]
        total += n
        task_rrs.append(entry_counts["right_ratio"])

    return {
        "total_statements": total,
        "right": right,
        "wrong": wrong,
        "conflict": conflict,
        "unknown": unknown,
        "avg_right_ratio": sum(task_rrs) / len(task_rrs) if task_rrs else 0.0,
        "per_entry": per_entry,
    }


def _empty_result() -> dict[str, Any]:
    return {
        "total_statements": 0,
        "right": 0,
        "wrong": 0,
        "conflict": 0,
        "unknown": 0,
        "avg_right_ratio": 0.0,
        "per_entry": {},
    }
