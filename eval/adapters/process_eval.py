"""Adapter for the process-evaluation module.

Bypasses DataLoader: we call ``ProcessEvalPipeline._process_entry()`` directly
with entries supplied by the caller, then aggregate via ``_aggregate_results()``.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from eval.config import (
    PROCESS_EVAL_CACHE,
    PROCESS_EVAL_DIR,
    PROCESS_EVAL_API_TYPE,
    PROCESS_EVAL_MAX_WORKERS,
    PROCESS_EVAL_MODEL,
)

# Add process_eval/ to sys.path so its internal imports resolve.
_pe_dir = str(PROCESS_EVAL_DIR)
if _pe_dir not in sys.path:
    sys.path.insert(0, _pe_dir)

from process_evaluator.pipeline import ProcessEvalPipeline  # noqa: E402

logger = logging.getLogger(__name__)


def _build_config(
    model_name: str,
    llm_model: str,
    api_type: str,
    max_workers: int,
) -> dict:
    """Build a config dict that ProcessEvalPipeline accepts."""
    return {
        "data": {
            # DataLoader is never called — just needs a valid key.
            "data_dir": "/tmp/miroeval_dummy",
            "data_type": "text",
        },
        "target_models": [model_name],
        "llm": {
            "model": llm_model,
            "api_type": api_type,
            "max_tokens": 8192,
            "temperature": 0.1,
            "retry_count": 3,
            "retry_backoff": 2.0,
        },
        "preprocessing": {
            "max_chars": 50000,
            "report_max_chars": 30000,
        },
        "execution": {
            "max_workers": max_workers,
            "continue_on_error": True,
        },
        "entry_selection": {},
        "cache": {
            "enabled": True,
            "cache_dir": PROCESS_EVAL_CACHE,
        },
        "output": {
            "results_dir": str(PROCESS_EVAL_DIR / "outputs" / "results"),
            "results_file": str(
                PROCESS_EVAL_DIR / "outputs" / "results" / "process_eval_results.json"
            ),
        },
    }


def evaluate(
    entries: list[dict[str, Any]],
    model_name: str,
    *,
    llm_model: str = PROCESS_EVAL_MODEL,
    api_type: str = PROCESS_EVAL_API_TYPE,
    max_workers: int = PROCESS_EVAL_MAX_WORKERS,
) -> dict[str, Any]:
    """Run process evaluation on *entries* and return results dict."""
    config = _build_config(model_name, llm_model, api_type, max_workers)
    pipeline = ProcessEvalPipeline(config)

    # Process entries directly — bypassing DataLoader.load_all().
    results: dict[str, Any] = {}
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for entry in entries:
            entry_dict = dict(entry)  # ensure plain dict
            entry_id = entry_dict.get("id", "?")
            key = f"{model_name}_{entry_id}"
            future = executor.submit(pipeline._process_entry, model_name, entry_dict)
            future_to_key[future] = key

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
                if result:
                    results[key] = result
                else:
                    failed += 1
            except Exception as e:
                logger.error("process_eval: error processing %s: %s", key, e)
                failed += 1

    logger.info("process_eval: completed %d, failed %d", len(results), failed)

    # Aggregate into summary.
    final = pipeline._aggregate_results(results)

    # Reshape for the response.
    model_summary = final.get("summary", {}).get(model_name, {})
    return {
        "intrinsic_avg": model_summary.get("intrinsic_avg"),
        "alignment_avg": model_summary.get("alignment_avg"),
        "overall_avg": model_summary.get("overall_avg"),
        "dimensions": {
            k: v
            for k, v in model_summary.items()
            if k not in ("intrinsic_avg", "alignment_avg", "overall_avg")
        },
        "per_entry": {
            k: {
                "intrinsic_scores": v.get("intrinsic_scores"),
                "alignment_scores": v.get("alignment_scores"),
            }
            for k, v in final.get("entry_results", {}).items()
        },
    }
