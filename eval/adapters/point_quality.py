"""Adapter for the point-quality evaluation module."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from eval.config import (
    DATA_DIR,
    POINT_QUALITY_CACHE,
    POINT_QUALITY_DIR,
    POINT_QUALITY_API_TYPE,
    POINT_QUALITY_MAX_WORKERS,
    POINT_QUALITY_MODEL,
)

# Add point_quality/ to sys.path so its internal imports resolve.
_pq_dir = str(POINT_QUALITY_DIR)
if _pq_dir not in sys.path:
    sys.path.insert(0, _pq_dir)

from deepresearcharena.evaluator.pointwise_evaluator import PointwiseEvaluator  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate(
    entries: list[dict[str, Any]],
    model_name: str,
    *,
    evaluator_model: str = POINT_QUALITY_MODEL,
    api_type: str = POINT_QUALITY_API_TYPE,
    max_workers: int = POINT_QUALITY_MAX_WORKERS,
) -> dict[str, Any]:
    """Run point-quality evaluation on *entries* and return results dict."""
    data_dir = str(DATA_DIR)
    cache_dir = POINT_QUALITY_CACHE
    os.makedirs(cache_dir, exist_ok=True)

    # BaseEvaluator.__init__ calls os.makedirs("outputs") with a relative path.
    # Temporarily switch CWD to point_quality/ so artefacts land there.
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(POINT_QUALITY_DIR))
        evaluator = PointwiseEvaluator(
            data_dir=data_dir,
            model_name=evaluator_model,
            api_type=api_type,
            cache_dir=cache_dir,
        )
    finally:
        os.chdir(prev_cwd)

    # Populate evaluator's queries and model_results from entries.
    skipped = 0
    for entry in entries:
        qid = entry["id"]
        task_prompt = entry.get("rewritten_query") or entry.get("query", "")
        report = entry.get("response")
        if not report:
            skipped += 1
            continue

        query_dict: dict[str, Any] = {"id": qid, "prompt": task_prompt}

        # Handle attachments from 'files' field.
        files = entry.get("files", [])
        if files:
            attachment_parts = []
            for file_info in files:
                file_path = file_info.get("dir", "")
                if file_path:
                    full_path = os.path.join(
                        data_dir, "input_queries", "multimodal", file_path
                    )
                    if os.path.isfile(full_path):
                        content = evaluator._read_attachment_file(full_path)
                        attachment_parts.append(content)
                    else:
                        logger.warning(
                            "point_quality: attachment not found: %s", full_path
                        )
            if attachment_parts:
                query_dict["attachment_parts"] = attachment_parts
                query_dict["attachment"] = "\n\n---\n\n".join(attachment_parts)

        evaluator.queries[qid] = query_dict

        if model_name not in evaluator.model_results:
            evaluator.model_results[model_name] = {}
        evaluator.model_results[model_name][qid] = report

    if skipped:
        logger.warning("point_quality: skipped %d entries without response", skipped)

    logger.info(
        "point_quality: evaluating %d queries for model '%s'",
        len(evaluator.queries),
        model_name,
    )

    results = evaluator.evaluate_all_queries(
        model_names=[model_name],
        query_selection_config={"selection_method": "first"},
        max_workers=max_workers,
    )

    # Extract summary for the response.
    summary = results.get("summary", {})
    model_summary = summary.get("models", {}).get(model_name, {})

    return {
        "average_total_score": model_summary.get("average_total_score", 0.0),
        "dimension_averages": model_summary.get("dimension_averages", {}),
        "per_entry": {
            str(qid): _extract_query_scores(qr, model_name)
            for qid, qr in results.get("query_results", {}).items()
        },
    }


def _extract_query_scores(query_result: dict, model_name: str) -> dict:
    """Pull the total + dimension scores out of a single query result."""
    mr = query_result.get("model_results", {}).get(model_name, {})
    fs = mr.get("final_scores", {})
    return {
        "total_weighted_score": fs.get("total_weighted_score", 0.0),
        "dimension_scores": {
            k: v
            for k, v in fs.items()
            if k.endswith("_score") and k != "total_weighted_score"
        },
    }
