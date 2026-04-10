"""Incremental evaluation runner with manifest-based state tracking.

Key features:
- Per-entry x per-dimension state tracking via manifest.json
- Manifest updated after EACH entry completes (not batch-at-end)
- Automatic resume: skips completed entries, re-runs pending/failed
- Per-entry result persistence (one JSON per entry per dimension)
- Failure reasons recorded in per-entry result files
- Aggregation from per-entry results into dimension summaries
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from statistics import mean
from typing import Any

from miroeval.core.config import DATA_DIR, OUTPUTS_DIR
from miroeval.core.utils import load_entries, split_entries

logger = logging.getLogger(__name__)

ALL_DIMENSIONS = ("factual", "quality", "process")

# Lock for thread-safe manifest writes (evaluators use ThreadPoolExecutor).
_manifest_lock = threading.Lock()


# ── Manifest management ──────────────────────────────────────────────────


def _model_dir(model_name: str) -> str:
    return os.path.join(str(OUTPUTS_DIR), model_name)


def _manifest_path(model_name: str) -> str:
    return os.path.join(_model_dir(model_name), "manifest.json")


def load_manifest(model_name: str) -> dict[str, Any] | None:
    path = _manifest_path(model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_manifest(manifest: dict[str, Any]) -> None:
    model_name = manifest["model_name"]
    d = _model_dir(model_name)
    os.makedirs(d, exist_ok=True)
    manifest["updated_at"] = datetime.now().isoformat()
    manifest["summary"] = _build_summary(manifest["entries"])
    with _manifest_lock:
        with open(_manifest_path(model_name), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def _build_summary(entries: dict[str, dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total_entries": len(entries)}
    for dim in ALL_DIMENSIONS:
        counts: dict[str, int] = {}
        for status in entries.values():
            s = status.get(dim, "pending")
            counts[s] = counts.get(s, 0) + 1
        summary[dim] = counts
    return summary


def create_manifest(
    model_name: str,
    entries: list[dict[str, Any]],
    input_file: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new manifest or merge new entries into an existing one."""
    existing = load_manifest(model_name)

    if existing is not None:
        for e in entries:
            eid = str(e["id"])
            if eid not in existing["entries"]:
                existing["entries"][eid] = {
                    "factual": "pending",
                    "quality": "pending",
                    "process": "pending",
                }
        existing["input_file"] = input_file
        if config:
            existing["config"] = config
        save_manifest(existing)
        return existing

    entry_statuses = {}
    for e in entries:
        entry_statuses[str(e["id"])] = {
            "factual": "pending",
            "quality": "pending",
            "process": "pending",
        }

    manifest = {
        "model_name": model_name,
        "input_file": input_file,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "config": config or {},
        "entries": entry_statuses,
        "summary": _build_summary(entry_statuses),
    }
    save_manifest(manifest)
    return manifest


def _update_entry_status(
    manifest: dict[str, Any], entry_id: str, dim: str, status: str
) -> None:
    """Update a single entry's status and persist manifest immediately."""
    manifest["entries"].setdefault(entry_id, {})[dim] = status
    save_manifest(manifest)


# ── Per-entry result I/O ─────────────────────────────────────────────────


def _entry_result_path(model_name: str, dim: str, entry_id: int | str) -> str:
    return os.path.join(_model_dir(model_name), dim, f"entry_{entry_id}.json")


def save_entry_result(
    model_name: str, dim: str, entry_id: int | str, result: dict[str, Any]
) -> None:
    path = _entry_result_path(model_name, dim, entry_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_entry_result(
    model_name: str, dim: str, entry_id: int | str
) -> dict[str, Any] | None:
    path = _entry_result_path(model_name, dim, entry_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Dimension summaries ──────────────────────────────────────────────────


def _summary_path(model_name: str, dim: str) -> str:
    return os.path.join(_model_dir(model_name), dim, "summary.json")


def save_dim_summary(model_name: str, dim: str, summary: dict[str, Any]) -> None:
    path = _summary_path(model_name, dim)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _combined_results_path(model_name: str) -> str:
    return os.path.join(_model_dir(model_name), "results.json")


def save_combined_results(model_name: str, results: dict[str, Any]) -> None:
    path = _combined_results_path(model_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── Runner ────────────────────────────────────────────────────────────────


def get_pending_entries(
    manifest: dict[str, Any],
    dim: str,
    *,
    force_ids: set[str] | None = None,
) -> list[str]:
    """Get entry IDs that need evaluation for a given dimension.

    If *force_ids* is given, ONLY those IDs are returned (regardless of status).
    Otherwise, returns all pending/failed entries.
    """
    if force_ids:
        return [eid for eid in force_ids if eid in manifest["entries"]]

    return [
        eid for eid, status in manifest["entries"].items()
        if status.get(dim) in ("pending", "failed")
    ]


def run_evaluation(
    input_path: str,
    model_name: str,
    *,
    dimensions: set[str] | None = None,
    force_ids: set[str] | None = None,
    # Quality params
    quality_model: str = "gpt-5.1",
    quality_api_type: str = "openai",
    quality_max_workers: int = 20,
    # Process params
    process_model: str = "gpt-5.2",
    process_api_type: str = "openai",
    process_max_workers: int = 10,
    # Factual params
    factual_max_concurrent: int = 10,
    factual_max_chunks: int = 10,
) -> dict[str, Any]:
    """Run incremental evaluation.

    Only evaluates entries/dimensions that are pending or failed.
    Completed entries are skipped automatically.
    Manifest is updated after EACH entry completes.
    """
    dims = dimensions or set(ALL_DIMENSIONS)

    entries, _meta = load_entries(input_path)
    entries_by_id = {str(e["id"]): e for e in entries}

    manifest = create_manifest(model_name, entries, input_path)

    logger.info(
        "Model %s: %d entries, dimensions: %s",
        model_name, len(entries), dims,
    )

    if "quality" in dims:
        pending = get_pending_entries(manifest, "quality", force_ids=force_ids)
        if pending:
            logger.info("Quality: %d entries to evaluate", len(pending))
            _run_quality(
                model_name, manifest, entries_by_id, pending,
                model=quality_model, api_type=quality_api_type,
                max_workers=quality_max_workers,
            )
        else:
            logger.info("Quality: all entries completed, skipping")

    if "process" in dims:
        pending = get_pending_entries(manifest, "process", force_ids=force_ids)
        if pending:
            logger.info("Process: %d entries to evaluate", len(pending))
            _run_process(
                model_name, manifest, entries_by_id, pending,
                model=process_model, api_type=process_api_type,
                max_workers=process_max_workers,
            )
        else:
            logger.info("Process: all entries completed, skipping")

    if "factual" in dims:
        pending = get_pending_entries(manifest, "factual", force_ids=force_ids)
        if pending:
            logger.info("Factual: %d entries to evaluate", len(pending))
            _run_factual(
                model_name, manifest, entries_by_id, pending,
                max_concurrent=factual_max_concurrent,
                max_chunks=factual_max_chunks,
            )
        else:
            logger.info("Factual: all entries completed, skipping")

    combined = aggregate(model_name)
    return combined


def _run_quality(
    model_name: str,
    manifest: dict[str, Any],
    entries_by_id: dict[str, dict[str, Any]],
    pending_ids: list[str],
    *,
    model: str,
    api_type: str,
    max_workers: int,
) -> None:
    from miroeval.quality import QualityEvaluator

    evaluator = QualityEvaluator(
        model=model,
        api_type=api_type,
        cache_dir=os.path.join(_model_dir(model_name), ".cache", "quality"),
        data_dir=str(DATA_DIR),
    )

    def _on_done(eid_str: str, result: dict | None, error: str | None) -> None:
        if result is not None:
            save_entry_result(model_name, "quality", eid_str, result)
            _update_entry_status(manifest, eid_str, "quality", "completed")
        else:
            save_entry_result(model_name, "quality", eid_str, {"error": error or "unknown"})
            _update_entry_status(manifest, eid_str, "quality", "failed")

    pending_entries = [entries_by_id[eid] for eid in pending_ids if eid in entries_by_id]
    results = evaluator.evaluate_batch(
        pending_entries, max_workers=max_workers, on_entry_done=_on_done
    )

    if results.get("summary"):
        save_dim_summary(model_name, "quality", results["summary"])


def _run_process(
    model_name: str,
    manifest: dict[str, Any],
    entries_by_id: dict[str, dict[str, Any]],
    pending_ids: list[str],
    *,
    model: str,
    api_type: str,
    max_workers: int,
) -> None:
    from miroeval.process import ProcessEvaluator

    evaluator = ProcessEvaluator(
        model=model,
        api_type=api_type,
        cache_dir=os.path.join(_model_dir(model_name), ".cache", "process"),
    )

    def _on_done(key: str, result: dict | None, error: str | None) -> None:
        # key is "{model_name}_{entry_id}"
        eid_str = key.split("_", 1)[-1] if "_" in key else key
        if result is not None:
            save_entry_result(model_name, "process", eid_str, result)
            _update_entry_status(manifest, eid_str, "process", "completed")
        else:
            save_entry_result(model_name, "process", eid_str, {"error": error or "unknown"})
            _update_entry_status(manifest, eid_str, "process", "failed")

    pending_entries = [entries_by_id[eid] for eid in pending_ids if eid in entries_by_id]
    results = evaluator.evaluate_batch(
        pending_entries, model_name, max_workers=max_workers, on_entry_done=_on_done
    )

    if results.get("summary"):
        save_dim_summary(model_name, "process", results["summary"])


def _run_factual(
    model_name: str,
    manifest: dict[str, Any],
    entries_by_id: dict[str, dict[str, Any]],
    pending_ids: list[str],
    *,
    max_concurrent: int,
    max_chunks: int,
) -> None:
    from miroeval.factual import FactualEvaluator

    evaluator = FactualEvaluator(
        max_concurrent=max_concurrent,
        max_concurrent_chunks=max_chunks,
    )

    # Split into text and multimodal for separate MiroFlow configs.
    pending_entries = [entries_by_id[eid] for eid in pending_ids if eid in entries_by_id]
    text_entries, mm_entries = split_entries(pending_entries)

    per_entry: dict[str, Any] = {}

    if text_entries:
        text_res = evaluator.evaluate_batch(text_entries, model_name, mode="text")
        per_entry.update(text_res.get("per_entry", {}))

    if mm_entries:
        mm_res = evaluator.evaluate_batch(mm_entries, model_name, mode="multimodal")
        per_entry.update(mm_res.get("per_entry", {}))

    # Update manifest per-entry (factual runs as a batch in MiroFlow,
    # so we update after the batch completes).
    for eid_str, result in per_entry.items():
        save_entry_result(model_name, "factual", eid_str, result)
        if isinstance(result, dict) and "right" in result:
            _update_entry_status(manifest, eid_str, "factual", "completed")
        else:
            _update_entry_status(manifest, eid_str, "factual", "failed")

    # Mark entries that didn't appear in results as failed.
    for eid in pending_ids:
        if eid not in per_entry:
            _update_entry_status(manifest, eid, "factual", "failed")


# ── Aggregation ───────────────────────────────────────────────────────────


def aggregate(model_name: str) -> dict[str, Any]:
    """Rebuild summaries from per-entry result files."""
    manifest = load_manifest(model_name)
    if manifest is None:
        return {"model_name": model_name, "error": "No manifest found"}

    combined: dict[str, Any] = {
        "model_name": model_name,
        "updated_at": datetime.now().isoformat(),
    }

    # Quality
    quality_results = _load_all_entry_results(model_name, "quality", manifest)
    if quality_results:
        scores = [r["total_weighted_score"] for r in quality_results.values() if r.get("total_weighted_score", 0) > 0]
        combined["quality"] = {
            "average_total_score": mean(scores) if scores else 0.0,
            "total_queries": len(scores),
        }
        save_dim_summary(model_name, "quality", combined["quality"])

    # Process
    process_results = _load_all_entry_results(model_name, "process", manifest)
    if process_results:
        from miroeval.process.evaluator import INTRINSIC_DIMS, ALIGNMENT_DIMS
        dim_scores: dict[str, list[float]] = {}
        for r in process_results.values():
            for key, dims in [("intrinsic_scores", INTRINSIC_DIMS), ("alignment_scores", ALIGNMENT_DIMS)]:
                scores_dict = r.get(key)
                if not scores_dict:
                    continue
                for dim in dims:
                    if dim in scores_dict and isinstance(scores_dict[dim], dict):
                        s = scores_dict[dim].get("score")
                        if s is not None:
                            dim_scores.setdefault(dim, []).append(float(s))

        intr = [mean(dim_scores[d]) for d in INTRINSIC_DIMS if d in dim_scores]
        algn = [mean(dim_scores[d]) for d in ALIGNMENT_DIMS if d in dim_scores]
        combined["process"] = {
            "intrinsic_avg": mean(intr) if intr else None,
            "alignment_avg": mean(algn) if algn else None,
            "overall_avg": mean(intr + algn) if (intr or algn) else None,
        }
        save_dim_summary(model_name, "process", combined["process"])

    # Factual
    factual_results = _load_all_entry_results(model_name, "factual", manifest)
    if factual_results:
        right = wrong = conflict = unknown = total = 0
        ratios: list[float] = []
        for r in factual_results.values():
            if isinstance(r, dict) and "right" in r:
                right += r["right"]
                wrong += r["wrong"]
                conflict += r.get("conflict", 0)
                unknown += r.get("unknown", 0)
                total += r.get("total", 0)
                if "right_ratio" in r:
                    ratios.append(r["right_ratio"])
        combined["factual"] = {
            "total_statements": total,
            "right": right,
            "wrong": wrong,
            "conflict": conflict,
            "unknown": unknown,
            "avg_right_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        }
        save_dim_summary(model_name, "factual", combined["factual"])

    combined["summary"] = manifest.get("summary", {})
    save_combined_results(model_name, combined)
    return combined


def _load_all_entry_results(
    model_name: str, dim: str, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Load completed entry results, skipping entries with errors."""
    results = {}
    for eid, status in manifest["entries"].items():
        if status.get(dim) == "completed":
            r = load_entry_result(model_name, dim, eid)
            if r is not None and "error" not in r:
                results[eid] = r
    return results


# ── Status ────────────────────────────────────────────────────────────────


def get_status(model_name: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    """Get evaluation status for one or all models."""
    if model_name:
        manifest = load_manifest(model_name)
        if manifest is None:
            return {"model_name": model_name, "status": "not_found"}
        return {
            "model_name": model_name,
            "summary": manifest.get("summary", {}),
            "input_file": manifest.get("input_file"),
            "updated_at": manifest.get("updated_at"),
        }

    results = []
    outputs = str(OUTPUTS_DIR)
    if not os.path.exists(outputs):
        return results
    for name in sorted(os.listdir(outputs)):
        manifest_file = os.path.join(outputs, name, "manifest.json")
        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                m = json.load(f)
            results.append({
                "model_name": name,
                "summary": m.get("summary", {}),
                "updated_at": m.get("updated_at"),
            })
    return results
