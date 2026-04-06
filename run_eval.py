#!/usr/bin/env python3
"""
Unified MiroEval evaluation — run all three dimensions with one command.

Usage:
    python run_eval.py --input data/method_results/my_model.json --model_name my_model
    python run_eval.py --input results.json --model_name test --evaluations factual_eval point_quality
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so ``eval`` package resolves.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import config as cfg  # noqa: E402  (also loads .env)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_eval")

ALL_EVALUATIONS = ("factual_eval", "point_quality", "process_eval")


# ── helpers ────────────────────────────────────────────────────────────────


def load_input(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load entries from a JSON file.

    Accepts two formats:
      - Plain JSON array: ``[{entry}, ...]``
      - Wrapped object:   ``{"entries": [...], "model_name": "...", ...}``

    Returns ``(entries, metadata)`` where *metadata* contains any extra
    top-level keys from the wrapped format (empty dict for plain arrays).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
        meta = {k: v for k, v in data.items() if k != "entries"}
        return entries, meta
    raise ValueError(
        f"Expected JSON array or object with 'entries' key, got {type(data).__name__}"
    )


def split_entries(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split entries into text (no files) and multimodal (with files)."""
    text, multimodal = [], []
    for e in entries:
        if e.get("files"):
            multimodal.append(e)
        else:
            text.append(e)
    return text, multimodal


def merge_factual_results(
    text_res: dict[str, Any], mm_res: dict[str, Any]
) -> dict[str, Any]:
    """Merge verdict counts from text + multimodal factual runs."""
    merged_per_entry = {**text_res.get("per_entry", {}), **mm_res.get("per_entry", {})}

    # Re-compute avg_right_ratio from merged per_entry.
    ratios = [
        v["right_ratio"]
        for v in merged_per_entry.values()
        if isinstance(v, dict) and "right_ratio" in v
    ]

    return {
        "total_statements": text_res["total_statements"] + mm_res["total_statements"],
        "right": text_res["right"] + mm_res["right"],
        "wrong": text_res["wrong"] + mm_res["wrong"],
        "conflict": text_res["conflict"] + mm_res["conflict"],
        "unknown": text_res["unknown"] + mm_res["unknown"],
        "avg_right_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "per_entry": merged_per_entry,
    }


# ── main ───────────────────────────────────────────────────────────────────


async def run_all(
    entries: list[dict[str, Any]],
    model_name: str,
    evaluations: set[str],
    *,
    # factual eval
    factual_max_concurrent: int,
    factual_max_chunks: int,
    # point quality
    pq_model: str,
    pq_api_type: str,
    pq_max_workers: int,
    # process eval
    pe_model: str,
    pe_api_type: str,
    pe_max_workers: int,
) -> dict[str, Any]:
    text_entries, mm_entries = split_entries(entries)
    logger.info(
        "Entries: %d total (%d text, %d multimodal)",
        len(entries),
        len(text_entries),
        len(mm_entries),
    )

    # ── Init Hydra once (before parallel work) if factual eval requested ──
    if "factual_eval" in evaluations:
        from eval.adapters.factual_eval import init_hydra

        init_hydra()

    # ── Build coroutine dict ──────────────────────────────────────────────
    coros: dict[str, Any] = {}

    if "factual_eval" in evaluations:
        from eval.adapters.factual_eval import evaluate as factual_evaluate

        fe_kwargs = dict(
            max_concurrent=factual_max_concurrent,
            max_concurrent_chunks=factual_max_chunks,
        )
        if text_entries:
            coros["factual_text"] = asyncio.to_thread(
                factual_evaluate, text_entries, model_name,
                mode="text", **fe_kwargs,
            )
        if mm_entries:
            coros["factual_mm"] = asyncio.to_thread(
                factual_evaluate, mm_entries, model_name,
                mode="multimodal", **fe_kwargs,
            )

    if "point_quality" in evaluations:
        from eval.adapters.point_quality import evaluate as pq_evaluate

        coros["point_quality"] = asyncio.to_thread(
            pq_evaluate, entries, model_name,
            evaluator_model=pq_model,
            api_type=pq_api_type,
            max_workers=pq_max_workers,
        )

    if "process_eval" in evaluations:
        from eval.adapters.process_eval import evaluate as pe_evaluate

        coros["process_eval"] = asyncio.to_thread(
            pe_evaluate, entries, model_name,
            llm_model=pe_model,
            api_type=pe_api_type,
            max_workers=pe_max_workers,
        )

    if not coros:
        logger.warning("No evaluations to run.")
        return {"model_name": model_name, "entries_count": len(entries)}

    # ── Run concurrently ──────────────────────────────────────────────────
    results_list = await asyncio.gather(*coros.values(), return_exceptions=True)
    results_map: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for name, result in zip(coros.keys(), results_list):
        if isinstance(result, BaseException):
            tb = traceback.format_exception(type(result), result, result.__traceback__)
            errors[name] = str(result)
            logger.error("%s failed:\n%s", name, "".join(tb))
        else:
            results_map[name] = result

    # ── Assemble output ───────────────────────────────────────────────────
    empty_factual = {
        "total_statements": 0,
        "right": 0,
        "wrong": 0,
        "conflict": 0,
        "unknown": 0,
        "avg_right_ratio": 0.0,
        "per_entry": {},
    }

    output: dict[str, Any] = {
        "model_name": model_name,
        "entries_count": len(entries),
        "text_entries": len(text_entries),
        "multimodal_entries": len(mm_entries),
    }

    if "factual_eval" in evaluations:
        output["factual_eval"] = merge_factual_results(
            results_map.get("factual_text", empty_factual),
            results_map.get("factual_mm", empty_factual),
        )
    if "point_quality" in evaluations and "point_quality" in results_map:
        output["point_quality"] = results_map["point_quality"]
    if "process_eval" in evaluations and "process_eval" in results_map:
        output["process_eval"] = results_map["process_eval"]

    if errors:
        output["errors"] = errors

    return output


def print_summary(output: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print(f"  MiroEval Results — {output['model_name']}")
    print(f"  Entries: {output['entries_count']} "
          f"({output.get('text_entries', '?')} text, "
          f"{output.get('multimodal_entries', '?')} multimodal)")
    print("=" * 70)

    if "factual_eval" in output:
        fe = output["factual_eval"]
        print(f"\n  [Factual Eval]")
        print(f"    S_factual (avg right ratio): {fe['avg_right_ratio']:.4f}")
        print(f"    Statements: {fe['total_statements']} "
              f"(R={fe['right']}, W={fe['wrong']}, "
              f"U={fe['unknown']}, C={fe['conflict']})")

    if "point_quality" in output:
        pq = output["point_quality"]
        print(f"\n  [Point Quality]")
        print(f"    S_quality (avg total score): {pq['average_total_score']:.3f}")
        dims = pq.get("dimension_averages", {})
        if dims:
            for dim, score in sorted(dims.items()):
                print(f"      {dim}: {score:.3f}")

    if "process_eval" in output:
        pe = output["process_eval"]
        print(f"\n  [Process Eval]")
        print(f"    S_process (overall avg): {pe.get('overall_avg', 'N/A')}")
        if pe.get("intrinsic_avg") is not None:
            print(f"    Intrinsic avg: {pe['intrinsic_avg']:.2f}")
        if pe.get("alignment_avg") is not None:
            print(f"    Alignment avg: {pe['alignment_avg']:.2f}")

    if "errors" in output:
        print(f"\n  [Errors]")
        for name, err in output["errors"].items():
            print(f"    {name}: {err}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="MiroEval — unified three-dimensional evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to JSON array file with evaluation entries",
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Model name identifier (auto-detected from input file if wrapped format)",
    )

    # ── General ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--evaluations", nargs="+",
        default=list(ALL_EVALUATIONS), choices=ALL_EVALUATIONS,
        help="Evaluations to run",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: outputs/<model_name>_<timestamp>/)",
    )

    # ── Factual eval ──────────────────────────────────────────────────────
    fe_group = parser.add_argument_group("factual eval")
    fe_group.add_argument(
        "--factual_max_concurrent", type=int,
        default=cfg.FACTUAL_EVAL_MAX_CONCURRENT,
        help="Max parallel tasks (process-level)",
    )
    fe_group.add_argument(
        "--factual_max_chunks", type=int,
        default=cfg.FACTUAL_EVAL_MAX_CHUNKS,
        help="Max parallel chunks per task",
    )

    # ── Point quality ─────────────────────────────────────────────────────
    pq_group = parser.add_argument_group("point quality")
    pq_group.add_argument(
        "--pq_model", type=str,
        default=cfg.POINT_QUALITY_MODEL,
        help="Judge LLM for point quality",
    )
    pq_group.add_argument(
        "--pq_api_type", type=str,
        default=cfg.POINT_QUALITY_API_TYPE,
        help="API type (openai / openrouter)",
    )
    pq_group.add_argument(
        "--pq_max_workers", type=int,
        default=cfg.POINT_QUALITY_MAX_WORKERS,
        help="Max parallel workers",
    )

    # ── Process eval ──────────────────────────────────────────────────────
    pe_group = parser.add_argument_group("process eval")
    pe_group.add_argument(
        "--pe_model", type=str,
        default=cfg.PROCESS_EVAL_MODEL,
        help="Judge LLM for process eval",
    )
    pe_group.add_argument(
        "--pe_api_type", type=str,
        default=cfg.PROCESS_EVAL_API_TYPE,
        help="API type (openai / openrouter)",
    )
    pe_group.add_argument(
        "--pe_max_workers", type=int,
        default=cfg.PROCESS_EVAL_MAX_WORKERS,
        help="Max parallel workers",
    )

    args = parser.parse_args()

    # Load entries
    entries, meta = load_input(args.input)
    model_name = args.model_name or meta.get("model_name", "api_model")
    evaluations = set(args.evaluations)
    logger.info("Loaded %d entries from %s", len(entries), args.input)

    # Run evaluations
    output = asyncio.run(
        run_all(
            entries,
            model_name,
            evaluations,
            factual_max_concurrent=args.factual_max_concurrent,
            factual_max_chunks=args.factual_max_chunks,
            pq_model=args.pq_model,
            pq_api_type=args.pq_api_type,
            pq_max_workers=args.pq_max_workers,
            pe_model=args.pe_model,
            pe_api_type=args.pe_api_type,
            pe_max_workers=args.pe_max_workers,
        )
    )

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"outputs/{args.model_name}_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print_summary(output)
    print(f"\n  Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
