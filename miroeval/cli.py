"""MiroEval CLI — unified evaluation entry point.

Usage:
    python -m miroeval run --input data/method_results/qwen3.json --model qwen3
    python -m miroeval run --model qwen3 --eval quality
    python -m miroeval retry --model qwen3
    python -m miroeval status
    python -m miroeval status --model qwen3
    python -m miroeval aggregate --model qwen3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from miroeval.core.config import (
    FACTUAL_MAX_CHUNKS,
    FACTUAL_MAX_CONCURRENT,
    PROCESS_API_TYPE,
    PROCESS_MAX_WORKERS,
    PROCESS_MODEL,
    QUALITY_API_TYPE,
    QUALITY_MAX_WORKERS,
    QUALITY_MODEL,
)

logger = logging.getLogger("miroeval")


def _setup_logging(model_name: str | None = None) -> None:
    """Configure logging to stderr + file (if model_name given)."""
    from miroeval.core.config import OUTPUTS_DIR

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # stderr handler (always).
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # File handler (per-model eval log).
    if model_name:
        log_dir = os.path.join(str(OUTPUTS_DIR), model_name)
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, "eval.log"), encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


def cmd_run(args: argparse.Namespace) -> None:
    """Run incremental evaluation."""
    _setup_logging(args.model)
    from miroeval.runner import run_evaluation

    dims = set(args.eval) if args.eval else None
    force_ids = set(args.entries.split(",")) if args.entries else None

    result = run_evaluation(
        input_path=args.input,
        model_name=args.model,
        dimensions=dims,
        force_ids=force_ids,
        quality_model=args.quality_model,
        quality_api_type=args.quality_api_type,
        quality_max_workers=args.quality_max_workers,
        process_model=args.process_model,
        process_api_type=args.process_api_type,
        process_max_workers=args.process_max_workers,
        factual_max_concurrent=args.factual_max_concurrent,
        factual_max_chunks=args.factual_max_chunks,
    )

    _print_result_summary(result)


def cmd_retry(args: argparse.Namespace) -> None:
    """Retry failed entries."""
    _setup_logging(args.model)
    from miroeval.runner import load_manifest, run_evaluation

    manifest = load_manifest(args.model)
    if manifest is None:
        print(f"No manifest found for model '{args.model}'.")
        sys.exit(1)

    dims = set(args.eval) if args.eval else None

    # Find failed entry IDs.
    failed_ids: set[str] = set()
    for eid, status in manifest["entries"].items():
        for dim in (dims or {"factual", "quality", "process"}):
            if status.get(dim) == "failed":
                failed_ids.add(eid)

    if not failed_ids:
        print(f"No failed entries for model '{args.model}'.")
        return

    print(f"Retrying {len(failed_ids)} failed entries...")

    result = run_evaluation(
        input_path=manifest["input_file"],
        model_name=args.model,
        dimensions=dims,
        force_ids=failed_ids,
        quality_model=args.quality_model,
        quality_api_type=args.quality_api_type,
        quality_max_workers=args.quality_max_workers,
        process_model=args.process_model,
        process_api_type=args.process_api_type,
        process_max_workers=args.process_max_workers,
        factual_max_concurrent=args.factual_max_concurrent,
        factual_max_chunks=args.factual_max_chunks,
    )
    _print_result_summary(result)


def cmd_status(args: argparse.Namespace) -> None:
    """Show evaluation status."""
    _setup_logging()
    from miroeval.runner import get_status

    if args.model:
        status = get_status(args.model)
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        statuses = get_status()
        if not statuses:
            print("No evaluations found in outputs/.")
            return

        # Table format.
        header = f"{'Model':<25} {'Factual':<15} {'Quality':<15} {'Process':<15}"
        print(header)
        print("-" * len(header))
        for s in statuses:
            name = s["model_name"]
            summary = s.get("summary", {})
            total = summary.get("total_entries", 0)

            def _dim_str(dim: str) -> str:
                d = summary.get(dim, {})
                c = d.get("completed", 0)
                return f"{c}/{total}"

            print(f"{name:<25} {_dim_str('factual'):<15} {_dim_str('quality'):<15} {_dim_str('process'):<15}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Re-aggregate results from per-entry files."""
    from miroeval.runner import aggregate

    result = aggregate(args.model)
    _print_result_summary(result)
    print(f"\nResults saved to outputs/{args.model}/results.json")


def _print_result_summary(output: dict[str, Any]) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print(f"  MiroEval — {output.get('model_name', '?')}")
    print("=" * 60)

    if "factual" in output:
        fe = output["factual"]
        print(f"\n  [Factual]  avg_right_ratio: {fe.get('avg_right_ratio', 0):.4f}")
        print(f"    R={fe.get('right', 0)} W={fe.get('wrong', 0)} "
              f"U={fe.get('unknown', 0)} C={fe.get('conflict', 0)}")

    if "quality" in output:
        pq = output["quality"]
        print(f"\n  [Quality]  avg_total_score: {pq.get('average_total_score', 0):.3f}")

    if "process" in output:
        pe = output["process"]
        print(f"\n  [Process]  overall_avg: {pe.get('overall_avg', 'N/A')}")
        if pe.get("intrinsic_avg") is not None:
            print(f"    intrinsic: {pe['intrinsic_avg']:.2f}  alignment: {pe.get('alignment_avg', 0):.2f}")

    print("=" * 60)


# ── Shared argument groups ────────────────────────────────────────────────


def _add_eval_params(parser: argparse.ArgumentParser) -> None:
    """Add common evaluation parameter arguments."""
    g = parser.add_argument_group("quality eval")
    g.add_argument("--quality-model", default=QUALITY_MODEL)
    g.add_argument("--quality-api-type", default=QUALITY_API_TYPE)
    g.add_argument("--quality-max-workers", type=int, default=QUALITY_MAX_WORKERS)

    g = parser.add_argument_group("process eval")
    g.add_argument("--process-model", default=PROCESS_MODEL)
    g.add_argument("--process-api-type", default=PROCESS_API_TYPE)
    g.add_argument("--process-max-workers", type=int, default=PROCESS_MAX_WORKERS)

    g = parser.add_argument_group("factual eval")
    g.add_argument("--factual-max-concurrent", type=int, default=FACTUAL_MAX_CONCURRENT)
    g.add_argument("--factual-max-chunks", type=int, default=FACTUAL_MAX_CHUNKS)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="miroeval",
        description="MiroEval — Benchmarking Deep Research Agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run incremental evaluation")
    p_run.add_argument("--input", required=True, help="Input JSON file or directory")
    p_run.add_argument("--model", required=True, help="Model name")
    p_run.add_argument("--eval", nargs="+", choices=["factual", "quality", "process"],
                       help="Dimensions to evaluate (default: all)")
    p_run.add_argument("--entries", help="Comma-separated entry IDs to force re-evaluate")
    _add_eval_params(p_run)

    # retry
    p_retry = sub.add_parser("retry", help="Retry failed entries")
    p_retry.add_argument("--model", required=True)
    p_retry.add_argument("--eval", nargs="+", choices=["factual", "quality", "process"])
    _add_eval_params(p_retry)

    # status
    p_status = sub.add_parser("status", help="Show evaluation status")
    p_status.add_argument("--model", help="Specific model (default: all)")

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Re-aggregate from per-entry results")
    p_agg.add_argument("--model", required=True)

    args = parser.parse_args()

    # Dispatch with underscore-converted attribute names.
    if args.command == "run":
        cmd_run(args)
    elif args.command == "retry":
        cmd_retry(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
