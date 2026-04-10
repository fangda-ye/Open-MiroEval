"""Factual correctness evaluator — thin wrapper around MiroFlow.

MiroFlow provides the agentic fact-checking infrastructure (agents, search
tools, segmentation).  This module handles:
- Hydra config management (text vs multimodal)
- Temp file bridging (MiroFlow expects file-based input)
- Verdict aggregation (Right/Wrong/Conflict/Unknown counting)

Requires ``miroflow`` to be installed (``pip install miroeval[factual]``).
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

from miroeval.core.config import REPO_ROOT

logger = logging.getLogger(__name__)

# Path to the factual module's config directory (Hydra configs, prompts).
FACTUAL_CONFIG_DIR = Path(__file__).resolve().parent / "config"
# MiroFlow framework lives in third_party/.
MIROFLOW_DIR = REPO_ROOT / "third_party" / "miroflow"

_TEXT_CONFIG = "benchmark_factual-eval_text"
_MULTIMODAL_CONFIG = "benchmark_factual-eval_multimodal"


class FactualEvaluator:
    """Factual evaluation via MiroFlow agent-based fact-checking."""

    def __init__(self, *, max_concurrent: int = 10, max_concurrent_chunks: int = 10):
        self.max_concurrent = max_concurrent
        self.max_concurrent_chunks = max_concurrent_chunks
        self._cfg_text = None
        self._cfg_multimodal = None
        self._hydra_ready = False

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate_entry(
        self,
        entry_id: int,
        entry: dict[str, Any],
        *,
        mode: str = "text",
    ) -> dict[str, Any] | None:
        """Evaluate a single entry's factual correctness.

        Returns dict with: total, right, wrong, conflict, unknown, right_ratio, statements.
        """
        result = self._evaluate_entries(
            [entry], model_name=f"entry_{entry_id}", mode=mode
        )
        per_entry = result.get("per_entry", {})
        if per_entry:
            return next(iter(per_entry.values()))
        return None

    def evaluate_batch(
        self,
        entries: list[dict[str, Any]],
        model_name: str,
        *,
        mode: str = "text",
    ) -> dict[str, Any]:
        """Evaluate a batch of entries.

        Returns {per_entry: {...}, summary: {...}}.
        """
        result = self._evaluate_entries(entries, model_name=model_name, mode=mode)
        # Build summary from per_entry.
        per_entry = result.get("per_entry", {})
        right = wrong = conflict = unknown = total = 0
        ratios: list[float] = []
        for v in per_entry.values():
            if isinstance(v, dict) and "right" in v:
                right += v["right"]
                wrong += v["wrong"]
                conflict += v.get("conflict", 0)
                unknown += v.get("unknown", 0)
                total += v.get("total", 0)
                if "right_ratio" in v:
                    ratios.append(v["right_ratio"])

        summary = {
            "total_statements": total,
            "right": right,
            "wrong": wrong,
            "conflict": conflict,
            "unknown": unknown,
            "avg_right_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        }
        return {"per_entry": per_entry, "summary": summary}

    # ── Hydra init ────────────────────────────────────────────────────────

    def _ensure_hydra(self) -> None:
        """One-time Hydra initialisation."""
        if self._hydra_ready:
            return

        # Ensure MiroFlow and config packages are importable.
        mf_dir = str(MIROFLOW_DIR)
        cfg_parent = str(FACTUAL_CONFIG_DIR.parent)  # miroeval/factual/
        for p in (mf_dir, cfg_parent):
            if p not in sys.path:
                sys.path.insert(0, p)

        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        prev_cwd = os.getcwd()
        try:
            os.chdir(cfg_parent)
            from config import load_config

            self._cfg_text = load_config(_TEXT_CONFIG)

            import hydra
            raw = hydra.compose(config_name=_MULTIMODAL_CONFIG)
            resolved = OmegaConf.to_container(raw, resolve=True)
            self._cfg_multimodal = OmegaConf.create(resolved)
            self._cfg_multimodal.output_dir = self._cfg_text.output_dir
        finally:
            os.chdir(prev_cwd)

        self._hydra_ready = True
        logger.info("factual: Hydra initialised")

    # ── Core execution ────────────────────────────────────────────────────

    def _evaluate_entries(
        self,
        entries: list[dict[str, Any]],
        model_name: str,
        mode: str = "text",
    ) -> dict[str, Any]:
        if not entries:
            return {"per_entry": {}}

        self._ensure_hydra()

        # Tell MiroFlow worker processes where to chdir so relative
        # config paths (config/llm/*, config/tool/*) resolve correctly.
        os.environ["MIROEVAL_FACTUAL_CWD"] = str(FACTUAL_CONFIG_DIR.parent)

        from omegaconf import OmegaConf

        cfg_template = self._cfg_multimodal if mode == "multimodal" else self._cfg_text

        # Write temp data file.
        tmp_dir = REPO_ROOT / "outputs" / "_factual_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_filename = f"eval_{model_name}_{uuid.uuid4().hex[:8]}.json"
        tmp_path = tmp_dir / tmp_filename

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False)

            prev_cwd = os.getcwd()
            os.chdir(str(FACTUAL_CONFIG_DIR.parent))
            try:
                return self._run(cfg_template, tmp_path, model_name)
            finally:
                os.chdir(prev_cwd)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run(
        self,
        cfg_template: Any,
        source_path: Path,
        model_name: str,
    ) -> dict[str, Any]:
        from omegaconf import OmegaConf
        from miroflow.benchmark.factual_eval_task_runner import run_factual_eval_tasks
        from miroflow.benchmark.run_factual_eval import _load_json_array_tasks
        from miroflow.logging.task_tracer import set_tracer

        cfg_dict = OmegaConf.to_container(cfg_template, resolve=True)
        cfg = OmegaConf.create(cfg_dict)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(REPO_ROOT / "outputs" / "_factual_logs" / f"run_{ts}")
        cfg.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        set_tracer(output_dir)

        tasks = _load_json_array_tasks(source_path)
        if not tasks:
            return {"per_entry": {}}

        execution_cfg = cfg.benchmark.execution
        results = run_factual_eval_tasks(
            cfg=cfg,
            tasks=tasks,
            max_concurrent=self.max_concurrent,
            max_concurrent_chunks=self.max_concurrent_chunks,
            max_retries=execution_cfg.get("max_retries", 5),
            chunk_timeout=execution_cfg.get("chunk_timeout", None),
            chunk_tracing=execution_cfg.get("chunk_tracing", True),
        )

        return self._parse_results(results)

    # ── Result parsing ────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(results: list) -> dict[str, Any]:
        """Parse MiroFlow TaskResult objects into per-entry verdict counts."""
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

            counts = {"right": 0, "wrong": 0, "conflict": 0, "unknown": 0}
            statements = []
            for stmt in core_state:
                verdict = stmt.get("verification", "").strip().lower()
                if verdict == "right":
                    counts["right"] += 1
                elif verdict == "wrong":
                    counts["wrong"] += 1
                elif verdict == "conflict":
                    counts["conflict"] += 1
                else:
                    counts["unknown"] += 1
                statements.append(stmt)

            n = sum(counts.values())
            counts["total"] = n
            counts["right_ratio"] = counts["right"] / n if n > 0 else 0.0
            counts["statements"] = statements
            per_entry[entry_id] = counts

        return {"per_entry": per_entry}
