"""Process quality evaluator.

Two-phase pipeline:
  Phase 1: Preprocess + structure (raw process trace → unified JSON schema)
  Phase 2: Evaluate (intrinsic 5-dim + alignment 3-dim scoring)
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

from miroeval.core.llm import LLMClient
from miroeval.core.cache import FileCache
from miroeval.core.utils import extract_json
from miroeval.process.prompts import (
    ALIGNMENT_EVAL_PROMPT,
    COMPRESS_PROMPT,
    INTRINSIC_EVAL_PROMPT,
    STRUCTURING_PROMPT,
)

logger = logging.getLogger(__name__)

INTRINSIC_DIMS = [
    "search_breadth",
    "analytical_depth",
    "progressive_refinement",
    "critical_thinking",
    "efficiency",
]
ALIGNMENT_DIMS = ["findings_to_report", "report_to_process", "contradiction"]

# Max chars per chunk sent to LLM for compression.
_CHUNK_INPUT_CHARS = 40_000


class ProcessEvaluator:
    """Process quality evaluator (intrinsic + alignment)."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        api_type: str = "openai",
        cache_dir: str = "outputs/cache",
        max_chars: int = 50_000,
        report_max_chars: int = 30_000,
    ):
        self.llm = LLMClient(model=model, api_type=api_type)
        self._struct_cache = FileCache(cache_dir, "structuring")
        self._intrinsic_cache = FileCache(cache_dir, "intrinsic")
        self._alignment_cache = FileCache(cache_dir, "alignment")
        self.max_chars = max_chars
        self.report_max_chars = report_max_chars

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate_entry(
        self,
        entry_id: int,
        model_name: str,
        query: str,
        process: str,
        report: str,
    ) -> dict[str, Any] | None:
        """Evaluate a single entry's process and alignment.

        Returns dict with: intrinsic_scores, alignment_scores, structured_process.
        """
        # Phase 1: preprocess + structure.
        preprocessed = self._preprocess(process, query)
        structured = self._structure(entry_id, model_name, preprocessed, query)
        if structured is None:
            return None

        # Phase 2: evaluate.
        intrinsic = self._eval_intrinsic(entry_id, model_name, structured, query)
        alignment = self._eval_alignment(entry_id, model_name, structured, report, query)

        if intrinsic is None and alignment is None:
            return None

        return {
            "intrinsic_scores": intrinsic,
            "alignment_scores": alignment,
            "structured_process": structured,
        }

    def evaluate_batch(
        self,
        entries: list[dict[str, Any]],
        model_name: str,
        *,
        max_workers: int = 10,
        on_entry_done: Any = None,
    ) -> dict[str, Any]:
        """Evaluate a batch of entries in parallel.

        Each entry dict must have: id, rewritten_query (or query), response, process.

        Args:
            on_entry_done: Optional callback ``(key: str, result: dict | None, error: str | None) -> None``
                called after each entry completes.

        Returns {per_entry: {...}, summary: {...}}.
        """
        from tqdm import tqdm

        results: dict[str, Any] = {}
        failed = 0

        def _run(e: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
            eid = e["id"]
            key = f"{model_name}_{eid}"
            query = e.get("rewritten_query") or e.get("query", "")
            process = e.get("process", "")
            report = e.get("response", "")
            if not process:
                return key, None, "no process trace"
            try:
                return key, self.evaluate_entry(eid, model_name, query, process, report), None
            except Exception as exc:
                logger.error("Process eval failed for %s: %s", key, exc)
                return key, None, str(exc)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_run, e): e["id"] for e in entries}
            with tqdm(total=len(futs), desc="Process", unit="entry") as pbar:
                for fut in as_completed(futs):
                    key, result, error = fut.result()
                    if result is not None:
                        results[key] = result
                    else:
                        failed += 1
                        results[key] = {"error": error or "unknown"}
                    if on_entry_done:
                        on_entry_done(key, result, error)
                    pbar.set_postfix(ok=len(results) - failed, fail=failed)
                    pbar.update(1)

        logger.info("Process eval: %d completed, %d failed", len(results) - failed, failed)
        summary = self._aggregate(results, model_name)
        return {"per_entry": results, "summary": summary}

    # ── Phase 1: Preprocessing ────────────────────────────────────────────

    def _preprocess(self, process_text: str, query: str) -> str:
        """Compress long process traces via LLM chunking."""
        if not process_text or not process_text.strip():
            return ""
        if len(process_text) <= _CHUNK_INPUT_CHARS:
            return self._compress_chunk(process_text, query, 1, 1)

        chunks = self._split_chunks(process_text)
        if len(chunks) == 1:
            return self._compress_chunk(chunks[0], query, 1, 1)

        compressed = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = {
                pool.submit(self._compress_chunk, c, query, i + 1, len(chunks)): i
                for i, c in enumerate(chunks)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    compressed[idx] = fut.result()
                except Exception:
                    compressed[idx] = chunks[idx][:8000]

        result = "\n\n---\n\n".join(c for c in compressed if c)
        if len(result) > self.max_chars:
            result = result[: self.max_chars] + "\n\n[... truncated ...]"
        return result

    def _compress_chunk(
        self, text: str, query: str, idx: int, total: int
    ) -> str:
        prompt = COMPRESS_PROMPT.format(
            query=query, chunk_idx=idx, total_chunks=total, chunk_text=text
        )
        resp = self.llm.generate([{"role": "user", "content": prompt}])
        if resp == "$ERROR$" or not resp.strip():
            return text[:8000]
        return resp

    def _split_chunks(self, text: str) -> list[str]:
        """Split process text at structural boundaries."""
        stripped = text.strip()
        if stripped.startswith("["):
            try:
                steps = json.loads(stripped)
                if isinstance(steps, list) and steps:
                    return self._chunk_json(steps)
            except (json.JSONDecodeError, TypeError):
                pass

        markers = list(re.finditer(
            r"\[(reasoning|web_search|scrape|run_python_code|Step \d+)\]", text
        ))
        if len(markers) >= 2:
            return self._chunk_at_markers(text, markers)

        return self._chunk_paragraphs(text)

    def _chunk_json(self, steps: list) -> list[str]:
        chunks, current, length = [], [], 0
        for s in steps:
            s_str = json.dumps(s, ensure_ascii=False)
            if length + len(s_str) > _CHUNK_INPUT_CHARS and current:
                chunks.append(json.dumps(current, ensure_ascii=False, indent=1))
                current, length = [], 0
            current.append(s)
            length += len(s_str)
        if current:
            chunks.append(json.dumps(current, ensure_ascii=False, indent=1))
        return chunks

    def _chunk_at_markers(self, text: str, matches: list) -> list[str]:
        bounds = [m.start() for m in matches] + [len(text)]
        chunks, start, length = [], 0, 0
        for i in range(len(bounds) - 1):
            seg = bounds[i + 1] - bounds[i]
            if length + seg > _CHUNK_INPUT_CHARS and length > 0:
                chunks.append(text[start : bounds[i]])
                start, length = bounds[i], 0
            length += seg
        if start < len(text):
            chunks.append(text[start:])
        return chunks or [text]

    def _chunk_paragraphs(self, text: str) -> list[str]:
        paras = re.split(r"\n\s*\n", text)
        chunks, parts, length = [], [], 0
        for p in paras:
            if length + len(p) > _CHUNK_INPUT_CHARS and parts:
                chunks.append("\n\n".join(parts))
                parts, length = [], 0
            parts.append(p)
            length += len(p)
        if parts:
            chunks.append("\n\n".join(parts))
        return chunks or [text]

    # ── Phase 1: Structuring ──────────────────────────────────────────────

    def _structure(
        self, entry_id: int, model_name: str, preprocessed: str, query: str
    ) -> dict | None:
        cache_key = f"{model_name}_{entry_id}"
        cached = self._struct_cache.get(cache_key)
        if cached is not None:
            return cached

        if not preprocessed.strip():
            return None

        prompt = STRUCTURING_PROMPT.format(query=query, process_text=preprocessed)
        result = self.llm.generate_json([{"role": "user", "content": prompt}])

        if not isinstance(result, dict):
            return None
        if "steps" not in result or "global_findings" not in result:
            return None

        self._struct_cache.set(cache_key, result)
        return result

    # ── Phase 2: Intrinsic evaluation ─────────────────────────────────────

    def _eval_intrinsic(
        self, entry_id: int, model_name: str, structured: dict, query: str
    ) -> dict | None:
        cache_key = f"{model_name}_{entry_id}"
        cached = self._intrinsic_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = INTRINSIC_EVAL_PROMPT.format(
            query=query,
            structured_process=json.dumps(structured, ensure_ascii=False, indent=2),
        )
        result = self.llm.generate_json([{"role": "user", "content": prompt}])
        if not isinstance(result, dict):
            return None

        for dim in INTRINSIC_DIMS:
            if dim not in result or not isinstance(result[dim], dict) or "score" not in result[dim]:
                logger.error("Missing/invalid intrinsic dim '%s' for %s", dim, cache_key)
                return None

        self._intrinsic_cache.set(cache_key, result)
        return result

    # ── Phase 2: Alignment evaluation ─────────────────────────────────────

    def _eval_alignment(
        self,
        entry_id: int,
        model_name: str,
        structured: dict,
        report: str,
        query: str,
    ) -> dict | None:
        cache_key = f"{model_name}_{entry_id}"
        cached = self._alignment_cache.get(cache_key)
        if cached is not None:
            return cached

        findings = structured.get("global_findings", [])
        findings_text = json.dumps(findings, ensure_ascii=False, indent=2)

        trunc_report = report[: self.report_max_chars]
        if len(report) > self.report_max_chars:
            trunc_report += "\n\n[... report truncated ...]"

        prompt = ALIGNMENT_EVAL_PROMPT.format(
            query=query, global_findings=findings_text, report=trunc_report
        )
        result = self.llm.generate_json([{"role": "user", "content": prompt}])
        if not isinstance(result, dict):
            return None

        for dim in ALIGNMENT_DIMS:
            if dim not in result or not isinstance(result[dim], dict) or "score" not in result[dim]:
                logger.error("Missing/invalid alignment dim '%s' for %s", dim, cache_key)
                return None

        self._alignment_cache.set(cache_key, result)
        return result

    # ── Aggregation ───────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(results: dict[str, Any], model_name: str) -> dict[str, Any]:
        dim_scores: dict[str, list[float]] = {}

        for result in results.values():
            if "error" in result:
                continue  # skip failed entries
            for scores_key, dims in [
                ("intrinsic_scores", INTRINSIC_DIMS),
                ("alignment_scores", ALIGNMENT_DIMS),
            ]:
                scores = result.get(scores_key)
                if not scores:
                    continue
                for dim in dims:
                    if dim in scores and isinstance(scores[dim], dict):
                        s = scores[dim].get("score")
                        if s is not None:
                            dim_scores.setdefault(dim, []).append(float(s))

        dimensions = {
            d: {"avg": mean(v), "count": len(v)}
            for d, v in dim_scores.items()
            if v
        }

        intrinsic_avgs = [
            dimensions[d]["avg"] for d in INTRINSIC_DIMS if d in dimensions
        ]
        alignment_avgs = [
            dimensions[d]["avg"] for d in ALIGNMENT_DIMS if d in dimensions
        ]

        intrinsic_avg = mean(intrinsic_avgs) if intrinsic_avgs else None
        alignment_avg = mean(alignment_avgs) if alignment_avgs else None

        overall = None
        if intrinsic_avg is not None and alignment_avg is not None:
            overall = mean([intrinsic_avg, alignment_avg])
        elif intrinsic_avg is not None:
            overall = intrinsic_avg
        elif alignment_avg is not None:
            overall = alignment_avg

        return {
            "intrinsic_avg": intrinsic_avg,
            "alignment_avg": alignment_avg,
            "overall_avg": overall,
            "dimensions": dimensions,
        }
