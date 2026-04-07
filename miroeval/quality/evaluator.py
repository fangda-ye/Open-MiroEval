"""Adaptive point-wise quality evaluator.

5-stage pipeline:
  0. Key facts extraction (for multimodal queries with attachments)
  1. Query-specific dimension generation
  2. Hierarchical weight assignment
  3. Per-dimension criteria generation
  4. Scoring + hierarchical aggregation
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

from miroeval.core.llm import LLMClient
from miroeval.core.cache import CacheManager
from miroeval.core.utils import extract_json
from miroeval.quality.prompts import (
    CRITERIA_GENERATION_PROMPT,
    CRITERIA_GENERATION_WITH_KEY_FACTS_PROMPT,
    DIMENSION_GENERATION_PROMPT,
    DIMENSION_GENERATION_WITH_ATTACHMENT_PROMPT,
    KEY_FACTS_EXTRACTION_PROMPT,
    SCORING_PROMPT,
    SCORING_WITH_KEY_FACTS_PROMPT,
    WEIGHT_GENERATION_PROMPT,
)

logger = logging.getLogger(__name__)

# Image extensions that need LLM-based reading.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Fixed evaluation dimensions (always present).
FIXED_DIMS: dict[str, str] = {
    "coverage": "Breadth, depth, and relevance of coverage",
    "insight": "Depth, originality, logic, and value of analysis",
    "instruction_following": "Accuracy in meeting all requirements",
    "clarity": "Readability, fluency, structure, and ease of understanding",
}


class QualityEvaluator:
    """Adaptive point-wise quality evaluator for deep-research reports."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.1",
        api_type: str = "openai",
        cache_dir: str = "outputs/cache",
        data_dir: str | None = None,
    ):
        self.llm = LLMClient(model=model, api_type=api_type)
        self.cache = CacheManager(cache_dir)
        self.data_dir = data_dir  # for resolving attachment paths

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate_entry(
        self,
        entry_id: int,
        query: str,
        report: str,
        *,
        attachment_parts: list[str] | None = None,
        criteria_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single entry (query + report).

        If *criteria_data* is provided (from a previous run), only Stage 4
        (scoring) is executed.  Otherwise the full 5-stage pipeline runs.

        Returns a dict with keys: total_weighted_score, dimension_scores,
        dimensions_detail (raw scores per criterion).
        """
        # Build a lightweight query dict for internal methods.
        query_obj: dict[str, Any] = {"id": entry_id, "prompt": query}
        if attachment_parts:
            query_obj["attachment_parts"] = attachment_parts
            query_obj["attachment"] = "\n\n---\n\n".join(attachment_parts)

        if criteria_data is None:
            criteria_data = self._create_criteria(entry_id, query_obj)

        scores = self._score_report(
            entry_id,
            query_obj,
            report,
            criteria_data["all_criteria"],
            key_facts=criteria_data.get("key_facts"),
        )

        final = self._aggregate_scores(
            scores,
            criteria_data["all_criteria"],
            criteria_data["dimension_weights"],
        )

        return {
            "total_weighted_score": final.get("total_weighted_score", 0.0),
            "dimension_scores": {
                k: v for k, v in final.items()
                if k.endswith("_score") and k != "total_weighted_score"
            },
            "dimensions_detail": scores,
            "criteria_data": criteria_data,
        }

    def evaluate_batch(
        self,
        entries: list[dict[str, Any]],
        *,
        max_workers: int = 10,
        on_entry_done: Any = None,
    ) -> dict[str, Any]:
        """Evaluate a batch of entries in parallel.

        Each entry dict must have: id, rewritten_query (or query), response.
        Optional: files (list of FileAttachment dicts).

        Args:
            on_entry_done: Optional callback ``(entry_id: str, result: dict | None, error: str | None) -> None``
                called after each entry completes (success or failure).

        Returns {per_entry: {...}, summary: {...}}.
        """
        from tqdm import tqdm

        results: dict[str, Any] = {}
        failed = 0

        def _run_one(entry: dict[str, Any]) -> tuple[int, dict[str, Any] | None, str | None]:
            eid = entry["id"]
            query = entry.get("rewritten_query") or entry.get("query", "")
            report = entry.get("response", "")
            if not report:
                return eid, None, "no response"

            att_parts = self._resolve_files(entry.get("files", []))

            try:
                result = self.evaluate_entry(
                    eid, query, report, attachment_parts=att_parts or None
                )
                return eid, result, None
            except Exception as e:
                logger.error("Quality eval failed for entry %s: %s", eid, e)
                return eid, None, str(e)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, e): e["id"] for e in entries}
            with tqdm(total=len(futures), desc="Quality", unit="entry") as pbar:
                for fut in as_completed(futures):
                    eid, result, error = fut.result()
                    eid_str = str(eid)
                    if result is not None:
                        results[eid_str] = result
                    else:
                        failed += 1
                        results[eid_str] = {"error": error or "unknown"}
                    if on_entry_done:
                        on_entry_done(eid_str, result, error)
                    pbar.set_postfix(ok=len(results) - failed, fail=failed)
                    pbar.update(1)

        logger.info("Quality eval: %d completed, %d failed", len(results) - failed, failed)

        # Summary (exclude failed entries).
        ok_results = {k: v for k, v in results.items() if "error" not in v}
        scores = [r["total_weighted_score"] for r in ok_results.values() if r.get("total_weighted_score", 0) > 0]
        dim_scores: dict[str, list[float]] = {}
        for r in ok_results.values():
            for k, v in r.get("dimension_scores", {}).items():
                if v is not None:
                    dim_scores.setdefault(k, []).append(v)

        summary = {
            "average_total_score": mean(scores) if scores else 0.0,
            "total_queries": len(scores),
            "dimension_averages": {
                k: mean(v) for k, v in dim_scores.items() if v
            },
        }

        return {"per_entry": results, "summary": summary}

    # ── Stage 0: Key facts extraction ─────────────────────────────────────

    def _extract_key_facts(
        self, entry_id: int, query_obj: dict[str, Any]
    ) -> list[dict[str, str]]:
        cache_key = f"key_facts_{entry_id}"
        cached = self.cache.get_cache("key_facts").get(cache_key)
        if cached is not None:
            return cached

        parts = query_obj.get("attachment_parts", [])
        if not parts:
            att = query_obj.get("attachment", "")
            parts = [att] if att and att.strip() else []
        if not parts:
            return []

        prompt_text = query_obj["prompt"]
        all_facts: list[dict[str, str]] = []

        for i, part in enumerate(parts):
            if not part or not part.strip():
                continue
            formatted = KEY_FACTS_EXTRACTION_PROMPT.format(
                task_prompt=prompt_text, attachment_content=part
            )
            resp = self.llm.generate([{"role": "user", "content": formatted}])
            parsed = extract_json(resp)
            if isinstance(parsed, list):
                all_facts.extend(parsed)

        self.cache.get_cache("key_facts").set(cache_key, all_facts)
        return all_facts

    # ── Stage 1: Dimension generation ─────────────────────────────────────

    def _generate_dimensions(
        self,
        entry_id: int,
        query_obj: dict[str, Any],
        key_facts: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        cache_key = f"dimensions_{entry_id}"
        cached = self.cache.get_cache("dimensions").get(cache_key)
        if cached is not None:
            return cached

        prompt_text = query_obj["prompt"]
        if key_facts:
            formatted = DIMENSION_GENERATION_WITH_ATTACHMENT_PROMPT.format(
                task_prompt=prompt_text,
                key_facts_json=json.dumps(key_facts, ensure_ascii=False, indent=2),
            )
        else:
            formatted = DIMENSION_GENERATION_PROMPT.format(task_prompt=prompt_text)

        resp = self.llm.generate([{"role": "user", "content": formatted}])
        dims = extract_json(resp)
        if not isinstance(dims, list):
            dims = []

        self.cache.get_cache("dimensions").set(cache_key, dims)
        return dims

    # ── Stage 2: Weight generation ────────────────────────────────────────

    def _generate_weights(
        self,
        entry_id: int,
        query_obj: dict[str, Any],
        additional_dims: list[dict[str, Any]],
    ) -> dict[str, float]:
        cache_key = f"weights_{entry_id}_{len(additional_dims)}"
        cached = self.cache.get_cache("weights").get(cache_key)
        if cached is not None:
            return cached

        formatted = WEIGHT_GENERATION_PROMPT.format(
            task_prompt=query_obj["prompt"],
            additional_dimensions_json=json.dumps(additional_dims, ensure_ascii=False, indent=2),
        )
        resp = self.llm.generate([{"role": "user", "content": formatted}])
        weights_raw = self._extract_json_from_analysis(resp)

        if isinstance(weights_raw, dict):
            total = sum(weights_raw.values())
            if total > 0:
                weights_raw = {k: v / total for k, v in weights_raw.items()}
            # Normalise keys.
            weights = {
                k.lower().replace(" ", "_").replace("-", "_"): v
                for k, v in weights_raw.items()
            }
        else:
            weights = self._default_weights(additional_dims)

        self.cache.get_cache("weights").set(cache_key, weights)
        return weights

    # ── Stage 3: Criteria generation ──────────────────────────────────────

    def _generate_criteria(
        self,
        entry_id: int,
        query_obj: dict[str, Any],
        dim_name: str,
        all_dims: dict[str, str],
        key_facts: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        cache_key = f"criteria_{entry_id}_{dim_name}"
        cached = self.cache.get_cache("criteria").get(cache_key)
        if cached is not None:
            return cached

        meta_dims_str = "\n".join(
            f"- **{d}**: {all_dims[d]}" for d in all_dims
        )

        if key_facts:
            formatted = CRITERIA_GENERATION_WITH_KEY_FACTS_PROMPT.format(
                task_prompt=query_obj["prompt"],
                num_dimensions=len(all_dims),
                meta_dimensions=meta_dims_str,
                dimension_name=dim_name,
                key_facts_json=json.dumps(key_facts, ensure_ascii=False, indent=2),
            )
        else:
            formatted = CRITERIA_GENERATION_PROMPT.format(
                task_prompt=query_obj["prompt"],
                num_dimensions=len(all_dims),
                meta_dimensions=meta_dims_str,
                dimension_name=dim_name,
            )

        criteria = None
        for attempt in range(2):
            resp = self.llm.generate([{"role": "user", "content": formatted}])
            parsed = self._extract_json_from_analysis(resp)
            if isinstance(parsed, list) and parsed:
                total_w = sum(item.get("weight", 0) for item in parsed)
                if total_w > 0:
                    for item in parsed:
                        item["weight"] = item.get("weight", 0) / total_w
                criteria = parsed
                break

        if criteria is None:
            dim_def = all_dims.get(dim_name, f"Quality of {dim_name}")
            criteria = self._default_criteria(dim_name, dim_def)

        self.cache.get_cache("criteria").set(cache_key, criteria)
        return criteria

    # ── Full criteria pipeline (Stages 0-3) ───────────────────────────────

    def _create_criteria(
        self, entry_id: int, query_obj: dict[str, Any]
    ) -> dict[str, Any]:
        has_attachment = bool(
            query_obj.get("attachment", "").strip()
            if isinstance(query_obj.get("attachment"), str)
            else query_obj.get("attachment_parts")
        )

        key_facts = self._extract_key_facts(entry_id, query_obj) if has_attachment else None
        additional_dims = self._generate_dimensions(entry_id, query_obj, key_facts)
        weights = self._generate_weights(entry_id, query_obj, additional_dims)

        all_dims = dict(FIXED_DIMS)
        dynamic_names = set()
        for item in additional_dims:
            key = item["meta_dimension_name"].lower().replace(" ", "_").replace("-", "_")
            all_dims[key] = item["definition"]
            dynamic_names.add(key)

        all_criteria: dict[str, list[dict[str, Any]]] = {}
        for dim_name in all_dims:
            kf = key_facts if (has_attachment and key_facts and dim_name in dynamic_names) else None
            all_criteria[dim_name] = self._generate_criteria(
                entry_id, query_obj, dim_name, all_dims, key_facts=kf
            )

        return {
            "all_criteria": all_criteria,
            "all_dims_with_definition": all_dims,
            "dimension_weights": weights,
            "additional_dimensions": additional_dims,
            "key_facts": key_facts,
            "has_attachment": has_attachment,
        }

    # ── Stage 4: Scoring ──────────────────────────────────────────────────

    def _score_single_dim(
        self,
        entry_id: int,
        task_prompt: str,
        report: str,
        dim_name: str,
        criteria_list: list[dict[str, Any]],
        key_facts: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        criteria_json = json.dumps(
            [{"criterion": c["criterion"], "explanation": c["explanation"]} for c in criteria_list],
            ensure_ascii=False,
            indent=2,
        )

        if key_facts:
            formatted = SCORING_WITH_KEY_FACTS_PROMPT.format(
                task_prompt=task_prompt,
                report=report,
                criteria_of_one_dimension_json=criteria_json,
                key_facts_json=json.dumps(key_facts, ensure_ascii=False, indent=2),
            )
        else:
            formatted = SCORING_PROMPT.format(
                task_prompt=task_prompt,
                report=report,
                criteria_of_one_dimension_json=criteria_json,
            )

        max_retries = 3
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.llm.generate([{"role": "user", "content": formatted}])
                parsed = self._extract_json_from_analysis(resp)
                if not isinstance(parsed, list):
                    raise ValueError("Expected JSON list from scoring response")

                resp_map = {item["criterion"]: item for item in parsed}
                dim_scores = []
                for c in criteria_list:
                    item = resp_map[c["criterion"]]
                    dim_scores.append({
                        "criterion": c["criterion"],
                        "analysis": item["analysis"],
                        "report_score_0_to_10": float(item["report_score_0_to_10"]),
                    })
                return dim_name, dim_scores

            except Exception as e:
                last_err = e
                logger.warning(
                    "Scoring dim '%s' attempt %d/%d failed: %s",
                    dim_name, attempt + 1, max_retries, e,
                )

        raise RuntimeError(
            f"Dimension '{dim_name}' scoring failed after {max_retries} attempts: {last_err}"
        )

    def _score_report(
        self,
        entry_id: int,
        query_obj: dict[str, Any],
        report: str,
        all_criteria: dict[str, list[dict[str, Any]]],
        key_facts: list[dict[str, str]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        cache_key = f"scores_{entry_id}_{hash(report)}"
        cached = self.cache.get_cache("scores").get(cache_key)
        if cached is not None:
            return cached

        task_prompt = query_obj["prompt"]
        final_scores: dict[str, list[dict[str, Any]]] = {}
        has_errors = False

        workers = min(4, len(all_criteria))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    self._score_single_dim,
                    entry_id, task_prompt, report, dn, cl, key_facts,
                ): dn
                for dn, cl in all_criteria.items()
            }
            for fut in as_completed(futs):
                dn = futs[fut]
                try:
                    name, scores = fut.result()
                    final_scores[name] = scores
                except Exception as e:
                    logger.error("Scoring failed for dim '%s': %s", dn, e)
                    final_scores[dn] = []
                    has_errors = True

        if not has_errors:
            self.cache.get_cache("scores").set(cache_key, final_scores)

        return final_scores

    # ── Aggregation ───────────────────────────────────────────────────────

    @staticmethod
    def _aggregate_scores(
        scores: dict[str, list[dict[str, Any]]],
        all_criteria: dict[str, list[dict[str, Any]]],
        dimension_weights: dict[str, float],
    ) -> dict[str, float]:
        final: dict[str, Any] = {}
        dim_score_map: dict[str, float] = {}

        for dim_name, criteria_list in all_criteria.items():
            dim_scores = scores.get(dim_name, [])
            if not isinstance(dim_scores, list) or not dim_scores:
                final[f"{dim_name}_score"] = None
                continue

            weighted_sum = 0.0
            total_w = 0.0
            for i, c in enumerate(criteria_list):
                if i < len(dim_scores):
                    s = dim_scores[i]
                    if (
                        isinstance(s, dict)
                        and c["criterion"] == s["criterion"]
                        and "report_score_0_to_10" in s
                    ):
                        weighted_sum += float(s["report_score_0_to_10"]) * float(c["weight"])
                        total_w += float(c["weight"])

            if total_w > 0:
                score = weighted_sum / total_w
                final[f"{dim_name}_score"] = score
                dim_score_map[dim_name] = score
            else:
                final[f"{dim_name}_score"] = None

        # Weighted total (redistribute weight from failed dims).
        ok_weight = sum(dimension_weights.get(d, 0) for d in dim_score_map)
        total = 0.0
        if ok_weight > 0:
            for d, s in dim_score_map.items():
                total += s * (dimension_weights.get(d, 0) / ok_weight)
        final["total_weighted_score"] = total

        return final

    # ── Attachment resolution ─────────────────────────────────────────────

    def _resolve_files(self, files: list[dict[str, Any]]) -> list[str]:
        """Resolve file attachments to text content."""
        if not files or not self.data_dir:
            return []

        parts: list[str] = []
        for f_info in files:
            f_path = f_info.get("dir", "")
            if not f_path:
                continue
            full_path = os.path.join(self.data_dir, "input_queries", f_path)
            if os.path.isfile(full_path):
                parts.append(self._read_file(full_path))
            else:
                logger.warning("Attachment not found: %s", full_path)
        return parts

    def _read_file(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext in (".txt", ".csv", ".md", ".json", ".jsonl", ".tsv"):
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
            elif ext == ".pdf":
                return self._extract_pdf(filepath)
            elif ext in _IMAGE_EXTENSIONS:
                return self._describe_image(filepath)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.warning("Failed to read %s: %s", filepath, e)
            return f"[Error reading attachment: {filepath}]"

    def _describe_image(self, filepath: str) -> str:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(filepath)[1].lower()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".bmp": "image/bmp", ".webp": "image/webp",
            ".tiff": "image/tiff", ".tif": "image/tiff",
        }.get(ext, "image/png")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Please carefully examine this image and extract all key information. "
                    "Include: 1) Detailed description, 2) All text/numbers/labels, "
                    "3) Any data/charts/tables, 4) Key observations."
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
        resp = self.llm.generate(messages, max_tokens=4096)
        name = os.path.basename(filepath)
        return f"[Image: {name}]\n{resp}" if resp != "$ERROR$" else f"[Error describing image: {name}]"

    @staticmethod
    def _extract_pdf(filepath: str) -> str:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            return f"[PDF: {os.path.basename(filepath)} — PyPDF2 required]"
        except Exception as e:
            return f"[Error extracting PDF: {e}]"

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_json_from_analysis(self, text: str) -> Any:
        """Extract JSON from a response that may contain <analysis> and <json_output> tags."""
        if not isinstance(text, str):
            return None
        # Try <json_output> tags first.
        m = re.search(r"<json_output>\s*(.*?)\s*</json_output>", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return extract_json(text)

    @staticmethod
    def _default_weights(additional_dims: list[dict[str, Any]]) -> dict[str, float]:
        n = 4 + len(additional_dims)
        w = 1.0 / n
        weights = {k: w for k in FIXED_DIMS}
        for d in additional_dims:
            key = d.get("meta_dimension_name", "").lower().replace(" ", "_").replace("-", "_")
            weights[key] = w
        return weights

    @staticmethod
    def _default_criteria(dim_name: str, definition: str) -> list[dict[str, Any]]:
        return [
            {"criterion": f"Core quality of {dim_name}", "explanation": f"Primary aspects of: {definition}", "weight": 0.5},
            {"criterion": f"Depth and specificity of {dim_name}", "explanation": f"Detailed analysis for: {definition}", "weight": 0.3},
            {"criterion": f"Relevance of {dim_name}", "explanation": f"Task alignment for: {definition}", "weight": 0.2},
        ]
