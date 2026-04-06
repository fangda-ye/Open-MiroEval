"""Data models for MiroEval.

All types are TypedDict — zero runtime overhead, full IDE support.
"""

from __future__ import annotations

from typing import Any, TypedDict


# ── Input data models ─────────────────────────────────────────────────────


class FileAttachment(TypedDict, total=False):
    """A file attached to a multimodal query."""

    filename: str  # e.g. "attachment_71_01.jpg"
    type: str  # "image", "pdf", "doc", "txt", etc.
    dir: str  # relative path from data/input_queries/
    size: str  # human-readable, e.g. "1.5 MB"


class EvalEntry(TypedDict, total=False):
    """A single evaluation entry (query + model output).

    This is the unified format consumed by all evaluation modules.
    """

    id: int
    chat_id: str
    rewritten_query: str
    response: str  # model-generated report
    process: str  # model research process trace
    files: list[FileAttachment]
    annotation: dict[str, Any]


# ── Per-entry result models ───────────────────────────────────────────────


class FactualEntryResult(TypedDict, total=False):
    """Factual evaluation result for a single entry."""

    total: int
    right: int
    wrong: int
    conflict: int
    unknown: int
    right_ratio: float
    statements: list[dict[str, Any]]  # individual statement verdicts


class QualityEntryResult(TypedDict, total=False):
    """Quality evaluation result for a single entry."""

    total_weighted_score: float
    dimension_scores: dict[str, float]
    dimensions_detail: dict[str, Any]  # criteria + per-criteria scores


class ProcessEntryResult(TypedDict, total=False):
    """Process evaluation result for a single entry."""

    intrinsic_scores: dict[str, Any]  # 5 dimensions
    alignment_scores: dict[str, Any]  # 3 dimensions
    structured_process: dict[str, Any]  # intermediate structuring result


# ── Aggregated result models ──────────────────────────────────────────────


class FactualSummary(TypedDict, total=False):
    total_statements: int
    right: int
    wrong: int
    conflict: int
    unknown: int
    avg_right_ratio: float


class QualitySummary(TypedDict, total=False):
    average_total_score: float
    dimension_averages: dict[str, float]
    total_queries: int


class ProcessSummary(TypedDict, total=False):
    intrinsic_avg: float | None
    alignment_avg: float | None
    overall_avg: float | None
    dimensions: dict[str, dict[str, float]]  # dim -> {avg, count}


# ── Manifest (incremental evaluation state) ───────────────────────────────


class EntryStatus(TypedDict):
    """Per-entry evaluation status across all dimensions."""

    factual: str  # "pending" | "running" | "completed" | "failed"
    quality: str
    process: str


class ManifestSummary(TypedDict):
    total_entries: int
    factual: dict[str, int]  # {"completed": N, "failed": N, ...}
    quality: dict[str, int]
    process: dict[str, int]


class Manifest(TypedDict, total=False):
    """Tracks incremental evaluation state for a model."""

    model_name: str
    input_file: str
    created_at: str
    updated_at: str
    config: dict[str, Any]
    entries: dict[str, EntryStatus]  # entry_id -> status
    summary: ManifestSummary
