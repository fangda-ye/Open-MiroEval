"""Eval-level configuration: path constants, env loading, defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
FACTUAL_EVAL_DIR = REPO_ROOT / "factual_eval"
POINT_QUALITY_DIR = REPO_ROOT / "point_quality"
PROCESS_EVAL_DIR = REPO_ROOT / "process_eval"
DATA_DIR = REPO_ROOT / "data"

# ── Load root .env (single configuration point) ───────────────────────────
load_dotenv(REPO_ROOT / ".env", override=False)

# ── Default evaluator settings (overridable via env vars) ─────────────────
POINT_QUALITY_MODEL = os.getenv("POINT_QUALITY_MODEL", "gpt-5.1")
POINT_QUALITY_API_TYPE = os.getenv("POINT_QUALITY_API_TYPE", "openai")
POINT_QUALITY_MAX_WORKERS = int(os.getenv("POINT_QUALITY_MAX_WORKERS", "20"))

PROCESS_EVAL_MODEL = os.getenv("PROCESS_EVAL_MODEL", "gpt-5.2")
PROCESS_EVAL_API_TYPE = os.getenv("PROCESS_EVAL_API_TYPE", "openai")
PROCESS_EVAL_MAX_WORKERS = int(os.getenv("PROCESS_EVAL_MAX_WORKERS", "10"))

FACTUAL_EVAL_MAX_CONCURRENT = int(os.getenv("FACTUAL_EVAL_MAX_CONCURRENT", "10"))
FACTUAL_EVAL_MAX_CHUNKS = int(os.getenv("FACTUAL_EVAL_MAX_CHUNKS", "10"))

# ── Cache directories (absolute) ──────────────────────────────────────────
POINT_QUALITY_CACHE = str(POINT_QUALITY_DIR / "outputs" / "cache")
PROCESS_EVAL_CACHE = str(PROCESS_EVAL_DIR / "outputs" / "cache")
