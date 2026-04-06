"""Centralized configuration for MiroEval.

Single source of truth for paths, env loading, and default settings.
All values are overridable via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# ── Env loading (idempotent) ──────────────────────────────────────────────

_env_loaded = False


def load_env() -> None:
    """Load the root .env file. Safe to call multiple times."""
    global _env_loaded
    if _env_loaded:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)
    _env_loaded = True


# Auto-load on import so downstream code can rely on env vars immediately.
load_env()


# ── Default settings (all overridable via env vars) ───────────────────────

def _get(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


# Quality evaluation
QUALITY_MODEL = _get("MIROEVAL_QUALITY_MODEL", "gpt-5.1")
QUALITY_API_TYPE = _get("MIROEVAL_QUALITY_API_TYPE", "openai")
QUALITY_MAX_WORKERS = _get_int("MIROEVAL_QUALITY_MAX_WORKERS", 20)

# Process evaluation
PROCESS_MODEL = _get("MIROEVAL_PROCESS_MODEL", "gpt-5.2")
PROCESS_API_TYPE = _get("MIROEVAL_PROCESS_API_TYPE", "openai")
PROCESS_MAX_WORKERS = _get_int("MIROEVAL_PROCESS_MAX_WORKERS", 10)

# Factual evaluation
FACTUAL_MAX_CONCURRENT = _get_int("MIROEVAL_FACTUAL_MAX_CONCURRENT", 10)
FACTUAL_MAX_CHUNKS = _get_int("MIROEVAL_FACTUAL_MAX_CHUNKS", 10)
