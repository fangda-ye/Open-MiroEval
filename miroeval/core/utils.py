"""Shared utility functions for MiroEval."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict | list | None:
    """Extract JSON object or array from LLM response text.

    Tries, in order:
    1. Direct ``json.loads``
    2. Fenced code block (```json ... ```)
    3. ``<json_output>`` XML tags
    4. First balanced ``{ ... }`` or ``[ ... ]`` block
    """
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Fenced code block
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. <json_output> tags
    match = re.search(r"<json_output>\s*(.*?)\s*</json_output>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 4. Bracket-matching extraction
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    logger.warning("Failed to extract JSON from response (len=%d)", len(text))
    return None


def load_entries(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load evaluation entries from a JSON file.

    Accepts:
    - Plain JSON array: ``[{entry}, ...]``
    - Wrapped object: ``{"entries": [...], "model_name": "...", ...}``

    Returns ``(entries, metadata)``.
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
