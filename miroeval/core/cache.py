"""Thread-safe file-backed JSON cache for MiroEval.

Provides ``FileCache`` (single named cache) and ``CacheManager``
(multiple named caches under one directory).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


class FileCache:
    """Thread-safe, JSON-backed key-value cache persisted to a single file."""

    def __init__(self, cache_dir: str, cache_name: str = "default"):
        self._cache_dir = cache_dir
        self._cache_name = cache_name
        self._file_path = os.path.join(cache_dir, f"{cache_name}_cache.json")
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(
                    "Cache '%s' loaded: %d entries",
                    self._cache_name,
                    len(self._data),
                )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load cache '%s': %s", self._cache_name, e)
                self._data = {}

    def _save(self) -> None:
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error("Failed to save cache '%s': %s", self._cache_name, e)

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, key: str | int, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(str(key), default)

    def set(self, key: str | int, value: Any) -> None:
        with self._lock:
            self._data[str(key)] = value
            self._save()

    def has(self, key: str | int) -> bool:
        with self._lock:
            return str(key) in self._data

    def remove(self, key: str | int) -> bool:
        with self._lock:
            k = str(key)
            if k in self._data:
                del self._data[k]
                self._save()
                return True
            return False

    def batch_set(self, items: dict[str | int, Any]) -> None:
        with self._lock:
            self._data.update({str(k): v for k, v in items.items()})
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._save()

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return list(self._data.items())


class CacheManager:
    """Manages multiple named FileCache instances under one directory."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._caches: dict[str, FileCache] = {}

    def get_cache(self, name: str) -> FileCache:
        """Get or create a named cache."""
        if name not in self._caches:
            self._caches[name] = FileCache(
                cache_dir=self._base_dir, cache_name=name
            )
        return self._caches[name]

    def clear_cache(self, name: str) -> None:
        if name in self._caches:
            self._caches[name].clear()

    def clear_all(self) -> None:
        for cache in self._caches.values():
            cache.clear()

    def sizes(self) -> dict[str, int]:
        return {name: cache.size() for name, cache in self._caches.items()}
