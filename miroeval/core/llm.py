"""Unified synchronous LLM client for MiroEval.

Supports OpenAI and OpenRouter APIs with retry, exponential backoff,
and cost tracking.  Used by quality and process evaluation modules.

Note: factual evaluation uses MiroFlow's own async LLM client internally.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from openai import OpenAI

from miroeval.core.utils import extract_json

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """Synchronous LLM client with retry and cost tracking."""

    # Class-level cost accumulator across all instances.
    _global_cost: float = 0.0

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_type: str = "auto",
        max_tokens: int = 8192,
        temperature: float = 0.1,
        retry_count: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_count = retry_count
        self.retry_backoff = retry_backoff
        self.instance_cost: float = 0.0

        # Auto-detect API type: "/" in model name → openrouter, else openai.
        if api_type == "auto":
            api_type = "openrouter" if "/" in model else "openai"

        self.api_type = api_type
        self._client = self._build_client(api_type)

    # ── Client construction ───────────────────────────────────────────────

    @staticmethod
    def _build_client(api_type: str) -> OpenAI:
        if api_type == "openrouter":
            return OpenAI(
                base_url=os.environ.get(
                    "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
                ),
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )
        # Default: openai-compatible
        kwargs: dict[str, Any] = {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        }
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    # ── Generation ────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a text completion with retry + exponential backoff.

        Returns ``"$ERROR$"`` after all retries are exhausted.
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        for attempt in range(self.retry_count):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
                self._track_cost(response)

                # Warn if a reasoning model consumed tokens but returned empty.
                if not content.strip() and response.usage:
                    details = getattr(
                        response.usage, "completion_tokens_details", None
                    )
                    reasoning = (
                        getattr(details, "reasoning_tokens", 0) if details else 0
                    )
                    if reasoning:
                        logger.warning(
                            "Reasoning model used %d tokens but produced empty "
                            "content. Consider increasing max_tokens (currently %d).",
                            reasoning,
                            max_tokens,
                        )

                return content

            except Exception as e:
                wait = self.retry_backoff**attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    self.retry_count,
                    e,
                    wait,
                )
                time.sleep(wait)

        logger.error("LLM call failed after %d attempts", self.retry_count)
        return "$ERROR$"

    def generate_json(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict | list | None:
        """Generate and parse JSON from the response."""
        raw = self.generate(messages, **kwargs)
        if raw == "$ERROR$":
            return None
        return extract_json(raw)

    # ── Cost tracking ─────────────────────────────────────────────────────

    def _track_cost(self, response: Any) -> None:
        """Accumulate cost estimate from usage data."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        # Generic estimate: $1.25/M input, $10/M output.
        cost = (prompt_tokens * 1.25 + completion_tokens * 10) / 1_000_000
        self.instance_cost += cost
        LLMClient._global_cost += cost

    @classmethod
    def global_cost(cls) -> float:
        """Total cost across all LLMClient instances."""
        return cls._global_cost

    @classmethod
    def reset_global_cost(cls) -> None:
        cls._global_cost = 0.0
