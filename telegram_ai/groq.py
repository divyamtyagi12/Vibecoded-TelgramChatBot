"""Backward-compatibility shim re-exporting OpenRouterClient as GroqClient."""

from __future__ import annotations

import httpx  # noqa: F401

from telegram_ai.openrouter import (  # noqa: F401
    OpenRouterClient,
    QuotaExhaustedError,
    _CODE_KEYWORDS_RE,
    _DEFAULT_VISION_MODEL,
    _DETAILED_KEYWORDS_RE,
    _FALLBACK_MODELS,
    _FALLBACK_VISION_MODELS,
    _GROQ_API_URL,
    _LIVE_KEYWORDS_RE,
    _OPENROUTER_API_URL,
    _OPENROUTER_FALLBACK_MODELS,
    _OPENROUTER_FREE_MODEL,
    _REFERENTIAL_REPLY_RE,
    _TOOLS,
    _clean_chat_response,
    _redact_secrets,
)

GroqClient = OpenRouterClient

__all__ = [
    "GroqClient",
    "OpenRouterClient",
    "QuotaExhaustedError",
    "httpx",
]
