"""LLM abstraction: one place that turns environment variables into a chat model."""

from __future__ import annotations

import os
import threading
from typing import Any

from langchain.chat_models import init_chat_model

# Sensible starting points. Set MODEL_NAME to override; model names change often.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-5-5",
    "google_genai": "gemini-2.0-flash",
    "xai": "grok-4",
}

# Friendly spellings people tend to put in LLM_PROVIDER.
PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google_genai",
    "google": "google_genai",
    "google-genai": "google_genai",
    "grok": "xai",
    "claude": "anthropic",
}

DEFAULT_PROVIDER = "google_genai"

_cache: dict[tuple, Any] = {}
_lock = threading.Lock()


def get_provider() -> str:
    """Return the normalised provider name from LLM_PROVIDER."""
    raw = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    return PROVIDER_ALIASES.get(raw, raw)


def get_model_name(provider: str | None = None) -> str:
    """Return MODEL_NAME, or the provider's default when it is not set."""
    provider = provider or get_provider()
    name = os.getenv("MODEL_NAME", "").strip()
    if name:
        return name
    if provider in DEFAULT_MODELS:
        return DEFAULT_MODELS[provider]
    raise ValueError(
        f"MODEL_NAME is not set and there is no default model for provider '{provider}'. "
        f"Set MODEL_NAME in your environment or .env file."
    )


def describe_llm() -> str:
    """Human-readable 'provider:model' label (does not initialise the model)."""
    try:
        provider = get_provider()
        return f"{provider}:{get_model_name(provider)}"
    except ValueError as exc:  # unknown provider and no MODEL_NAME
        return f"unconfigured ({exc})"


def get_chat_model():
    """Return the chat model, creating it once and caching it globally.

    The cache key includes provider/model/temperature, so the model is not rebuilt
    per request but a changed configuration (e.g. in tests) is still honoured.
    """
    provider = get_provider()
    name = get_model_name(provider)
    temperature = os.getenv("LLM_TEMPERATURE", "").strip()
    key = (provider, name, temperature)
    with _lock:
        if key not in _cache:
            kwargs: dict[str, Any] = {"model_provider": provider}
            # Temperature is optional: some reasoning models reject non-default values.
            if temperature:
                kwargs["temperature"] = float(temperature)
            _cache[key] = init_chat_model(name, **kwargs)
        return _cache[key]


def reset_chat_model_cache() -> None:
    """Drop cached models (used by tests)."""
    with _lock:
        _cache.clear()


def content_to_text(content: Any) -> str:
    """Flatten message content to plain text.

    Providers such as Anthropic and Gemini may return a list of content blocks
    instead of a string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)
