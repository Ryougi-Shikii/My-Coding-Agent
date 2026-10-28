"""Token usage: how much a conversation (or one request) has used, and a per-request budget.

Providers report usage on every reply (``AIMessage.usage_metadata``) and LangGraph stores it with the
messages, so nothing extra has to be saved: the totals are computed from the conversation itself.
Calls made outside the message list (the one-shot model call inside ``review_changes``) are added
through :func:`record_extra`.

``RUN_TOKEN_BUDGET`` (default 300000, 0 = unlimited) is the number of tokens one user message may use
across all stages; when it is reached the run stops and says so instead of continuing to spend.
"""

from __future__ import annotations

import os
import threading
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from agent.context import is_user_turn

DEFAULT_BUDGET = 3_000_000

_lock = threading.Lock()
_extra: dict[str, dict[str, int]] = {}   # key (workspace path) -> extra usage from tool-side calls


def token_budget() -> int:
    try:
        return max(0, int(os.getenv("RUN_TOKEN_BUDGET", "").strip() or DEFAULT_BUDGET))
    except ValueError:
        return DEFAULT_BUDGET


def _numbers(metadata: dict | None) -> dict[str, int]:
    metadata = metadata or {}
    inp = int(metadata.get("input_tokens") or 0)
    out = int(metadata.get("output_tokens") or 0)
    total = int(metadata.get("total_tokens") or (inp + out))
    return {"input": inp, "output": out, "total": total}


def message_usage(message: BaseMessage) -> dict[str, int]:
    if isinstance(message, AIMessage):
        return _numbers(getattr(message, "usage_metadata", None))
    return {"input": 0, "output": 0, "total": 0}


def sum_usage(messages: Sequence[BaseMessage]) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "total": 0, "calls": 0}
    for message in messages:
        used = message_usage(message)
        if used["total"]:
            totals["calls"] += 1
        for key in ("input", "output", "total"):
            totals[key] += used[key]
    return totals


def turn_usage(messages: Sequence[BaseMessage]) -> dict[str, int]:
    """Usage since the last real user message (the current request)."""
    messages = list(messages)
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        if is_user_turn(messages[index]):
            start = index
            break
    return sum_usage(messages[start:])


def record_extra(key: str, metadata: dict | None) -> None:
    used = _numbers(metadata)
    if not used["total"]:
        return
    with _lock:
        bucket = _extra.setdefault(key, {"input": 0, "output": 0, "total": 0, "calls": 0})
        for name in ("input", "output", "total"):
            bucket[name] += used[name]
        bucket["calls"] += 1


def extra_usage(key: str) -> dict[str, int]:
    with _lock:
        return dict(_extra.get(key, {"input": 0, "output": 0, "total": 0, "calls": 0}))


def summary(messages: Sequence[BaseMessage], key: str | None = None) -> dict[str, int]:
    """Conversation totals including tool-side model calls recorded for `key`."""
    totals = sum_usage(messages)
    if key:
        extra = extra_usage(key)
        for name in ("input", "output", "total", "calls"):
            totals[name] += extra.get(name, 0)
    return totals


def over_budget(messages: Sequence[BaseMessage]) -> tuple[bool, int, int]:
    """(exceeded, tokens used by this request, budget)."""
    budget = token_budget()
    used = turn_usage(messages)["total"]
    return (bool(budget) and used >= budget, used, budget)


def format_usage(totals: dict[str, int]) -> str:
    return (f"{totals['total']:,} tokens ({totals['input']:,} in / {totals['output']:,} out, "
            f"{totals['calls']} model calls)")
