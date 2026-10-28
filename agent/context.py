"""Keeping the context (and therefore the cost) of every model call small.

An agent loop re-sends the whole conversation on every step, so cost grows roughly with the
square of the number of steps. Three measures keep it flat:

1. **Stage isolation.** A build stage (planner, coder, reviewer, committer) is shown a short briefing
   (the task, the hand-over note and the saved plan / todo list) plus only the messages of its own
   stage, not the transcript of the stages before it. The files in ``.agent/`` carry the rest.
2. **Compaction.** Tool results older than the most recent messages are shortened (head + tail), and
   old images are replaced by a text note. The stored conversation is never changed; only what is
   sent to the model.
3. **A hard size cap.** If the chat history is still too large, the oldest turns are dropped, and the
   transcripts of finished builds are collapsed to their outcome.

Everything is configurable with environment variables (see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

PIPELINE_PREFIX = "[pipeline]"
IMAGE_WEIGHT_CHARS = 4_000  # an image costs roughly as much as this many characters of text


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
        return value if value >= 0 else default
    except ValueError:
        return default


def max_context_chars() -> int:
    """Upper bound for the messages sent in one call (about 4 characters per token)."""
    return _int_env("CONTEXT_MAX_CHARS", 100_000)


def old_tool_chars() -> int:
    """Old tool results are shortened to about this many characters."""
    return _int_env("CONTEXT_OLD_TOOL_CHARS", 1_200)


def keep_recent() -> int:
    """This many of the newest messages keep their tool results in full."""
    return _int_env("CONTEXT_KEEP_RECENT", 8)


def keep_images() -> int:
    """Only this many of the most recent images stay in the context."""
    return _int_env("CONTEXT_KEEP_IMAGES", 2)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def text_of(content) -> str:
    """Text parts of a message content (string or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) if isinstance(b, dict) and b.get("type") in (None, "text")
                       else (b if isinstance(b, str) else "") for b in content)
    return str(content)


def has_image(message: BaseMessage) -> bool:
    content = message.content
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in ("image_url", "image") for b in content)


def is_pipeline_note(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and text_of(message.content).startswith(PIPELINE_PREFIX)


def is_user_turn(message: BaseMessage) -> bool:
    """A real message from the user (not a pipeline note and not an attached image)."""
    return isinstance(message, HumanMessage) and not is_pipeline_note(message) and not has_image(message)


def message_chars(message: BaseMessage) -> int:
    size = len(text_of(message.content))
    if has_image(message):
        size += IMAGE_WEIGHT_CHARS * sum(
            1 for b in message.content if isinstance(b, dict) and b.get("type") in ("image_url", "image"))
    for call in getattr(message, "tool_calls", None) or []:
        size += len(str(call.get("args", ""))) + len(call.get("name", "")) + 20
    return size


def total_chars(messages: Sequence[BaseMessage]) -> int:
    return sum(message_chars(m) for m in messages)


def shorten(text: str, limit: int) -> str:
    """Keep the start and the end of a long text; say how much was cut."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = max(0, limit - head)
    cut = len(text) - head - tail
    return (text[:head] + f"\n... [{cut} characters omitted to save context; call the tool again "
            f"if you need them] ...\n" + (text[-tail:] if tail else ""))


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------
def compact(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Copies of `messages` with old tool results shortened and old images removed."""
    messages = list(messages)
    n = len(messages)
    # The boundary only moves in steps, so the prefix of the conversation stays byte-identical
    # between calls (provider prompt caches keep working) instead of changing on every message.
    step = 6
    boundary = max(0, ((n - keep_recent()) // step) * step) if n > keep_recent() else 0
    limit = old_tool_chars()
    image_positions = [i for i, m in enumerate(messages) if has_image(m)]
    stale_images = set(image_positions[:-keep_images()] if keep_images() else image_positions)
    out: list[BaseMessage] = []
    for index, message in enumerate(messages):
        if index in stale_images:
            note = text_of(message.content) or "[image]"
            message = message.model_copy(update={"content": note + " (image removed to save context)"})
        elif index < boundary and isinstance(message, ToolMessage):
            text = text_of(message.content)
            if len(text) > limit:
                message = message.model_copy(update={"content": shorten(text, limit)})
        out.append(message)
    return out


def collapse_builds(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Replace the inner transcript of every FINISHED build by a one-line marker.

    A build starts with the answer to the start_build call and ends with the pipeline note that
    reports its outcome ("The build finished with status ..."). The chat only needs that outcome.
    """
    out: list[BaseMessage] = []
    skipping = False
    skipped = 0
    messages = list(messages)
    for index, message in enumerate(messages):
        if skipping:
            if is_pipeline_note(message) and "The build finished with status" in text_of(message.content):
                out.append(HumanMessage(content=f"{PIPELINE_PREFIX} (the {skipped} messages of the build "
                                                f"stages are omitted to save context)"))
                out.append(message)
                skipping = False
            else:
                skipped += 1
            continue
        out.append(message)
        if isinstance(message, ToolMessage) and message.name == "start_build" \
                and getattr(message, "status", "success") != "error":
            # Collapse only if this build is complete (its outcome note exists later on).
            if any(is_pipeline_note(m) and "The build finished with status" in text_of(m.content)
                   for m in messages[index + 1:]):
                skipping, skipped = True, 0
    return out


# ---------------------------------------------------------------------------
# Fitting into a budget
# ---------------------------------------------------------------------------
def fit(messages: Sequence[BaseMessage], max_chars: int | None = None) -> list[BaseMessage]:
    """Drop the oldest turns until the messages fit. The result always starts with a real user
    message (providers reject a conversation that starts with a tool result or assistant reply) and
    always contains the latest user message."""
    messages = list(messages)
    max_chars = max_context_chars() if max_chars is None else max_chars
    if not max_chars or total_chars(messages) <= max_chars:
        return messages
    starts = [i for i, m in enumerate(messages) if is_user_turn(m)]
    if not starts:
        return messages
    for start in starts:
        if total_chars(messages[start:]) <= max_chars:
            return messages[start:]
    return messages[starts[-1]:]  # even the newest turn alone is large: keep it whole


def chat_context(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """What the chat node sends: finished builds collapsed, old output shortened, size capped."""
    return fit(compact(collapse_builds(messages)))


# ---------------------------------------------------------------------------
# Build stages
# ---------------------------------------------------------------------------
def _read_agent_file(workspace: Path, name: str, limit: int = 5_000) -> str:
    try:
        text = (workspace / ".agent" / name).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return shorten(text, limit) if text else ""


def stage_context(messages: Sequence[BaseMessage], stage: str, stage_start: int, task: str,
                  workspace: Path) -> list[BaseMessage]:
    """Briefing + the messages of this stage only."""
    messages = list(messages)
    window = messages[stage_start:]
    note = messages[stage_start - 1] if 0 < stage_start <= len(messages) else None
    parts: list[str] = []
    note_text = text_of(note.content) if note is not None and is_pipeline_note(note) else ""
    first_is_user = bool(window) and is_user_turn(window[0])
    if note_text:
        parts.append(note_text)
    if task and not first_is_user and task[:200] not in note_text:
        parts.append(f"Task of this build:\n{task}")
    if stage != "planner":
        plan = _read_agent_file(workspace, "PLAN.md")
        todo = _read_agent_file(workspace, "TODO.md")
        if plan:
            parts.append(f"Saved plan (.agent/PLAN.md):\n{plan}")
        if todo:
            parts.append(f"Saved todo list (.agent/TODO.md):\n{todo}")
    if parts and not parts[0].startswith(PIPELINE_PREFIX):
        parts[0] = f"{PIPELINE_PREFIX} {parts[0]}"
    briefing: list[BaseMessage] = [HumanMessage(content="\n\n".join(parts))] if parts else []
    if not briefing and not window:  # should not happen; never send an empty conversation
        briefing = [HumanMessage(content=f"{PIPELINE_PREFIX} You are the {stage} stage.")]
    # In strict pipeline mode the planner starts at the user's own message: keep the conversation
    # that came before it (size-capped), so earlier chat context is not lost.
    earlier = chat_context(messages[:stage_start]) if first_is_user and stage_start else []
    return earlier + briefing + compact(window)


def build_messages(state: dict, stage: str, system_prompt: str, workspace: Path) -> list[BaseMessage]:
    """The full message list for one model call of `stage`."""
    if stage == "chat":
        body = chat_context(state["messages"])
    else:
        body = stage_context(state["messages"], stage, state.get("stage_start", 0),
                             state.get("task", ""), workspace)
    return [SystemMessage(content=system_prompt), *body]


def react_messages(messages: Sequence[BaseMessage], system_prompt: str) -> list[BaseMessage]:
    """Message list for the single-agent (react) mode."""
    return [SystemMessage(content=system_prompt), *chat_context(messages)]


__all__ = ["build_messages", "chat_context", "compact", "collapse_builds", "fit", "react_messages",
           "stage_context", "total_chars", "AIMessage"]
