"""Test doubles: a scripted chat model that needs no API key."""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = itertools.count(1)


def call(name: str, **args) -> dict:
    """A tool call dict as found in AIMessage.tool_calls."""
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def ai(text: str = "", *calls: dict) -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


def stage_key(messages) -> str:
    """Which role is asking? Decided from the system prompt (first message)."""
    system = str(messages[0].content)
    for stage in ("CHAT", "PLANNER", "CODER", "REVIEWER", "COMMITTER"):
        if f"ROLE: {stage}" in system:
            return stage.lower()
    if "strict senior code reviewer" in system:
        return "review"  # the one-shot model call inside the review_changes tool
    return "react"


class ScriptedChatModel(BaseChatModel):
    """Replays queued replies per role. Anything exhausted answers with plain text.

    `script` maps role -> list of AIMessage (or str for the review role). `responder`, if
    given, is called first with (key, messages) and may return a message to override the queue.
    """

    script: dict = Field(default_factory=dict)
    responder: Any = None
    log: list = Field(default_factory=list)  # (role, number of messages) per call

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # tools are irrelevant to the script
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        key = stage_key(messages)
        self.log.append((key, len(messages)))
        reply = None
        if self.responder is not None:
            reply = self.responder(key, messages)
        if reply is None:
            queue = self.script.get(key, [])
            reply = queue.pop(0) if queue else "(script exhausted)"
        if isinstance(reply, str):
            reply = AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def roles(self) -> list[str]:
        return [role for role, _ in self.log]
