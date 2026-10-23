"""Shared fixtures: an isolated workspace, no real Docker, no real LLM."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent import graph as graph_module
from agent import history
from agent import sandbox
from agent import tools as tools_module
from tests.fakes import ScriptedChatModel


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    """Every test gets its own conversation database: tests never touch real saved chats."""
    monkeypatch.setenv("CONVERSATION_DB", str(tmp_path / "conversations.sqlite"))
    monkeypatch.delenv("AGENT_MODE", raising=False)
    history.reset()
    graph_module._graphs.clear()
    yield
    history.reset()
    graph_module._graphs.clear()


@pytest.fixture
def ws(tmp_path, monkeypatch) -> Path:
    """A fresh workspace directory wired into the tools module."""
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    monkeypatch.setattr(tools_module, "WORKSPACE", root)
    tools_module.ensure_workspace(root)
    # No global git identity, so git_commit has to configure a local one.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    # Docker is never available in tests, and unsafe local execution is off.
    monkeypatch.setattr(sandbox, "docker_available", lambda force=False: False)
    monkeypatch.delenv("ALLOW_UNSAFE_LOCAL_EXECUTION", raising=False)
    monkeypatch.delenv("APPROVAL_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_IMAGE", raising=False)
    return root


@pytest.fixture
def git_ready(ws) -> Path:
    """Workspace that already is a git repository."""
    tools_module.git_init.invoke({})
    return ws


@pytest.fixture
def fake_llm(ws, monkeypatch):
    """Patch get_chat_model everywhere; returns a factory taking a script dict."""

    def install(script=None, responder=None) -> ScriptedChatModel:
        model = ScriptedChatModel(script=script or {}, responder=responder)
        monkeypatch.setattr(graph_module, "get_chat_model", lambda: model)
        monkeypatch.setattr(tools_module, "get_chat_model", lambda: model)
        graph_module._graphs.clear()
        return model

    yield install
    graph_module._graphs.clear()


def git(root: Path, *args: str) -> str:
    """Run git in `root` for assertions (independent of the code under test)."""
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout
