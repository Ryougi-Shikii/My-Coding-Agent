"""Smoke-test app.py WITHOUT Streamlit or a server.

app.py is executed against a small fake `streamlit` module that records widget calls and lets
the test press buttons / type into the chat box, emulating Streamlit's rerun model. This checks
the control flow (history, approval buttons, approve/reject, new conversation), not pixels.
"""

from __future__ import annotations

import functools
import runpy
import sys
from pathlib import Path

import pytest

from agent import graph as g
from tests.conftest import git
from tests.fakes import ai, call
from tests.test_graph import happy_script, planner_steps

APP = Path(__file__).resolve().parent.parent / "app.py"


class RerunSignal(Exception):
    pass


class StopSignal(Exception):
    pass


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value


class FakeStreamlit:
    def __init__(self):
        self.session_state = AttrDict()
        self.pressed: set[str] = set()
        self.next_prompt: str | None = None
        self.text_values: dict[str, str] = {}
        self.log: list[tuple] = []  # widget calls of the current script run
        self.sidebar = self

    # `with st.sidebar:`, `with st.chat_message(...):`, `with st.expander(...):`
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def button(self, label, **kwargs):
        self.log.append(("button", label))
        return label in self.pressed

    def chat_input(self, placeholder="", disabled=False, **kwargs):
        self.log.append(("chat_input", disabled))
        value, self.next_prompt = self.next_prompt, None  # one-shot, like the real widget
        return None if disabled else value

    def text_input(self, label, key=None, **kwargs):
        return self.text_values.get(key, "")

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        if key and key not in self.session_state:
            self.session_state[key] = options[index]
        return self.session_state.get(key, options[index])

    def columns(self, spec, **kwargs):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def cache_resource(self, func=None, **kwargs):
        return functools.cache(func) if func else functools.cache

    def stop(self):
        raise StopSignal

    def rerun(self):
        raise RerunSignal

    def __getattr__(self, name):  # every other st.* call is just recorded
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            self.log.append((name, args))
            return self

        return method

    # helpers for assertions
    def shown(self, name: str) -> list:
        return [entry[1] for entry in self.log if entry[0] == name]

    def buttons(self) -> list[str]:
        return [entry[1] for entry in self.log if entry[0] == "button"]


@pytest.fixture
def app(fake_llm, ws, monkeypatch):
    fake = FakeStreamlit()
    fake.session_state.mode = "pipeline"  # these flows are the strict pipeline; auto has its own tests
    monkeypatch.setitem(sys.modules, "streamlit", fake)

    def run_script(max_runs: int = 12) -> None:
        """Execute app.py, re-executing after every st.rerun(), until the page settles."""
        for _ in range(max_runs):
            fake.log = []
            try:
                runpy.run_path(str(APP), run_name="__main__")
            except RerunSignal:
                fake.pressed = set()  # buttons are only 'pressed' for one run
                continue
            except StopSignal:
                pass
            fake.pressed = set()
            return
        raise AssertionError("app.py kept rerunning")

    fake.run = run_script
    fake.install_model = fake_llm
    return fake


def test_first_render_shows_sidebar_info_and_empty_chat(app):
    app.install_model({})
    app.run()
    assert ("chat_input", False) in app.log
    assert "Docker not available — code will not run" in app.shown("error")[0]
    assert "New conversation" in app.buttons()
    assert not {"✅ Approve", "⛔ Reject"} & set(app.buttons())
    assert app.session_state.messages == [] and app.session_state.thread_id


def test_workspace_files_are_listed_in_sidebar(app, ws):
    (ws / "notes.txt").write_text("x")
    app.install_model({})
    app.run()
    listed = [args[0] for args in app.shown("text")]
    assert "📄 notes.txt" in listed and "📁 .agent" in listed
    assert str(ws) in [args[0] for args in app.shown("caption")]  # the live workspace path


def test_dangerous_call_shows_approve_and_reject_and_blocks_input(app, ws):
    app.install_model(happy_script())
    app.run()
    thread_id = app.session_state.thread_id
    app.next_prompt = "write hello.py"
    app.run()
    # The graph is paused: buttons are shown, input is locked, nothing has been written yet.
    assert {"✅ Approve", "⛔ Reject"} <= set(app.buttons())
    assert ("chat_input", True) in app.log
    assert not (ws / "hello.py").exists()
    assert app.session_state.thread_id == thread_id
    kinds = [m["kind"] for m in app.session_state.messages]
    assert kinds[0] == "user" and "ai" in kinds and "tool" in kinds  # safe tools already ran


def test_approve_resumes_and_finishes_pipeline(app, ws):
    app.install_model(happy_script())
    app.run()
    app.next_prompt = "write hello.py"
    app.run()
    app.pressed = {"✅ Approve"}
    app.run()  # write_file runs, pipeline proceeds to the commit approval
    assert (ws / "hello.py").exists() and "✅ Approve" in app.buttons()
    assert not git(ws, "log", "--oneline").strip()  # not committed yet
    app.pressed = {"✅ Approve"}
    app.run()  # git_commit runs, pipeline finishes
    assert "Add hello.py" in git(ws, "log", "--oneline")
    assert not {"✅ Approve", "⛔ Reject"} & set(app.buttons())
    assert ("chat_input", False) in app.log
    assert any("reviewed and committed" in args[0] for args in app.shown("success"))
    stages = [m["node"] for m in app.session_state.messages if m["kind"] == "ai"]
    assert {"planner", "coder", "reviewer", "committer"} <= set(stages)


def test_reject_injects_tool_message_and_lets_the_agent_respond(app, ws):
    script = {"planner": planner_steps(),
              "coder": [ai("", call("write_file", path="hello.py", content="x")),
                        ai("Okay, I won't write it.")]}
    app.install_model(script)
    app.run()
    app.next_prompt = "write hello.py"
    app.run()
    app.text_values["reject_reason"] = "not now"
    app.pressed = {"⛔ Reject"}
    app.run()
    messages = app.session_state.messages
    rejected = [m for m in messages if m["kind"] == "tool" and m["error"]]
    assert rejected and "REJECTED" in rejected[0]["content"] and "not now" in rejected[0]["content"]
    assert any(m["kind"] == "ai" and "won't write it" in m["content"] for m in messages)
    assert not (ws / "hello.py").exists()
    assert not {"✅ Approve", "⛔ Reject"} & set(app.buttons())  # nothing pending any more
    assert any("no file changes" in args[0] for args in app.shown("info"))


def test_new_conversation_resets_thread_and_messages(app, ws):
    app.install_model(happy_script())
    app.run()
    app.next_prompt = "write hello.py"
    app.run()
    old_thread = app.session_state.thread_id
    assert app.session_state.messages
    app.pressed = {"New conversation"}
    app.run()
    assert app.session_state.thread_id != old_thread
    assert app.session_state.messages == []
    assert not {"✅ Approve", "⛔ Reject"} & set(app.buttons())  # the new thread has no pending call


def test_react_mode_in_the_app(app, ws):
    (ws / "a.txt").write_text("old")
    app.install_model({"react": [ai("", call("read_file", path="a.txt")),
                                 ai("", call("write_file", path="a.txt", content="new")), "done"]})
    app.session_state.mode = "react"
    app.run()
    app.next_prompt = "edit a.txt"
    app.run()
    assert {"✅ Approve", "⛔ Reject"} <= set(app.buttons())
    assert (ws / "a.txt").read_text() == "old"
    app.pressed = {"✅ Approve"}
    app.run()
    assert (ws / "a.txt").read_text() == "new"


def test_startup_error_is_shown_instead_of_crashing(app, ws, monkeypatch):
    def boom():
        raise RuntimeError("GOOGLE_API_KEY missing")

    g._graphs.clear()
    monkeypatch.setattr(g, "get_chat_model", boom)
    app.run()
    assert any("Could not start the agent" in args[0] and "GOOGLE_API_KEY missing" in args[0]
               for args in app.shown("error"))
    assert ("chat_input", False) not in app.log and ("chat_input", True) not in app.log


def test_runtime_error_is_displayed(app, ws):
    def responder(key, messages):
        raise ConnectionError("rate limited")

    app.install_model({}, responder)
    app.run()
    app.next_prompt = "hello"
    app.run()
    assert any("ConnectionError: rate limited" in args[0] for args in app.shown("error"))
    assert ("chat_input", False) in app.log  # the user can try again


def test_app_source_uses_the_required_session_state_and_widgets():
    source = APP.read_text(encoding="utf-8")
    for needle in ("st.session_state", "thread_id", "docker_available", "get_sandbox_image",
                   "workspace_top_level", "New conversation", "Approve", "Reject",
                   "reject_tool_calls", "pending_tool_calls", "st.chat_input"):
        assert needle in source, needle


# ------------------------------------------------------------------------------ auto mode + history
def _label(app, prefix):
    matches = [b for b in app.buttons() if b.startswith(prefix)]
    assert matches, (prefix, app.buttons())
    return matches[0]


def test_auto_mode_greeting_has_no_approval(app, ws):
    app.session_state.mode = "auto"
    app.install_model({"chat": ["Hey! What shall we build?"]})
    app.run()
    app.next_prompt = "hi"
    app.run()
    assert not {"✅ Approve", "⛔ Reject"} & set(app.buttons())
    assert ("chat_input", False) in app.log
    assert [m["node"] for m in app.session_state.messages if m["kind"] == "ai"] == ["chat"]
    assert not (ws / "hello.py").exists()


def test_conversation_is_listed_and_can_be_reopened_after_new_conversation(app, ws):
    app.session_state.mode = "auto"
    app.install_model({"chat": ["Teal is nice.", "Cats are great."]})
    app.run()
    app.next_prompt = "my favourite colour is teal"
    app.run()
    first = app.session_state.thread_id
    app.pressed = {"New conversation"}
    app.run()
    assert app.session_state.thread_id != first and app.session_state.messages == []
    app.next_prompt = "tell me about cats"
    app.run()
    labels = app.buttons()
    assert any(b.startswith("● tell me about cats") for b in labels)
    assert any(b.startswith("my favourite colour is teal") for b in labels)
    app.pressed = {_label(app, "my favourite colour is teal")}
    app.run()
    assert app.session_state.thread_id == first
    shown = [(m["kind"], m["content"]) for m in app.session_state.messages]
    assert ("user", "my favourite colour is teal") in shown and ("ai", "Teal is nice.") in shown


def test_continue_last_conversation_after_the_app_restarts(app, ws):
    app.session_state.mode = "auto"
    app.install_model({"chat": ["noted"]})
    app.run()
    app.next_prompt = "remember this"
    app.run()
    # a browser refresh: new Streamlit session, same saved database
    app.session_state.clear()
    app.session_state.mode = "auto"
    g._graphs.clear()
    app.run()
    assert "▶ Continue last conversation" in app.buttons() and app.session_state.messages == []
    app.pressed = {"▶ Continue last conversation"}
    app.run()
    assert [m["content"] for m in app.session_state.messages if m["kind"] == "user"] == ["remember this"]
    assert "▶ Continue last conversation" not in app.buttons()


def test_pending_approval_survives_reopening_and_locks_the_sidebar(app, ws):
    app.session_state.mode = "auto"
    app.install_model({"chat": [ai("", call("run_tests")), "ok"]})
    app.run()
    app.next_prompt = "run the tests"
    app.run()
    assert {"✅ Approve", "⛔ Reject"} <= set(app.buttons())
    app.session_state.clear()
    app.session_state.mode = "auto"
    g._graphs.clear()
    app.run()
    assert "✅ Approve" not in app.buttons()  # a new, empty conversation
    app.pressed = {"▶ Continue last conversation"}
    app.run()
    assert {"✅ Approve", "⛔ Reject"} <= set(app.buttons())  # the same pending call is back
    assert ("chat_input", True) in app.log


def test_delete_conversation_from_the_sidebar(app, ws):
    app.session_state.mode = "auto"
    app.install_model({"chat": ["a"]})
    app.run()
    app.next_prompt = "to be deleted"
    app.run()
    app.pressed = {_label(app, "🗑")}
    app.run()
    assert app.session_state.messages == []  # the open conversation was deleted -> fresh one
    assert not any(b.startswith("to be deleted") for b in app.buttons())


def test_mode_change_keeps_the_conversation(app, ws):
    app.session_state.mode = "auto"
    app.install_model({"chat": ["hello"], "react": ["hello again"]})
    app.run()
    app.next_prompt = "hi"
    app.run()
    thread = app.session_state.thread_id
    app.session_state.mode_choice = "react"
    app.session_state.mode = "react"  # what the on_change callback does
    app.run()
    app.next_prompt = "still there?"
    app.run()
    assert app.session_state.thread_id == thread
    assert [m["content"] for m in app.session_state.messages if m["kind"] == "ai"][-1] == "hello again"