"""Saved conversations: the store, and resuming/switching from the CLI."""

from __future__ import annotations

import time

import pytest

import cli
from agent import graph as g
from agent import history
from tests.fakes import ai, call
from tests.test_cli import Terminal


def restart():
    """Simulate closing the terminal: drop every in-process handle, keep the database file."""
    history.reset()
    g._graphs.clear()


def run(term, *argv):
    return cli.run_cli(list(argv), input_fn=term.input, output_fn=term.out)


# ------------------------------------------------------------------------------ the store
class TestStore:
    def test_record_list_find_delete(self, ws):
        history.record("aaaa1111", "auto", "/w", "first question")
        time.sleep(0.01)
        history.record("bbbb2222", "react", "/w", "second   question\nwith lines")
        history.record("cccc3333", "auto", "/other", "elsewhere")
        rows = history.list_conversations("/w")
        assert [r["thread_id"] for r in rows] == ["bbbb2222", "aaaa1111"]  # newest first, per workspace
        assert rows[0]["title"] == "second question with lines"
        assert history.find_conversation("/w", "1")["thread_id"] == "bbbb2222"
        assert history.find_conversation("/w", "aaaa")["thread_id"] == "aaaa1111"
        assert history.find_conversation("/w", "bbbb2222")["thread_id"] == "bbbb2222"
        assert history.find_conversation("/w", "9") is None and history.find_conversation("/w", "zzz") is None
        assert history.find_conversation("/w", "cccc") is None  # other workspace
        assert history.latest_conversation("/w")["thread_id"] == "bbbb2222"
        history.delete_conversation("bbbb2222")
        assert [r["thread_id"] for r in history.list_conversations("/w")] == ["aaaa1111"]

    def test_title_is_kept_but_mode_and_time_update(self):
        history.record("t1", "auto", "/w", "original title")
        before = history.get_conversation("t1")["updated_at"]
        time.sleep(0.01)
        history.record("t1", "react", "/w", "ignored later text")
        row = history.get_conversation("t1")
        assert row["title"] == "original title" and row["mode"] == "react" and row["updated_at"] > before

    def test_ambiguous_prefix_matches_nothing(self):
        history.record("abc1", "auto", "/w", "a")
        history.record("abc2", "auto", "/w", "b")
        assert history.find_conversation("/w", "abc") is None

    def test_survives_a_restart_and_memory_mode_does_not(self, monkeypatch):
        history.record("t1", "auto", "/w", "persisted")
        history.reset()
        assert history.get_conversation("t1")["title"] == "persisted"
        monkeypatch.setenv("CONVERSATION_DB", "memory")
        history.reset()
        assert not history.persistent() and history.list_conversations("/w") == []
        history.record("m1", "auto", "/w", "volatile")
        assert history.find_conversation("/w", "1")["thread_id"] == "m1"
        history.delete_conversation("m1")
        assert history.list_conversations("/w") == []

    def test_format_age_and_title(self):
        now = 1_000_000.0
        assert history.format_age(now - 5, now) == "just now"
        assert history.format_age(now - 300, now) == "5m ago"
        assert history.format_age(now - 7200, now) == "2h ago"
        assert history.format_age(now - 3 * 86400, now) == "3d ago"
        assert len(history.make_title("x" * 200)) == 60 and history.make_title("  ") == "(untitled)"


# ------------------------------------------------------------------------------ CLI
class TestCliResume:
    def test_continue_picks_up_the_last_conversation_after_a_restart(self, fake_llm, ws):
        fake_llm({"chat": ["Teal is nice."]})
        assert run(Terminal("my favourite colour is teal", "/exit")) == 0
        restart()
        model = fake_llm({"chat": ["You told me teal."]})
        term = Terminal("what did I say?", "/exit")
        assert run(term, "--continue") == 0
        assert "Resumed: my favourite colour is teal" in term.text
        assert "you> my favourite colour is teal" in term.text and "Teal is nice." in term.text  # recap
        assert model.log[-1][1] >= 3  # the model saw the earlier exchange plus the new question
        assert len(history.list_conversations(str(ws))) == 1  # same conversation, not a new one

    def test_without_continue_a_new_conversation_starts(self, fake_llm, ws):
        fake_llm({"chat": ["one", "two"]})
        run(Terminal("first", "/exit"))
        restart()
        fake_llm({"chat": ["two"]})
        run(Terminal("second", "/exit"))
        assert len(history.list_conversations(str(ws))) == 2

    def test_continue_with_nothing_saved_starts_fresh(self, fake_llm, ws):
        fake_llm({})
        term = Terminal("/exit")
        assert run(term, "--continue") == 0 and "No saved conversation to continue" in term.text

    def test_list_flag_and_unknown_resume(self, fake_llm, ws):
        fake_llm({"chat": ["hello!"]})
        run(Terminal("hi there", "/exit"))
        restart()
        term = Terminal()
        assert run(term, "--list") == 0 and "hi there" in term.text and "auto" in term.text
        assert run(Terminal(), "--resume", "99") == 2

    def test_resume_by_number_inside_the_session(self, fake_llm, ws):
        fake_llm({"chat": ["a1", "b1", "a2"]})
        run(Terminal("topic A", "/new", "topic B", "/exit"))
        restart()
        fake_llm({"chat": ["a2"]})
        term = Terminal("/list", "/resume 2", "more on A", "/exit")
        run(term)
        assert "1. topic B" in term.text and "2. topic A" in term.text
        assert "Resumed: topic A" in term.text and "a2" in term.text
        titles = {r["title"]: r for r in history.list_conversations(str(ws))}
        assert len(titles) == 2

    def test_resume_picker_and_cancel(self, fake_llm, ws):
        fake_llm({"chat": ["x"]})
        run(Terminal("only one", "/exit"))
        restart()
        fake_llm({})
        term = Terminal("/resume", "", "/resume", "1", "/exit")
        run(term)
        assert any("Resume which one?" in p for p in term.prompts) and "Resumed: only one" in term.text

    def test_delete_confirms_and_removes(self, fake_llm, ws):
        fake_llm({"chat": ["x", "y"]})
        term = Terminal("first", "/new", "second", "/delete 1", "n", "/delete 1", "y", "/list", "/exit")
        run(term)
        assert "Kept." in term.text and "Deleted." in term.text
        assert [r["title"] for r in history.list_conversations(str(ws))] == ["first"]

    def test_deleting_the_open_conversation_starts_a_new_one(self, fake_llm, ws):
        fake_llm({"chat": ["x"]})
        term = Terminal("hello", "/delete 1", "y", "/exit")
        run(term)
        assert "Started a new conversation." in term.text and history.list_conversations(str(ws)) == []

    def test_pending_approval_is_asked_again_after_a_restart(self, fake_llm, ws):
        script = {"chat": [ai("", call("run_tests")), "Tests pass."]}
        fake_llm(script)
        # the terminal closes while the approval prompt is open (EOF)
        assert run(Terminal("run the tests")) == 0
        restart()
        fake_llm({"chat": ["Tests pass."]})
        term = Terminal("y", "/exit")
        run(term, "--continue")
        assert "waiting for your approval" in term.text and "run_tests" in term.text
        assert term.approval_prompts() == 1

    def test_resumed_conversation_uses_its_stored_mode_unless_overridden(self, fake_llm, ws):
        fake_llm({"react": ["r"]})
        run(Terminal("hi", "/exit"), "--mode", "react")
        restart()
        fake_llm({})
        term = Terminal("/exit")
        run(term, "--continue")
        assert "mode:       react" in term.text
        restart()
        fake_llm({})
        term = Terminal("/exit")
        run(term, "--continue", "--mode", "auto")
        assert "mode:       auto" in term.text

    def test_mode_command_keeps_the_conversation(self, fake_llm, ws):
        model = fake_llm({"chat": ["chat reply"], "react": ["react reply"]})
        term = Terminal("hello", "/mode", "/mode react", "again", "/mode bogus", "/exit")
        run(term)
        assert "Mode is now react" in term.text and "react reply" in term.text
        assert "Unknown mode 'bogus'" in term.text
        (row,) = history.list_conversations(str(ws))
        assert row["mode"] == "react"

    def test_mode_change_blocked_while_approval_is_pending(self, fake_llm, ws):
        fake_llm({"chat": [ai("", call("run_tests"))]})
        run(Terminal("run them"))  # EOF at the approval prompt leaves it pending
        restart()
        fake_llm({})
        term = Terminal("n", "", "/mode react", "/exit")
        run(term, "--continue")  # resume asks; answering 'n' clears it, then /mode works
        assert "Mode is now react" in term.text

    def test_unknown_command(self, fake_llm, ws):
        fake_llm({})
        term = Terminal("/frobnicate", "/exit")
        run(term)
        assert "Unknown command /frobnicate" in term.text
