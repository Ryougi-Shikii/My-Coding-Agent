"""CLI: scripted conversations with a fake model."""

from __future__ import annotations

import pytest

from agent import graph as g
import cli
from tests.conftest import git
from tests.fakes import ai, call
from tests.test_graph import committer_steps, happy_script, planner_steps


class Terminal:
    """Scripted stdin/stdout."""

    def __init__(self, *inputs):
        self.inputs = list(inputs)
        self.prompts: list[str] = []
        self.lines: list[str] = []

    def input(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    def out(self, text: str = "") -> None:
        self.lines.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def approval_prompts(self) -> int:
        return sum("Approve?" in p for p in self.prompts)


def run(term: Terminal, *argv: str) -> int:
    return cli.run_cli(list(argv), input_fn=term.input, output_fn=term.out)


def test_full_pipeline_conversation_with_two_approvals(fake_llm, ws):
    fake_llm(happy_script())
    term = Terminal("write hello.py", "y", "y", "/exit")
    assert run(term) == 0
    assert term.approval_prompts() == 2  # write_file and git_commit only
    for stage in ("planner", "coder", "reviewer", "committer"):
        assert f"[{stage}]" in term.text
    assert "The agent wants to run:" in term.text and "needs approval" in term.text
    assert "content: print('hi')" in term.text or "print('hi')" in term.text  # args are shown
    assert (ws / "hello.py").exists() and "Add hello.py" in git(ws, "log", "--oneline")


def test_rejecting_tells_the_agent_and_changes_nothing(fake_llm, ws):
    script = {"planner": planner_steps(),
              "coder": [ai("", call("write_file", path="hello.py", content="x")),
                        ai("OK, skipping the file.")]}
    fake_llm(script)
    term = Terminal("write hello.py", "n", "too risky", "/exit")
    assert run(term) == 0
    assert "Rejected" in term.text and "REJECTED by the user" in term.text
    assert "OK, skipping the file." in term.text
    assert not (ws / "hello.py").exists()


def test_invalid_answer_reprompts(fake_llm, ws):
    script = {"planner": planner_steps(),
              "coder": [ai("", call("write_file", path="a.py", content="x=1\n")), "done"],
              "reviewer": ["no review"] * 5}
    fake_llm(script)
    term = Terminal("go", "maybe", "y", "/exit")
    run(term)
    assert "Please answer y or n." in term.text and (ws / "a.py").exists()


def test_react_mode_flag(fake_llm, ws):
    fake_llm({"react": ["hello from react"]})
    term = Terminal("hi", "/exit")
    assert run(term, "--mode", "react") == 0
    assert "hello from react" in term.text and "mode:       react" in term.text


def test_commands(fake_llm, ws):
    fake_llm({})
    term = Terminal("", "/help", "/state", "/status", "/new", "/exit")
    assert run(term) == 0
    assert "/exit" in term.text and "## PLAN.md" in term.text
    assert "Started a new conversation." in term.text and "'stage'" in term.text


def test_header_shows_sandbox_and_approval_info(fake_llm, ws):
    fake_llm({})
    term = Terminal("/exit")
    run(term)
    assert "docker:     NOT available" in term.text and "python:3.12-slim" in term.text
    assert "approval:   dangerous" in term.text and str(ws) in term.text


def test_eof_exits_cleanly(fake_llm, ws):
    fake_llm({})
    assert run(Terminal()) == 0


def test_startup_failure_is_reported(ws, monkeypatch):
    def boom():
        raise RuntimeError("missing API key")

    g._graphs.clear()
    monkeypatch.setattr(g, "get_chat_model", boom)
    term = Terminal()
    assert run(term) == 2
    assert "Could not start the agent" in term.text and "missing API key" in term.text


def test_runtime_error_does_not_kill_the_session(fake_llm, ws):
    def responder(key, messages):
        raise ConnectionError("network down")

    fake_llm({}, responder)
    term = Terminal("hello", "/exit")
    assert run(term) == 0
    assert "Error: ConnectionError: network down" in term.text


def test_format_call_marks_danger_and_truncates():
    text = cli.format_call({"name": "write_file", "dangerous": True,
                            "args": {"path": "a.py", "content": "x\n" * 500}})
    assert "[needs approval]" in text and "more chars" in text
    assert "[safe]" in cli.format_call({"name": "read_file", "dangerous": False, "args": {"path": "a"}})


@pytest.mark.parametrize("flag", ["--mode"])
def test_bad_mode_is_rejected_by_argparse(flag, ws):
    with pytest.raises(SystemExit):
        cli.run_cli([flag, "chaos"], input_fn=lambda p: "", output_fn=lambda t: None)
