"""Command-line front end for the coding agent.

    python cli.py                 # pipeline mode (planner -> coder -> reviewer -> committer)
    python cli.py --mode react    # classic single-agent mode

Safe tools run automatically; dangerous ones (write/edit/run/install/git_commit) pause and ask
for your approval. Type /help inside the prompt for commands.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Callable

from agent import sandbox
from agent.graph import (
    MODES,
    default_mode,
    drive,
    get_graph,
    get_run_status,
    make_config,
    message_to_event,
    pending_tool_calls,
    reject_tool_calls,
)
from agent import tools as agent_tools
from agent.llm import describe_llm
from agent.tools import approval_mode, read_agent_state

HELP = """\
Commands:
  /help     show this help
  /state    show the saved plan, todo list and memory
  /status   show pipeline stage / status of this conversation
  /new      start a new conversation (new thread)
  /exit     quit
Anything else is sent to the agent."""

PREVIEW_CHARS = 600   # per tool argument, when proposing a tool call
RESULT_CHARS = 1500   # per tool result, when displaying it


class Style:
    """Tiny ANSI helper; becomes a no-op when output is not a terminal."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, t: str) -> str: return self._wrap("2", t)
    def bold(self, t: str) -> str: return self._wrap("1", t)
    def red(self, t: str) -> str: return self._wrap("31", t)
    def green(self, t: str) -> str: return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def cyan(self, t: str) -> str: return self._wrap("36", t)


def _clip(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def format_call(call: dict) -> str:
    """Multi-line description of one proposed tool call."""
    lines = [f"{call['name']}" + ("  [needs approval]" if call["dangerous"] else "  [safe]")]
    for key, value in call["args"].items():
        text = _clip(value, PREVIEW_CHARS)
        if "\n" in text:
            text = "\n      " + text.replace("\n", "\n      ")
        lines.append(f"    {key}: {text}")
    return "\n".join(lines)


def print_event(event: dict, out: Callable[[str], None], st: Style) -> None:
    kind = event["kind"]
    if kind == "ai":
        if event["content"].strip():
            label = "[" + event["node"] + "]"
            out(st.cyan(st.bold(label)) + " " + event["content"].strip())
        for call in event["tool_calls"]:
            note = "needs approval" if call["dangerous"] else "auto"
            out(st.dim("  -> " + call["name"] + " (" + note + ")"))
    elif kind == "tool":
        color = st.red if event["error"] else st.dim
        out(color(f"  <- {event['name']}: {_clip(event['content'], RESULT_CHARS)}"))
    else:  # pipeline notes
        out(st.dim(f"  >> {_clip(event['content'], RESULT_CHARS)}"))


def run_turn(graph, config, text: str | None, ask: Callable[[str], str],
             out: Callable[[str], None], st: Style) -> None:
    """Send one user message and keep going until the graph is finished.

    Dangerous tool batches trigger an approval prompt; safe ones are handled by `drive`.
    """
    events = drive(graph, config, text)
    while True:
        for event in events:
            print_event(event, out, st)
        pending = pending_tool_calls(graph, config)
        if not pending:
            return
        out(st.yellow(st.bold("\nThe agent wants to run:")))
        for call in pending:
            out("  " + format_call(call))
        while True:
            answer = ask(st.bold("Approve? [y]es / [n]o: ")).strip().lower()
            if answer in ("y", "yes"):
                events = drive(graph, config, None)
                break
            if answer in ("n", "no"):
                reason = ask("Reason (optional): ").strip()
                # The injected ToolMessages are not part of the stream, so show them here.
                for message in reject_tool_calls(graph, config, reason):
                    print_event(message_to_event("tools", message), out, st)
                out(st.red("Rejected. The agent will be told."))
                events = drive(graph, config, None)
                break
            out("Please answer y or n.")


def run_cli(argv: list[str] | None = None, input_fn: Callable[[str], str] = input,
            output_fn: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="LangGraph coding agent (CLI)")
    parser.add_argument("--mode", choices=MODES, default=default_mode(),
                        help="pipeline (default) or react")
    args = parser.parse_args(argv)
    st = Style(sys.stdout.isatty())

    try:
        graph = get_graph(args.mode)
    except Exception as exc:  # missing API key, missing provider package, bad MODEL_NAME...
        output_fn(f"Could not start the agent: {type(exc).__name__}: {exc}")
        output_fn("Check LLM_PROVIDER / MODEL_NAME / API key in your .env file.")
        return 2

    thread_id = str(uuid.uuid4())
    config = make_config(thread_id)
    docker = "available" if sandbox.docker_available() else "NOT available"
    output_fn(st.bold("LangGraph coding agent"))
    output_fn(f"  model:      {describe_llm()}")
    output_fn(f"  mode:       {args.mode}")
    output_fn(f"  approval:   {approval_mode()} (dangerous tools need your approval)")
    workspace = agent_tools.WORKSPACE
    output_fn(f"  workspace:  {workspace}")
    output_fn(f"  docker:     {docker}, image {sandbox.get_sandbox_image(workspace)}")
    output_fn("Type /help for commands.\n")

    while True:
        try:
            line = input_fn(st.bold("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/help":
            output_fn(HELP)
        elif line == "/new":
            thread_id = str(uuid.uuid4())
            config = make_config(thread_id)
            output_fn("Started a new conversation.")
        elif line == "/state":
            output_fn(read_agent_state.invoke({}))
        elif line == "/status":
            output_fn(str(get_run_status(graph, config)))
        else:
            try:
                run_turn(graph, config, line, input_fn, output_fn, st)
            except KeyboardInterrupt:
                output_fn(st.red("\nInterrupted. Use /new to start fresh if the state looks odd."))
            except Exception as exc:
                output_fn(st.red(f"Error: {type(exc).__name__}: {exc}"))
                output_fn("Use /new to start a new conversation if this keeps happening.")
            output_fn("")


if __name__ == "__main__":
    raise SystemExit(run_cli())
