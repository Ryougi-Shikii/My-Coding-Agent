"""Command-line front end for the coding agent.

    python cli.py                   # auto mode: chat first, build only when asked
    python cli.py --continue        # pick up the most recent conversation in this workspace
    python cli.py --resume 2        # ...or a specific one (number from --list, or an id prefix)
    python cli.py --list            # list saved conversations and exit
    python cli.py --mode react      # classic single agent;  --mode pipeline = strict fixed pipeline

Safe tools run automatically; dangerous ones (write/edit/run/install/git_commit) pause and ask
for your approval. Conversations are saved, so they survive closing the terminal.
Type /help inside the prompt for commands.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from agent import history, sandbox
from agent.graph import (
    MODE_HELP,
    MODES,
    default_mode,
    drive,
    get_graph,
    get_run_status,
    history_events,
    make_config,
    message_to_event,
    needs_human,
    pending_tool_calls,
    reject_tool_calls,
)
from agent import tools as agent_tools
from agent.llm import describe_llm
from agent.tools import approval_mode, read_agent_state

HELP = """\
Commands:
  /help           show this help
  /state          show the saved plan, todo list and memory
  /status         show the current stage / build status of this conversation
  /new            start a new conversation
  /list           list saved conversations of this workspace
  /resume [n|id]  continue a saved conversation (no argument: choose from a list)
  /delete n|id    delete a saved conversation
  /mode [name]    show or change the mode: auto (chat first), react, pipeline
  /exit           quit
Anything else is sent to the agent. Just chat: ask questions, brainstorm, or ask it to build."""

RECAP_EVENTS = 8      # how much of a resumed conversation is replayed
RECAP_CHARS = 400
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

    text=None only deals with whatever is already pending (a resumed conversation).
    Dangerous tool batches trigger an approval prompt; safe ones are handled by `drive`.
    """
    if text is not None:
        events = drive(graph, config, text)
    else:
        pending = pending_tool_calls(graph, config)
        events = drive(graph, config, None) if pending and not needs_human(pending) else []
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


def conversation_lines(rows: list[dict], current: str | None = None) -> list[str]:
    lines = []
    for number, row in enumerate(rows, 1):
        mark = "*" if row["thread_id"] == current else " "
        lines.append(f"{mark}{number:>2}. {row['title']}  ({row['mode']}, "
                     f"{history.format_age(row['updated_at'])}, {row['thread_id'][:8]})")
    return lines


def recap(graph, config, out: Callable[[str], None], st: Style) -> None:
    """Replay the end of a stored conversation so you remember where you were."""
    events = [e for e in history_events(graph, config) if e["kind"] != "tool" or e["error"]]
    shown = events[-RECAP_EVENTS:]
    if len(events) > len(shown):
        out(st.dim(f"  ... {len(events) - len(shown)} earlier messages"))
    for event in shown:
        if event["kind"] == "user":
            out(st.bold("you> ") + _clip(event["content"], RECAP_CHARS))
        else:
            print_event({**event, "content": _clip(event["content"], RECAP_CHARS)}, out, st)


class Session:
    """The conversation currently open in the terminal."""

    def __init__(self, mode: str, thread_id: str | None = None):
        self.workspace = str(agent_tools.WORKSPACE)
        self.mode = mode
        self.thread_id = thread_id or history.new_thread_id()
        self.graph = get_graph(mode)
        self.config = make_config(self.thread_id, mode)

    def open(self, thread_id: str, mode: str) -> None:
        self.mode, self.thread_id = mode, thread_id
        self.graph = get_graph(mode)
        self.config = make_config(thread_id, mode)

    def pending(self) -> bool:
        return bool(pending_tool_calls(self.graph, self.config))


def run_cli(argv: list[str] | None = None, input_fn: Callable[[str], str] = input,
            output_fn: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="LangGraph coding agent (CLI)")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="auto (default): chat first, build when asked; react; pipeline (strict)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--continue", "-c", dest="cont", action="store_true",
                       help="continue the most recent conversation of this workspace")
    group.add_argument("--resume", "-r", metavar="N|ID", help="resume a saved conversation")
    group.add_argument("--list", "-l", action="store_true", help="list saved conversations and exit")
    args = parser.parse_args(argv)
    st = Style(sys.stdout.isatty())
    workspace = str(agent_tools.WORKSPACE)

    if args.list:
        rows = history.list_conversations(workspace)
        output_fn("\n".join(conversation_lines(rows)) if rows else "No saved conversations yet.")
        return 0

    target = None
    if args.cont:
        target = history.latest_conversation(workspace)
        if target is None:
            output_fn("No saved conversation to continue; starting a new one.")
    elif args.resume:
        target = history.find_conversation(workspace, args.resume)
        if target is None:
            output_fn(f"No conversation matches '{args.resume}'. Use --list to see them.")
            return 2

    mode = args.mode or (target["mode"] if target else None) or default_mode()
    try:
        session = Session(mode, target["thread_id"] if target else None)
    except Exception as exc:  # missing API key, missing provider package, bad MODEL_NAME...
        output_fn(f"Could not start the agent: {type(exc).__name__}: {exc}")
        output_fn("Check LLM_PROVIDER / MODEL_NAME / API key in your .env file.")
        return 2

    docker = "available" if sandbox.docker_available() else "NOT available"
    output_fn(st.bold("LangGraph coding agent"))
    output_fn(f"  model:      {describe_llm()}")
    output_fn(f"  mode:       {session.mode} - {MODE_HELP[session.mode]}")
    output_fn(f"  approval:   {approval_mode()} (dangerous tools need your approval)")
    output_fn(f"  workspace:  {workspace}")
    output_fn(f"  docker:     {docker}, image {sandbox.get_sandbox_image(agent_tools.WORKSPACE)}")
    if not history.persistent():
        output_fn("  history:    NOT saved (CONVERSATION_DB=memory)")
    output_fn("Type /help for commands.\n")

    def attempt(text: str | None) -> None:
        try:
            run_turn(session.graph, session.config, text, input_fn, output_fn, st)
        except KeyboardInterrupt:
            output_fn(st.red("\nInterrupted. Use /new to start fresh if the state looks odd."))
        except Exception as exc:
            output_fn(st.red(f"Error: {type(exc).__name__}: {exc}"))
            output_fn("Use /new to start a new conversation if this keeps happening.")

    def switch_to(row: dict) -> None:
        # keep the conversation's own mode, unless the user forced one with --mode before
        session.open(row["thread_id"], row["mode"] if row["mode"] in MODES else session.mode)
        output_fn(f"Resumed: {row['title']}  ({session.mode})")
        recap(session.graph, session.config, output_fn, st)
        if session.pending():
            output_fn(st.yellow("This conversation was waiting for your approval:"))
            attempt(None)
        output_fn("")

    if target:
        switch_to({**target, "mode": mode})

    while True:
        try:
            line = input_fn(st.bold("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0
        if not line:
            continue
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        if line in ("/exit", "/quit"):
            return 0
        if command == "/help":
            output_fn(HELP)
        elif command == "/new":
            session.open(history.new_thread_id(), session.mode)
            output_fn("Started a new conversation.")
        elif command == "/state":
            output_fn(read_agent_state.invoke({}))
        elif command == "/status":
            output_fn(str(get_run_status(session.graph, session.config)))
        elif command == "/list":
            rows = history.list_conversations(workspace)
            output_fn("\n".join(conversation_lines(rows, session.thread_id)) if rows
                      else "No saved conversations yet.")
        elif command == "/resume":
            rows = history.list_conversations(workspace)
            if not argument:
                if not rows:
                    output_fn("No saved conversations yet.")
                    continue
                output_fn("\n".join(conversation_lines(rows, session.thread_id)))
                try:
                    argument = input_fn("Resume which one? (number, or Enter to cancel): ").strip()
                except (EOFError, KeyboardInterrupt):
                    output_fn("")
                    return 0
                if not argument:
                    continue
            row = history.find_conversation(workspace, argument)
            if row is None:
                output_fn(f"No conversation matches '{argument}'. Try /list.")
            else:
                switch_to(row)
        elif command == "/delete":
            row = history.find_conversation(workspace, argument) if argument else None
            if row is None:
                output_fn("Usage: /delete <number|id>   (see /list)")
            elif input_fn(f"Delete '{row['title']}'? [y/N]: ").strip().lower() not in ("y", "yes"):
                output_fn("Kept.")
            else:
                history.delete_conversation(row["thread_id"])
                output_fn("Deleted.")
                if row["thread_id"] == session.thread_id:
                    session.open(history.new_thread_id(), session.mode)
                    output_fn("Started a new conversation.")
        elif command == "/mode":
            if not argument:
                output_fn(f"Current mode: {session.mode} - {MODE_HELP[session.mode]}")
                output_fn("\n".join(f"  {m:9} {MODE_HELP[m]}" for m in MODES))
            elif argument not in MODES:
                output_fn(f"Unknown mode '{argument}'. Choose from: {', '.join(MODES)}")
            elif session.pending():
                output_fn("Answer the pending approval first (resume the conversation to see it).")
            else:
                session.open(session.thread_id, argument)
                if history.get_conversation(session.thread_id):  # already saved: remember the new mode
                    history.record(session.thread_id, argument, workspace)
                output_fn(f"Mode is now {argument}. The conversation continues.")
        elif line.startswith("/"):
            output_fn(f"Unknown command {command}. Type /help.")
        else:
            attempt(line)
            output_fn("")


if __name__ == "__main__":
    raise SystemExit(run_cli())
