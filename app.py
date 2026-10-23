"""Streamlit front end for the LangGraph coding agent.

Run with:  streamlit run app.py

Chat first: ask questions, brainstorm, or ask for a build ("auto" mode). Dangerous tool calls
(write/edit/run/install/git commit) pause the graph and show Approve / Reject buttons; safe
read-only tools run automatically. Conversations are saved and can be reopened from the sidebar.
"""

from __future__ import annotations

import streamlit as st

from agent import history, sandbox
from agent import tools as agent_tools
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
    pending_tool_calls,
    reject_tool_calls,
)
from agent.llm import describe_llm
from agent.tools import STAGES, STEP_TO_STAGE, approval_mode, workspace_top_level

st.set_page_config(page_title="LangGraph Coding Agent", page_icon="🛠️", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("thread_id", history.new_thread_id())  # LangGraph thread for this conversation
    ss.setdefault("messages", [])                  # display log: user messages + graph events
    ss.setdefault("mode", default_mode())
    ss.setdefault("error", None)


def new_conversation() -> None:
    ss = st.session_state
    ss.thread_id = history.new_thread_id()
    ss.messages = []
    ss.error = None


def on_mode_change() -> None:
    """The conversation continues in the new mode (messages are shared by all modes)."""
    ss = st.session_state
    ss.mode = ss.mode_choice
    if history.get_conversation(ss.thread_id):
        history.record(ss.thread_id, ss.mode, str(agent_tools.WORKSPACE))


def open_conversation(row: dict) -> None:
    """Switch to a saved conversation: restore its mode and rebuild the visible history.

    Called at the top of a script run (before any widget is created), so it may set the
    selectbox's state.
    """
    ss = st.session_state
    mode = row["mode"] if row["mode"] in MODES else ss.mode
    ss.thread_id, ss.mode, ss.mode_choice = row["thread_id"], mode, mode
    ss.error = None
    try:
        graph = load_graph(mode)
        ss.messages = history_events(graph, make_config(row["thread_id"], mode))
    except Exception as exc:  # keep the app usable; the main flow reports startup errors
        ss.messages = []
        ss.error = f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def load_graph(mode: str):
    return get_graph(mode)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_args(args: dict) -> None:
    for key, value in args.items():
        if isinstance(value, str) and (len(value) > 80 or "\n" in value):
            st.caption(key)
            st.code(value, language=None)
        else:
            st.markdown(f"**{key}**: `{value}`")


def render_call(call: dict, expanded: bool = False) -> None:
    badge = "🔴 needs approval" if call["dangerous"] else "🟢 auto-approved"
    with st.expander(f"🔧 {call['name']} · {badge}", expanded=expanded):
        render_args(call["args"])


def render_event(event: dict) -> None:
    kind = event["kind"]
    if kind == "user":
        with st.chat_message("user"):
            st.markdown(event["content"])
    elif kind == "ai":
        with st.chat_message("assistant"):
            if event["node"] not in ("agent", "chat"):  # pipeline stage name
                st.caption(f"stage: **{event['node']}**")
            if event["content"].strip():
                st.markdown(event["content"])
            for call in event["tool_calls"]:
                render_call(call)
    elif kind == "tool":
        with st.chat_message("assistant", avatar="🧰"):
            label = f"{'❌' if event['error'] else '✅'} result of {event['name']}"
            with st.expander(label, expanded=False):
                st.code(event["content"], language=None)
    else:  # pipeline notes (hand-overs, reminders)
        st.caption("» " + event["content"].replace("\n", " ")[:400])


def render_sidebar(status: dict | None, busy: bool = False) -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("Coding agent")
        st.markdown(f"**Model:** `{describe_llm()}`")
        st.selectbox("Agent mode", MODES, index=MODES.index(ss.mode), key="mode_choice",
                     on_change=on_mode_change, disabled=busy,
                     help="\n\n".join(f"{m}: {MODE_HELP[m]}" for m in MODES))
        if busy:
            st.caption("Answer the pending approval before changing the mode.")
        st.caption(f"Approval: `{approval_mode()}` — "
                   + ("every tool needs approval." if approval_mode() == "all"
                      else "only dangerous tools need approval."))

        st.subheader("Sandbox")
        if sandbox.docker_available():
            st.success("Docker available")
        elif sandbox.unsafe_local_allowed():
            st.warning("Docker not available — UNSAFE local execution enabled")
        else:
            st.error("Docker not available — code will not run")
        st.markdown(f"Image: `{sandbox.get_sandbox_image(agent_tools.WORKSPACE)}`")

        if status and status.get("stage") is not None:
            st.subheader("Pipeline" if ss.mode == "pipeline" else "Build")
            stage, final = status["stage"], status["status"]
            chosen = [STEP_TO_STAGE[x] if x in STEP_TO_STAGE else x for x in status.get("steps") or []]
            names = [n for n in STAGES if n in chosen] if chosen and ss.mode == "auto" else list(STAGES)
            parts = []
            if ss.mode == "auto":
                parts.append("▶️ chat" if stage == "chat" else "💬 chat")
            if stage == "chat" and not final:
                names = []  # nothing has been built in this conversation yet
            for index, name in enumerate(names):
                if stage in ("done", "chat"):
                    icon = "✅" if final in ("committed", "reviewed", "coded") else "⏹️"
                elif name == stage:
                    icon = "▶️"
                else:
                    icon = "✅" if index < names.index(stage) else "⬜"
                parts.append(f"{icon} {name}")
            st.markdown("  \n".join(parts))
            if final and stage in ("done", "chat"):
                st.caption(f"Last build: {final}")
            if status["revisions"]:
                st.caption(f"Review rounds: {status['revisions']}")

        st.subheader("Conversations")
        rows = history.list_conversations(str(agent_tools.WORKSPACE), limit=12)
        if not history.persistent():
            st.caption("History is off (CONVERSATION_DB=memory); chats last until the app stops.")
        if rows and not ss.messages and not busy:
            if st.button("▶ Continue last conversation", use_container_width=True, key="continue_last"):
                ss.open_request = rows[0]["thread_id"]
                st.rerun()
        if not rows:
            st.caption("No saved conversations yet.")
        for number, row in enumerate(rows, 1):
            current = row["thread_id"] == ss.thread_id
            label = f"{'● ' if current else ''}{row['title']} · {row['mode']} · {history.format_age(row['updated_at'])}"
            open_col, delete_col = st.columns([5, 1])
            if open_col.button(label, key=f"open_{row['thread_id']}", disabled=current or busy,
                               use_container_width=True):
                ss.open_request = row["thread_id"]
                st.rerun()
            if delete_col.button(f"🗑 {number}", key=f"delete_{row['thread_id']}", disabled=busy,
                                 help="Delete this conversation"):
                ss.delete_request = row["thread_id"]
                st.rerun()

        st.subheader("Workspace")
        st.caption(str(agent_tools.WORKSPACE))
        entries = workspace_top_level()
        if entries:
            for name, is_dir in entries:
                st.text(("📁 " if is_dir else "📄 ") + name)
        else:
            st.caption("(empty)")

        st.divider()
        if st.button("New conversation", use_container_width=True):
            new_conversation()
            st.rerun()


def run_and_render(graph, config, user_text: str | None = None) -> None:
    """Drive the graph, show events as they arrive, then rerun to show approval buttons."""
    ss = st.session_state
    ss.error = None
    try:
        with st.spinner("Agent is working..."):
            for event in drive(graph, config, user_text):
                ss.messages.append(event)
                render_event(event)
    except Exception as exc:  # API errors, rate limits, recursion limit, ...
        ss.error = f"{type(exc).__name__}: {exc}"
    st.rerun()


FINAL_NOTES = {
    "reviewed": ("success", "Pipeline finished: changes were reviewed (not committed)."),
    "coded": ("info", "Pipeline finished: changes were made but not reviewed or committed."),
    "committed": ("success", "Pipeline finished: changes were reviewed and committed."),
    "no_changes": ("info", "Pipeline finished: no file changes were made, so nothing was reviewed or committed."),
    "stalled": ("warning", "Pipeline stopped: a stage did not complete its required steps."),
    "review_failed": ("warning", "Pipeline stopped: the reviewer still requested changes after the allowed rounds."),
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
init_state()
ss = st.session_state
st.title("🛠️ LangGraph Coding Agent")

# Sidebar buttons only record a request; it is handled here, before any widget exists.
if ss.get("delete_request"):
    doomed, ss.delete_request = ss.delete_request, None
    history.delete_conversation(doomed)
    if doomed == ss.thread_id:
        new_conversation()
if ss.get("open_request"):
    wanted, ss.open_request = ss.open_request, None
    row = history.get_conversation(wanted)
    if row:
        open_conversation(row)

graph = None
try:
    graph = load_graph(ss.mode)
except Exception as exc:  # e.g. missing API key or provider package
    config = None
    render_sidebar(None)
    st.error(f"Could not start the agent: {type(exc).__name__}: {exc}")
    st.info("Check LLM_PROVIDER, MODEL_NAME and your API key in the .env file, "
            "then restart the app.")
    st.stop()

config = make_config(ss.thread_id, ss.mode)
status = get_run_status(graph, config)
pending = pending_tool_calls(graph, config)
render_sidebar(status, busy=bool(pending))

for event in ss.messages:
    render_event(event)

if ss.error:
    st.error(ss.error)

note = FINAL_NOTES.get(status["status"]) if ss.mode == "pipeline" else None
if note:
    getattr(st, note[0])(note[1])

# Human approval: shown whenever the graph is paused before executing tools.
if pending:
    st.warning("The agent wants to run the tool call(s) below. Nothing has been executed yet.")
    for call in pending:
        render_call(call, expanded=True)
    reason = st.text_input("Rejection reason (optional)", key="reject_reason")
    approve_col, reject_col, _ = st.columns([1, 1, 4])
    approve = approve_col.button("✅ Approve", type="primary", key="approve")
    reject = reject_col.button("⛔ Reject", key="reject")
    if approve:
        run_and_render(graph, config, None)  # resume the graph
    elif reject:
        # Inject ToolMessages as the tools node, then resume so the model can respond.
        for message in reject_tool_calls(graph, config, reason):
            ss.messages.append(message_to_event("tools", message))
        run_and_render(graph, config, None)

prompt = st.chat_input("Chat, brainstorm, or ask the agent to build something...",
                       disabled=bool(pending))
if prompt:
    user_event = {"kind": "user", "content": prompt}
    ss.messages.append(user_event)
    render_event(user_event)
    run_and_render(graph, config, prompt)
