"""Streamlit front end for the LangGraph coding agent.

Run with:  streamlit run app.py

Dangerous tool calls (write/edit/run/install/git commit) pause the graph and show
Approve / Reject buttons. Safe read-only tools run automatically.
"""

from __future__ import annotations

import uuid

import streamlit as st

from agent import sandbox
from agent import tools as agent_tools
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
from agent.llm import describe_llm
from agent.tools import STAGES, approval_mode, workspace_top_level

st.set_page_config(page_title="LangGraph Coding Agent", page_icon="🛠️", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("thread_id", str(uuid.uuid4()))  # LangGraph thread for this conversation
    ss.setdefault("messages", [])                  # display log: user messages + graph events
    ss.setdefault("mode", default_mode())
    ss.setdefault("error", None)


def new_conversation() -> None:
    ss = st.session_state
    ss.thread_id = str(uuid.uuid4())
    ss.messages = []
    ss.error = None


def on_mode_change() -> None:
    st.session_state.mode = st.session_state.mode_choice
    new_conversation()  # a thread belongs to exactly one graph, so start fresh


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
            if event["node"] != "agent":  # pipeline stage name
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


def render_sidebar(status: dict | None) -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("Coding agent")
        st.markdown(f"**Model:** `{describe_llm()}`")
        st.selectbox("Agent mode", MODES, index=MODES.index(ss.mode), key="mode_choice",
                     on_change=on_mode_change,
                     help="pipeline: planner → coder → reviewer → committer (hard-enforced). "
                          "react: single create_react_agent loop.")
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
            st.subheader("Pipeline")
            stage, final = status["stage"], status["status"]
            parts = []
            for index, name in enumerate(STAGES):
                if stage == "done":
                    icon = "✅" if final == "committed" else "⏹️"
                elif name == stage:
                    icon = "▶️"
                else:
                    icon = "✅" if index < STAGES.index(stage) else "⬜"
                parts.append(f"{icon} {name}")
            st.markdown("  \n".join(parts))
            if status["revisions"]:
                st.caption(f"Review rounds: {status['revisions']}")

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

config = make_config(ss.thread_id)
status = get_run_status(graph, config)
render_sidebar(status)

for event in ss.messages:
    render_event(event)

if ss.error:
    st.error(ss.error)

note = FINAL_NOTES.get(status["status"]) if ss.mode == "pipeline" else None
if note:
    getattr(st, note[0])(note[1])

# Human approval: shown whenever the graph is paused before executing tools.
pending = pending_tool_calls(graph, config)
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

prompt = st.chat_input("Describe what you want the agent to build or change...",
                       disabled=bool(pending))
if prompt:
    user_event = {"kind": "user", "content": prompt}
    ss.messages.append(user_event)
    render_event(user_event)
    run_and_render(graph, config, prompt)
