"""LangGraph graphs and the shared run/approval driver.

Two modes share one tool set and one approval policy:

``react``     create_react_agent. The graph pauses before EVERY tool batch
              (interrupt_before=["tools"]); the driver auto-resumes batches that contain only
              safe tools, so humans only see dangerous ones.

``pipeline``  (default) A StateGraph with four hard-enforced nodes:

                  planner -> coder -> reviewer -> committer

              Enforcement is structural, not prompt-based:
              * each node is bound to its own tool set (STAGE_TOOLS) and the shared tools node
                re-checks the allow-list, so e.g. git_commit outside the committer is refused;
              * an `advance` gate decides every transition from tool results (plan+todo written,
                files changed, review_changes verdict, git_commit succeeded);
              * a conditional edge sends only dangerous batches through the `approval` node,
                the single place the graph interrupts. Safe batches never pause.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from agent import tools as tools_module
from agent.llm import content_to_text, get_chat_model
from agent.prompts import HANDOFF, STAGE_PROMPTS, SYSTEM_PROMPT, render_prompt
from agent.tools import (
    STAGE_TOOLS,
    STAGES,
    TOOLS,
    TOOLS_BY_NAME,
    is_error_text,
    parse_verdict,
    requires_approval,
)

MODES = ("pipeline", "react")
MAX_NUDGES = 3  # how often a stage may be reminded of missing required steps


def default_mode() -> str:
    mode = os.getenv("AGENT_MODE", "pipeline").strip().lower()
    return mode if mode in MODES else "pipeline"


def max_review_rounds() -> int:
    try:
        return max(0, int(os.getenv("MAX_REVIEW_ROUNDS", "3")))
    except ValueError:
        return 3


# One checkpointer for the whole process; threads are separated by thread_id.
_checkpointer = MemorySaver()
_graphs: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# react mode
# ---------------------------------------------------------------------------
def _build_react_graph(model, checkpointer):
    prompt = render_prompt(SYSTEM_PROMPT, tools_module.WORKSPACE)
    common = dict(checkpointer=checkpointer, interrupt_before=["tools"])
    # LangGraph renamed the system-prompt argument across versions: try prompt=, then
    # state_modifier=, then give up on attaching a prompt.
    for keyword in ("prompt", "state_modifier"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return create_react_agent(model, tools=TOOLS, **{keyword: prompt}, **common)
        except TypeError:
            continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_react_agent(model, tools=TOOLS, **common)


# ---------------------------------------------------------------------------
# pipeline mode
# ---------------------------------------------------------------------------
class PipelineState(MessagesState):
    stage: str        # planner | coder | reviewer | committer | done
    stage_start: int  # index into messages where the current stage began
    revisions: int    # how many times the reviewer sent work back to the coder
    nudges: int       # reminders issued in the current stage
    status: str       # "" while running; committed | no_changes | stalled | review_failed
    reviewed_tree: str  # fingerprint of the exact tree the reviewer approved ("" before approval)


def _window(state: PipelineState) -> list:
    return state["messages"][state["stage_start"]:]


def _succeeded(window: list) -> set[str]:
    """Names of tools whose call succeeded within this stage window."""
    return {m.name for m in window
            if isinstance(m, ToolMessage) and getattr(m, "status", "success") != "error"}


def run_tool_call(call: dict) -> ToolMessage:
    """Execute one tool call and wrap the result (or error) in a ToolMessage."""
    name, call_id = call["name"], call["id"]
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return ToolMessage(content=f"ERROR: unknown tool '{name}'", tool_call_id=call_id,
                           name=name, status="error")
    try:
        result = tool.invoke(call.get("args") or {})
    except Exception as exc:  # bad arguments, tool crash, ...
        result = f"ERROR: {type(exc).__name__}: {exc}"
    text = content_to_text(result) if not isinstance(result, str) else result
    return ToolMessage(content=text or "(no output)", tool_call_id=call_id, name=name,
                       status="error" if is_error_text(text) else "success")


def _handoff(next_stage: str) -> HumanMessage:
    # A user-role message also keeps the transcript valid for providers that reject a
    # conversation ending on an assistant message.
    return HumanMessage(content=f"[pipeline] {HANDOFF[next_stage]}")


def _go(state: PipelineState, next_stage: str, extra: str = "", **updates) -> dict:
    text = HANDOFF[next_stage] + (("\n\n" + extra) if extra else "")
    return {"stage": next_stage, "stage_start": len(state["messages"]) + 1, "nudges": 0,
            "messages": [HumanMessage(content=f"[pipeline] {text}")], **updates}


def _finish(status: str, note: str = "") -> dict:
    update: dict = {"stage": "done", "status": status}
    if note:
        update["messages"] = [AIMessage(content=f"[pipeline] {note}")]
    return update


def _nudge(state: PipelineState, text: str) -> dict:
    nudges = state["nudges"] + 1
    if nudges > MAX_NUDGES:
        return _finish("stalled", f"Stopped: the {state['stage']} stage did not complete its "
                                  f"required steps. {text}")
    return {"nudges": nudges, "messages": [HumanMessage(content=f"[pipeline] {text}")]}


def intake(state: PipelineState) -> dict:
    """Entry node: every new user message restarts the pipeline at the planner."""
    return {"stage": "planner", "stage_start": max(len(state["messages"]) - 1, 0),
            "revisions": 0, "nudges": 0, "status": "", "reviewed_tree": ""}


def advance(state: PipelineState) -> dict:
    """The gate. Decides, from tool evidence only, whether the current stage may hand over."""
    stage = state["stage"]
    window = _window(state)
    done = _succeeded(window)

    if stage == "planner":
        missing = {"update_plan", "update_todo"} - done
        if not missing:
            return _go(state, "coder")
        return _nudge(state, "Before the pipeline can continue you must call "
                             + " and ".join(sorted(missing)) + ".")

    if stage == "coder":
        if done & {"write_file", "edit_file"}:
            return _go(state, "reviewer")
        if state["revisions"] == 0:
            # Nothing was changed (a question, or the coder could not proceed): nothing to
            # review or commit, so the run ends here and the coder's answer is the result.
            return _finish("no_changes")
        return _nudge(state, "The reviewer requested changes. Make real edits with "
                             "write_file/edit_file to address the feedback.")

    if stage == "reviewer":
        reviews = [m for m in window if isinstance(m, ToolMessage)
                   and m.name == "review_changes" and m.status != "error"]
        if not reviews:
            return _nudge(state, "You must call review_changes (call git_init first if the "
                                 "workspace is not a git repository). Its VERDICT decides the "
                                 "outcome.")
        review_text = str(reviews[-1].content)
        verdict = parse_verdict(review_text)
        if verdict == "APPROVE":
            reviewed = tools_module.parse_reviewed_tree(review_text)
            if not reviewed:
                return _nudge(state, "The review did not report which tree it covered. "
                                     "Call review_changes again.")
            if tools_module.tree_fingerprint() != reviewed:
                # e.g. run_tests wrote files after the review: the approval no longer applies.
                return _nudge(state, "Files changed after the review (for example tests wrote "
                                     "files), so the approval does not cover them. Call "
                                     "review_changes again.")
            return _go(state, "committer", reviewed_tree=reviewed)
        if verdict == "NO_CHANGES":
            return _finish("no_changes", "The reviewer found no changes to commit.")
        if state["revisions"] >= max_review_rounds():
            return _finish("review_failed",
                           f"Review still requests changes after {state['revisions']} "
                           f"revision round(s); nothing was committed.\n\n{review_text[:3000]}")
        return _go(state, "coder",
                   "The reviewer requested changes. Address every point:\n\n" + review_text[:6000],
                   revisions=state["revisions"] + 1)

    if stage == "committer":
        if "git_commit" in done:
            return _finish("committed")
        return _nudge(state, "Stage the changes with git_add and call git_commit with a clear "
                             "message. The pipeline only finishes once git_commit succeeds.")

    return _finish(state.get("status") or "stalled")


def _allowed_calls(state: PipelineState, calls: list[dict]) -> list[dict]:
    allowed = set(STAGE_TOOLS.get(state["stage"], ()))
    return [c for c in calls if c["name"] in allowed]


def route_after_model(state: PipelineState) -> str:
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "advance"
    # Only calls that will actually run can require approval.
    if any(requires_approval(c["name"]) for c in _allowed_calls(state, calls)):
        return "approval"
    return "tools"


def tools_node(state: PipelineState) -> dict:
    """Runs the tool calls of the last AI message, enforcing the stage allow-list."""
    stage = state["stage"]
    allowed = set(STAGE_TOOLS.get(stage, ()))
    results: list[ToolMessage] = []
    for call in state["messages"][-1].tool_calls:
        if call["name"] not in allowed:
            results.append(_error_message(
                call, f"tool '{call['name']}' is not permitted in the {stage} stage. "
                      f"Permitted: {', '.join(sorted(allowed))}."))
        elif stage == "committer" and call["name"] in COMMIT_GUARDED_TOOLS and (
                tools_module.tree_fingerprint() != state.get("reviewed_tree")
                or not state.get("reviewed_tree")):
            # Hard guarantee: only the exact tree the reviewer approved can be staged/committed.
            results.append(_error_message(
                call, "the workspace no longer matches the tree the reviewer approved "
                      "(or nothing was approved). Nothing was staged or committed."))
        else:
            results.append(run_tool_call(call))
    return {"messages": results}


# Tools that would move unreviewed content into history.
COMMIT_GUARDED_TOOLS = frozenset({"git_add", "git_commit"})


def _error_message(call: dict, text: str) -> ToolMessage:
    return ToolMessage(content="ERROR: " + text, tool_call_id=call["id"], name=call["name"],
                       status="error")


def _build_pipeline_graph(model, checkpointer):
    prompts = {s: render_prompt(STAGE_PROMPTS[s], tools_module.WORKSPACE) for s in STAGES}
    bound = {s: model.bind_tools([TOOLS_BY_NAME[n] for n in STAGE_TOOLS[s]]) for s in STAGES}

    def make_stage_node(stage: str):
        def run(state: PipelineState) -> dict:
            reply = bound[stage].invoke([SystemMessage(content=prompts[stage]),
                                         *state["messages"]])
            return {"messages": [reply]}
        run.__name__ = stage
        return run

    graph = StateGraph(PipelineState)
    graph.add_node("intake", intake)
    for stage in STAGES:
        graph.add_node(stage, make_stage_node(stage))
    graph.add_node("approval", lambda state: {})  # pure interrupt point, does no work
    graph.add_node("tools", tools_node)
    graph.add_node("advance", advance)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "planner")
    for stage in STAGES:
        graph.add_conditional_edges(stage, route_after_model,
                                    {"approval": "approval", "tools": "tools", "advance": "advance"})
    graph.add_edge("approval", "tools")
    graph.add_conditional_edges("tools", lambda state: state["stage"], {s: s for s in STAGES})
    graph.add_conditional_edges("advance", lambda state: state["stage"],
                                {**{s: s for s in STAGES}, "done": END})
    return graph.compile(checkpointer=checkpointer, interrupt_before=["approval"])


# ---------------------------------------------------------------------------
# public builders
# ---------------------------------------------------------------------------
def build_graph(mode: str | None = None, model=None, checkpointer=None):
    """Build (uncached) a graph. Tests pass a fake `model` and their own `checkpointer`."""
    mode = mode or default_mode()
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {MODES}")
    model = model if model is not None else get_chat_model()
    checkpointer = checkpointer if checkpointer is not None else _checkpointer
    if mode == "react":
        return _build_react_graph(model, checkpointer)
    return _build_pipeline_graph(model, checkpointer)


def get_graph(mode: str | None = None):
    """Return the cached compiled graph for `mode` (building it on first use)."""
    mode = mode or default_mode()
    if mode not in _graphs:
        _graphs[mode] = build_graph(mode)
    return _graphs[mode]


# ---------------------------------------------------------------------------
# shared driver used by both the CLI and the Streamlit app
# ---------------------------------------------------------------------------
def make_config(thread_id: str) -> dict:
    try:
        limit = int(os.getenv("RECURSION_LIMIT", "150"))
    except ValueError:
        limit = 150
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": limit}


def _call_info(call: dict) -> dict:
    return {"id": call.get("id"), "name": call["name"], "args": call.get("args") or {},
            "dangerous": requires_approval(call["name"])}


def pending_tool_calls(graph, config) -> list[dict]:
    """Tool calls the graph is paused on (empty if it is not waiting at a tool boundary)."""
    snapshot = graph.get_state(config)
    if not snapshot.next or snapshot.next[0] not in ("tools", "approval"):
        return []
    messages = (snapshot.values or {}).get("messages") or []
    if not messages:
        return []
    return [_call_info(c) for c in (getattr(messages[-1], "tool_calls", None) or [])]


def needs_human(calls: list[dict]) -> bool:
    return any(requires_approval(c["name"]) for c in calls)


def reject_tool_calls(graph, config, reason: str = "") -> list[ToolMessage]:
    """Reject the pending tool calls.

    Injects a ToolMessage per call and updates the state *as the tools node*, so the graph
    continues exactly as if the tools had run; the next `drive` lets the model answer the rejection.
    """
    calls = pending_tool_calls(graph, config)
    if not calls:
        return []
    note = f" Reason: {reason}" if reason else ""
    messages = [
        ToolMessage(
            content=(f"REJECTED by the user: the call to '{c['name']}' was NOT executed.{note} "
                     f"Do not retry it unchanged; choose a different approach or ask the user."),
            tool_call_id=c["id"], name=c["name"], status="error")
        for c in calls
    ]
    graph.update_state(config, {"messages": messages}, as_node="tools")
    return messages


def message_to_event(node: str, message) -> dict:
    """Convert a LangChain message into a small UI-friendly dict."""
    if isinstance(message, AIMessage):
        return {"kind": "ai", "node": node, "content": content_to_text(message.content),
                "tool_calls": [_call_info(c) for c in (message.tool_calls or [])]}
    if isinstance(message, ToolMessage):
        return {"kind": "tool", "node": node, "name": message.name or "tool",
                "content": content_to_text(message.content),
                "error": getattr(message, "status", "success") == "error",
                "tool_call_id": message.tool_call_id}
    return {"kind": "pipeline", "node": node, "content": content_to_text(message.content)}


def _events_from_chunk(chunk: dict) -> Iterator[dict]:
    for node, update in chunk.items():
        if node.startswith("__") or not isinstance(update, dict):
            continue
        messages = update.get("messages") or []
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            yield message_to_event(node, message)


def get_run_status(graph, config) -> dict:
    """Pipeline progress for the UI (react mode returns stage=None)."""
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    return {"stage": values.get("stage"), "status": values.get("status", ""),
            "revisions": values.get("revisions", 0), "waiting": bool(snapshot.next)}


def drive(graph, config, user_text: str | None = None) -> Iterator[dict]:
    """Run the graph and yield events until it finishes or needs a human decision.

    * user_text given  -> start a new turn with that message.
    * user_text None   -> resume (after Approve, or after reject_tool_calls).
    Batches containing only safe tools are resumed automatically.
    After the generator ends, call pending_tool_calls() to see whether approval is needed.
    """
    graph_input: Any = {"messages": [HumanMessage(content=user_text)]} if user_text is not None else None
    while True:
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            yield from _events_from_chunk(chunk)
        pending = pending_tool_calls(graph, config)
        if pending and not needs_human(pending):
            graph_input = None  # safe batch: approve automatically and continue
            continue
        return
