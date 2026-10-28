"""LangGraph graphs and the shared run/approval driver.

Three modes share one tool set and one approval policy:

``auto``      (default) Chat first. A conversational node talks, discusses, brainstorms and can look
              at the project (read-only tools run without asking). Only when the user wants
              something built does it call ``start_build(task, steps)``, which hands over to the
              build pipeline below with just the steps that are needed ("code" for a tiny edit,
              "plan, code, review" for a feature, plus "commit" only on request). When the build
              ends, control returns to the chat node, which reports honestly what happened.

``pipeline``  Strict: every message runs planner -> coder -> reviewer -> committer.

``react``     create_react_agent. The graph pauses before EVERY tool batch
              (interrupt_before=["tools"]); the driver auto-resumes batches that contain only
              safe tools, so humans only see dangerous ones.

Build enforcement is structural, not prompt-based (``auto`` and ``pipeline``):
  * each node is bound to its own tool set (STAGE_TOOLS) and the shared tools node re-checks the
    allow-list, so e.g. git_commit outside the committer is refused;
  * the order is fixed (planner, coder, reviewer, committer); the chat node only chooses WHICH steps
    run, and a commit always requires a review;
  * an `advance` gate decides every transition from tool results (plan+todo written, files changed,
    review_changes verdict, git_commit succeeded), and only the exact tree the reviewer approved can
    be committed;
  * a conditional edge sends only dangerous batches through the `approval` node, the single place the
    graph interrupts. Safe batches never pause.

Conversations persist (see agent/history.py): the checkpointer is SQLite by default.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from agent import context as context_module
from agent import history, usage
from agent import tools as tools_module
from agent.llm import content_to_text, get_chat_model
from agent.prompts import (
    BUILD_START,
    BUILD_STATUS_DETAIL,
    CHAT_PROMPT,
    HANDOFF,
    STAGE_PROMPTS,
    SYSTEM_PROMPT,
    render_prompt,
)
from agent.tools import (
    CHAT_STAGE,
    STAGE_TOOLS,
    STAGES,
    TOOLS,
    TOOLS_BY_NAME,
    is_error_text,
    normalize_build_steps,
    parse_verdict,
    requires_approval,
)

MODES = ("auto", "react", "pipeline")
MAX_NUDGES = 3            # how often a stage may be reminded of missing required steps
MAX_BUILDS_PER_TURN = 3   # builds the chat node may start for one user message

MODE_HELP = {
    "auto": "chat first; starts the plan/code/review/commit pipeline only when you want something built",
    "react": "one agent that can do everything itself (each dangerous tool asks you)",
    "pipeline": "strict: every message runs planner -> coder -> reviewer -> committer",
}


def default_mode() -> str:
    mode = os.getenv("AGENT_MODE", "auto").strip().lower()
    return mode if mode in MODES else "auto"


def max_review_rounds() -> int:
    try:
        return max(0, int(os.getenv("MAX_REVIEW_ROUNDS", "3")))
    except ValueError:
        return 3


_graphs: dict[tuple[str, str], Any] = {}


# ---------------------------------------------------------------------------
# react mode
# ---------------------------------------------------------------------------
def _build_react_graph(model, checkpointer):
    def prompt(state):
        """System prompt for the CURRENT conversation's workspace + a size-capped history."""
        system = render_prompt(SYSTEM_PROMPT, tools_module.current_workspace())
        return context_module.react_messages(state["messages"], system)

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
# auto / pipeline modes
# ---------------------------------------------------------------------------
class PipelineState(MessagesState):
    stage: str          # chat | planner | coder | reviewer | committer | done
    steps: list         # stage names of the current build, in pipeline order
    after: str          # where control goes when a build ends: "chat" (auto) or "done" (pipeline)
    stage_start: int    # index into messages where the current stage began
    revisions: int      # how many times the reviewer sent work back to the coder
    nudges: int         # reminders issued in the current stage
    builds: int         # builds started for the current user message
    status: str         # "" while running; committed | reviewed | coded | no_changes |
                        # stalled | review_failed
    reviewed_tree: str  # fingerprint of the exact tree the reviewer approved ("" before approval)
    task: str           # the task of the current build (stages do not see each other's transcripts)
    halted: bool        # the token budget of this request was reached


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


def _error_message(call: dict, text: str) -> ToolMessage:
    return ToolMessage(content="ERROR: " + text, tool_call_id=call["id"], name=call["name"],
                       status="error")


def _go(state: PipelineState, next_stage: str, extra: str = "", **updates) -> dict:
    """Hand over to `next_stage` (a user-role message also keeps the transcript valid for
    providers that reject a conversation ending on an assistant message)."""
    text = HANDOFF[next_stage] + (("\n\n" + extra) if extra else "")
    return {"stage": next_stage, "stage_start": len(state["messages"]) + 1, "nudges": 0,
            "messages": [HumanMessage(content=f"[pipeline] {text}")], **updates}


def _conclude(state: PipelineState, status: str, note: str = "", **updates) -> dict:
    """End the build. In `auto` control returns to the chat node, which is told the outcome so it
    can report it to the user; in strict `pipeline` mode the run is over."""
    if state.get("after") == "chat":
        detail = BUILD_STATUS_DETAIL.get(status, "")
        extra = (note.strip() + "\n") if note.strip() else ""
        text = (f"[pipeline] The build finished with status '{status}'. {detail}\n{extra}"
                "You are back in the conversation: tell the user briefly and honestly what happened "
                "(what changed, what was verified, what was NOT done) and suggest next steps. Do not "
                "call start_build again unless the user asks.")
        return {"stage": CHAT_STAGE, "status": status, "nudges": 0,
                "messages": [HumanMessage(content=text)], **updates}
    update: dict = {"stage": "done", "status": status, **updates}
    if note:
        update["messages"] = [AIMessage(content=f"[pipeline] {note}")]
    return update


def _nudge(state: PipelineState, text: str) -> dict:
    nudges = state["nudges"] + 1
    if nudges > MAX_NUDGES:
        return _conclude(state, "stalled",
                         f"Stopped: the {state['stage']} stage did not complete its required "
                         f"steps. {text}")
    return {"nudges": nudges, "messages": [HumanMessage(content=f"[pipeline] {text}")]}


def _next_stage(state: PipelineState, stage: str) -> str | None:
    """The next chosen step after `stage`, or None if the build ends here."""
    chosen = state.get("steps") or list(STAGES)
    later = [s for s in STAGES if s in chosen and STAGES.index(s) > STAGES.index(stage)]
    return later[0] if later else None


def make_intake(chat: bool):
    """Entry node: every new user message starts here."""

    def intake(state: PipelineState) -> dict:
        update = {"revisions": 0, "nudges": 0, "builds": 0, "status": "", "reviewed_tree": "",
                  "halted": False}
        if chat:
            return {**update, "stage": CHAT_STAGE, "steps": [], "after": "chat", "task": ""}
        last = state["messages"][-1] if state["messages"] else None
        task = content_to_text(last.content) if isinstance(last, HumanMessage) else ""
        return {**update, "stage": "planner", "steps": list(STAGES), "after": "done",
                "task": task[:6000], "stage_start": max(len(state["messages"]) - 1, 0)}

    return intake


def handoff(state: PipelineState) -> dict:
    """Chat -> build. Answers the start_build call and switches to the first chosen stage."""
    messages = state["messages"]
    calls = messages[-1].tool_calls
    builds = state.get("builds", 0)
    error = None
    steps: list[str] = []
    task = ""
    if len(calls) != 1:
        error = "start_build must be called on its own, without other tool calls in the same message."
    elif builds >= MAX_BUILDS_PER_TURN:
        error = (f"{builds} builds were already started for this message. Talk to the user "
                 f"before starting another.")
    else:
        args = calls[0].get("args") or {}
        task = str(args.get("task") or "").strip()
        if not task:
            error = "'task' must not be empty."
        else:
            try:
                steps = normalize_build_steps(args.get("steps"))
            except ValueError as exc:
                error = str(exc)
    if error:  # every tool call must be answered; stay in chat so the model can react
        return {"messages": [_error_message(c, error) for c in calls]}

    first = steps[0]
    plan = " -> ".join(s for s in ("plan", "code", "review", "commit")
                       if tools_module.STEP_TO_STAGE[s] in steps)
    result = ToolMessage(content=f"Build started with steps: {plan}.", tool_call_id=calls[0]["id"],
                         name="start_build", status="success")
    note = HumanMessage(content="[pipeline] " + BUILD_START[first].replace("{task}", task[:4000]))
    return {"messages": [result, note], "stage": first, "steps": steps,
            "stage_start": len(messages) + 2, "revisions": 0, "nudges": 0, "status": "",
            "reviewed_tree": "", "builds": builds + 1, "task": task[:6000]}


def advance(state: PipelineState) -> dict:
    """The gate. Decides, from tool evidence only, whether the current stage may hand over."""
    stage = state["stage"]
    window = _window(state)
    done = _succeeded(window)
    chosen = state.get("steps") or list(STAGES)

    if state.get("halted"):
        return _conclude(state, "stalled", "Stopped: the token budget for this request was reached "
                         "(RUN_TOKEN_BUDGET), so no further steps were run.")

    if stage == "planner":
        missing = {"update_plan", "update_todo"} - done
        if not missing:
            nxt = _next_stage(state, "planner")
            return _go(state, nxt) if nxt else _conclude(state, "coded")
        return _nudge(state, "Before the pipeline can continue you must call "
                             + " and ".join(sorted(missing)) + ".")

    if stage == "coder":
        if done & {"write_file", "edit_file"}:
            nxt = _next_stage(state, "coder")
            return _go(state, nxt) if nxt else _conclude(state, "coded")
        if state["revisions"] == 0:
            # Nothing was changed (a question, or the coder could not proceed): nothing to
            # review or commit, so the build ends here.
            return _conclude(state, "no_changes")
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
            nxt = _next_stage(state, "reviewer")
            if nxt:
                return _go(state, nxt, reviewed_tree=reviewed)
            return _conclude(state, "reviewed", reviewed_tree=reviewed)
        if verdict == "NO_CHANGES":
            return _conclude(state, "no_changes", "The reviewer found no changes to commit.")
        if state["revisions"] >= max_review_rounds() or "coder" not in chosen:
            return _conclude(state, "review_failed",
                             f"Review requests changes (revision rounds used: "
                             f"{state['revisions']}); nothing was committed.\n\n{review_text[:3000]}")
        return _go(state, "coder",
                   "The reviewer requested changes. Address every point:\n\n" + review_text[:6000],
                   revisions=state["revisions"] + 1)

    if stage == "committer":
        if "git_commit" in done:
            return _conclude(state, "committed")
        return _nudge(state, "Stage the changes with git_add and call git_commit with a clear "
                             "message. The pipeline only finishes once git_commit succeeds.")

    return _conclude(state, state.get("status") or "stalled")


def _allowed_calls(state: PipelineState, calls: list[dict]) -> list[dict]:
    allowed = set(STAGE_TOOLS.get(state["stage"], ()))
    return [c for c in calls if c["name"] in allowed]


def route_after_model(state: PipelineState) -> str:
    stage = state["stage"]
    calls = getattr(state["messages"][-1], "tool_calls", None) or []
    if not calls:
        return END if stage == CHAT_STAGE else "advance"
    if stage == CHAT_STAGE and any(c["name"] == "start_build" for c in calls):
        return "handoff"
    # Only calls that will actually run can require approval.
    if any(requires_approval(c["name"]) for c in _allowed_calls(state, calls)):
        return "approval"
    return "tools"


# Tools that would move unreviewed content into history.
COMMIT_GUARDED_TOOLS = frozenset({"git_add", "git_commit"})


def _image_messages(calls: list[dict], results: list[ToolMessage]) -> list[HumanMessage]:
    """view_image answers with text; the picture itself is attached as a user message after the
    tool results (tool results cannot carry images on every provider)."""
    extra: list[HumanMessage] = []
    for call, result in zip(calls, results):
        if call["name"] != "view_image" or result.status == "error":
            continue
        try:
            mime, data, rel = tools_module.load_image((call.get("args") or {}).get("path", ""))
        except (ValueError, tools_module.PathError, OSError):
            continue
        extra.append(HumanMessage(content=[
            {"type": "text", "text": f"{context_module.PIPELINE_PREFIX} Image from view_image: {rel}"},
            tools_module.image_content_block(mime, data),
        ]))
    return extra


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
    return {"messages": results + _image_messages(state["messages"][-1].tool_calls, results)}


def _tag(reply, stage: str):
    """Remember which stage produced a message (shown when a conversation is reopened)."""
    try:
        reply.response_metadata = {**(reply.response_metadata or {}), "stage": stage}
    except Exception:  # not every message type allows it
        pass
    return reply


def _build_pipeline_graph(model, checkpointer, chat: bool):
    templates = {s: STAGE_PROMPTS[s] for s in STAGES}
    templates[CHAT_STAGE] = CHAT_PROMPT
    stages = ((CHAT_STAGE,) if chat else ()) + STAGES
    bound = {s: model.bind_tools([TOOLS_BY_NAME[n] for n in STAGE_TOOLS[s]]) for s in stages}

    def make_stage_node(stage: str):
        def run(state: PipelineState) -> dict:
            exceeded, used, budget = usage.over_budget(state["messages"])
            if exceeded:
                note = AIMessage(content=f"[budget] Stopped: this request has used {used:,} tokens "
                                         f"(limit {budget:,}, RUN_TOKEN_BUDGET). Send a smaller or "
                                         f"more specific request, or raise the limit.")
                return {"messages": [_tag(note, stage)], "halted": True}
            workspace = tools_module.current_workspace()
            system = render_prompt(templates[stage], workspace)
            reply = bound[stage].invoke(
                context_module.build_messages(state, stage, system, workspace))
            return {"messages": [_tag(reply, stage)]}
        run.__name__ = stage
        return run

    graph = StateGraph(PipelineState)
    graph.add_node("intake", make_intake(chat))
    for stage in stages:
        graph.add_node(stage, make_stage_node(stage))
    graph.add_node("approval", lambda state: {})  # pure interrupt point, does no work
    graph.add_node("tools", tools_node)
    graph.add_node("advance", advance)

    route_targets = {"approval": "approval", "tools": "tools", "advance": "advance"}
    if chat:
        graph.add_node("handoff", handoff)
        route_targets.update({"handoff": "handoff", END: END})
        graph.add_edge(START, "intake")
        graph.add_edge("intake", CHAT_STAGE)
        graph.add_conditional_edges("handoff", lambda state: state["stage"],
                                    {s: s for s in stages})
    else:
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "planner")
    for stage in stages:
        graph.add_conditional_edges(stage, route_after_model, route_targets)
    graph.add_edge("approval", "tools")
    graph.add_conditional_edges("tools", lambda state: state["stage"], {s: s for s in stages})
    graph.add_conditional_edges("advance", lambda state: state["stage"],
                                {**{s: s for s in stages}, "done": END})
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
    checkpointer = checkpointer if checkpointer is not None else history.get_checkpointer()
    if mode == "react":
        return _build_react_graph(model, checkpointer)
    return _build_pipeline_graph(model, checkpointer, chat=(mode == "auto"))


def get_graph(mode: str | None = None):
    """Return the cached compiled graph for `mode` (building it on first use)."""
    mode = mode or default_mode()
    key = (mode, history.db_path())
    if key not in _graphs:
        _graphs[key] = build_graph(mode)
    return _graphs[key]


# ---------------------------------------------------------------------------
# shared driver used by both the CLI and the Streamlit app
# ---------------------------------------------------------------------------
def make_config(thread_id: str, mode: str | None = None) -> dict:
    try:
        limit = int(os.getenv("RECURSION_LIMIT", "150"))
    except ValueError:
        limit = 150
    configurable = {"thread_id": thread_id}
    if mode:
        configurable["mode"] = mode  # remembered in the conversation index by drive()
    return {"configurable": configurable, "recursion_limit": limit}


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
        stage = (getattr(message, "response_metadata", None) or {}).get("stage")
        return {"kind": "ai", "node": stage or node, "content": content_to_text(message.content),
                "tool_calls": [_call_info(c) for c in (message.tool_calls or [])]}
    if isinstance(message, ToolMessage):
        return {"kind": "tool", "node": node, "name": message.name or "tool",
                "content": content_to_text(message.content),
                "error": getattr(message, "status", "success") == "error",
                "tool_call_id": message.tool_call_id}
    return {"kind": "pipeline", "node": node, "content": content_to_text(message.content)}


def messages_to_events(messages) -> list[dict]:
    """Rebuild the display log of a stored conversation."""
    events: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            text = content_to_text(message.content)
            if text.startswith("[pipeline]"):
                events.append({"kind": "pipeline", "node": "pipeline", "content": text})
            else:
                events.append({"kind": "user", "content": text})
        elif isinstance(message, (AIMessage, ToolMessage)):
            events.append(message_to_event("agent", message))
    return events


def history_events(graph, config) -> list[dict]:
    """Display events of the conversation stored under config's thread (empty if unknown)."""
    values = graph.get_state(config).values or {}
    return messages_to_events(values.get("messages") or [])


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
    workspace = str(tools_module.workspace_for((config.get("configurable") or {}).get("thread_id")))
    return {"stage": values.get("stage"), "status": values.get("status", ""),
            "revisions": values.get("revisions", 0), "steps": values.get("steps") or [],
            "waiting": bool(snapshot.next),
            "usage": usage.summary(values.get("messages") or [], workspace)}


def _remember(config, user_text: str) -> None:
    """Record the conversation in the index; persistence problems must never stop the chat."""
    try:
        configurable = config.get("configurable", {})
        history.record(configurable["thread_id"], configurable.get("mode") or default_mode(),
                       str(tools_module.workspace_for(configurable["thread_id"])), user_text)
    except Exception:
        pass


def _in_workspace(workspace, iterator) -> Iterator[Any]:
    """Iterate `iterator` with `workspace` active only while the graph is actually running (the
    ContextVar is set around each step, so it never leaks into the caller's own code between
    events, and LangGraph copies it into the worker threads that run the nodes)."""
    iterator = iter(iterator)
    while True:
        with tools_module.use_workspace(workspace):
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item


def drive(graph, config, user_text: str | None = None) -> Iterator[dict]:
    """Run the graph and yield events until it finishes or needs a human decision.

    * user_text given  -> start a new turn with that message.
    * user_text None   -> resume (after Approve, or after reject_tool_calls).
    Batches containing only safe tools are resumed automatically.
    After the generator ends, call pending_tool_calls() to see whether approval is needed.
    """
    graph_input: Any = None
    workspace = tools_module.workspace_for(config.get("configurable", {}).get("thread_id"))
    tools_module.ensure_workspace(workspace)
    if user_text is not None:
        _remember(config, user_text)
        graph_input = {"messages": [HumanMessage(content=user_text)]}
    while True:
        for chunk in _in_workspace(workspace, graph.stream(graph_input, config,
                                                           stream_mode="updates")):
            yield from _events_from_chunk(chunk)
        pending = pending_tool_calls(graph, config)
        if pending and not needs_human(pending):
            graph_input = None  # safe batch: approve automatically and continue
            continue
        return
