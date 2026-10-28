"""Graph behaviour: approval granularity and hard-enforced planner -> coder -> reviewer -> committer."""

from __future__ import annotations

import itertools

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import graph as g
from agent import tools as t
from tests.conftest import git
from tests.fakes import ai, call

_thread = itertools.count(1)


def new_graph(model, mode="pipeline"):
    graph = g.build_graph(mode, model=model, checkpointer=MemorySaver())
    return graph, g.make_config(f"t{next(_thread)}")


def run_to_end(graph, config, text, decide=None):
    """Drive until finished. `decide(pending)` returns True (approve) or False/str (reject)."""
    events = list(g.drive(graph, config, text))
    rounds = []
    while True:
        pending = g.pending_tool_calls(graph, config)
        if not pending:
            return events, rounds
        rounds.append([c["name"] for c in pending])
        verdict = decide(pending) if decide else True
        if verdict is True:
            events += list(g.drive(graph, config, None))
        else:
            g.reject_tool_calls(graph, config, verdict if isinstance(verdict, str) else "")
            events += list(g.drive(graph, config, None))


def stage_order(events):
    order = []
    for e in events:
        if e["kind"] == "ai" and (not order or order[-1] != e["node"]):
            order.append(e["node"])
    return order


def planner_steps():
    return [
        ai("", call("read_agent_state")),
        ai("", call("update_plan", plan_markdown="# Plan\nwrite hello.py"),
           call("update_todo", todo_markdown="- [ ] write hello.py")),
        ai("Plan ready."),
    ]


def committer_steps(message="Add hello.py"):
    return [ai("", call("git_add", path=".")), ai("", call("git_commit", message=message)),
            ai("Committed.")]


def happy_script():
    return {
        "planner": planner_steps(),
        "coder": [ai("", call("write_file", path="hello.py", content="print('hi')\n")),
                  ai("", call("check_python_syntax", path="hello.py")),
                  ai("Implemented hello.py.")],
        # Fresh workspace: the reviewer must init git itself before review_changes can work.
        "reviewer": [ai("", call("git_init")), ai("", call("review_changes")), ai("Reviewed.")],
        "review": ["Looks good.\nVERDICT: APPROVE"],
        "committer": committer_steps(),
    }


def status_of(graph, config):
    status = g.get_run_status(graph, config)
    status.pop("usage", None)  # token counts have their own tests
    return status


# =========================================================================== pipeline happy path
class TestPipelineHappyPath:
    def test_stages_run_in_order_and_only_dangerous_tools_pause(self, fake_llm, ws):
        model = fake_llm(happy_script())
        graph, config = new_graph(model)
        events, rounds = run_to_end(graph, config, "write hello.py")

        assert stage_order(events) == ["planner", "coder", "reviewer", "committer"]
        # 12 tool calls ran, but a human was only asked about write_file and git_commit.
        assert rounds == [["write_file"], ["git_commit"]]
        assert status_of(graph, config)["status"] == "committed"
        assert (ws / "hello.py").read_text() == "print('hi')\n"
        assert "Add hello.py" in git(ws, "log", "--oneline")
        assert (ws / ".agent" / "PLAN.md").read_text().startswith("# Plan")
        assert len(list((ws / ".agent" / "reviews").glob("review_*.md"))) == 1

    def test_nothing_is_written_before_approval(self, fake_llm, ws):
        graph, config = new_graph(fake_llm(happy_script()))
        list(g.drive(graph, config, "write hello.py"))
        pending = g.pending_tool_calls(graph, config)
        assert [c["name"] for c in pending] == ["write_file"] and pending[0]["dangerous"]
        assert not (ws / "hello.py").exists()
        assert graph.get_state(config).next == ("approval",)

    def test_safe_tool_results_are_streamed_without_asking(self, fake_llm, ws):
        graph, config = new_graph(fake_llm(happy_script()))
        events = list(g.drive(graph, config, "go"))
        tool_events = [e["name"] for e in events if e["kind"] == "tool"]
        assert tool_events[:3] == ["read_agent_state", "update_plan", "update_todo"]

    def test_every_new_user_message_restarts_at_the_planner(self, fake_llm, ws):
        script = {"planner": planner_steps() + planner_steps(),
                  "coder": ["Just an answer.", "Another answer."]}
        model = fake_llm(script)
        graph, config = new_graph(model)
        run_to_end(graph, config, "first question")
        assert status_of(graph, config) == {"stage": "done", "status": "no_changes",
                                            "revisions": 0, "waiting": False,
                                            "steps": ["planner", "coder", "reviewer", "committer"]}
        events, _ = run_to_end(graph, config, "second question")
        assert model.roles().count("planner") == 6  # 3 planner calls per run
        assert stage_order(events)[0] == "planner"


# =========================================================================== rejection
class TestRejection:
    def test_reject_injects_tool_messages_and_model_responds(self, fake_llm, ws):
        seen = []

        def responder(key, messages):
            if key == "coder":
                seen.append(messages[-1])
            return None

        script = {"planner": planner_steps(),
                  "coder": [ai("", call("write_file", path="hello.py", content="x")),
                            ai("Understood, I will not write it.")]}
        model = fake_llm(script, responder)
        graph, config = new_graph(model)
        events, rounds = run_to_end(graph, config, "write hello.py",
                                    decide=lambda pending: "not now")
        assert rounds == [["write_file"]]
        # The coder saw the rejection as the tool result of its own call...
        rejection = seen[-1]
        assert isinstance(rejection, ToolMessage) and rejection.status == "error"
        assert "REJECTED" in rejection.content and "not now" in rejection.content
        assert rejection.name == "write_file"
        # ...answered it, and nothing ran: no file, no review, no commit.
        assert not (ws / "hello.py").exists()
        assert status_of(graph, config)["status"] == "no_changes"
        assert "reviewer" not in model.roles() and "committer" not in model.roles()
        assert any(e["kind"] == "ai" and "will not write" in e["content"] for e in events)

    def test_rejection_message_answers_every_pending_call(self, fake_llm, ws):
        script = {"planner": planner_steps(),
                  "coder": [ai("", call("write_file", path="a.py", content="1"),
                               call("create_directory", path="d")), "ok"]}
        graph, config = new_graph(fake_llm(script))
        list(g.drive(graph, config, "go"))
        pending = g.pending_tool_calls(graph, config)
        messages = g.reject_tool_calls(graph, config)
        assert [m.tool_call_id for m in messages] == [c["id"] for c in pending]

    def test_reject_with_nothing_pending_is_a_noop(self, fake_llm, ws):
        graph, config = new_graph(fake_llm({}))
        assert g.reject_tool_calls(graph, config) == []


# =========================================================================== hard enforcement
class TestStageEnforcement:
    def test_coder_cannot_commit_or_review(self, fake_llm, ws):
        script = {"planner": planner_steps(),
                  "coder": [ai("", call("git_commit", message="sneaky"), call("review_changes"),
                               call("git_init")), ai("done")]}
        graph, config = new_graph(fake_llm(script))
        events, rounds = run_to_end(graph, config, "go")
        errors = [e for e in events if e["kind"] == "tool" and e["error"]]
        assert {e["name"] for e in errors} == {"git_commit", "review_changes", "git_init"}
        assert all("not permitted in the coder stage" in e["content"] for e in errors)
        assert rounds == []  # blocked calls never even ask for approval
        assert not (ws / ".git").exists()

    def test_reviewer_cannot_modify_files(self, fake_llm, ws):
        script = happy_script()
        script["reviewer"].insert(0, ai("", call("write_file", path="evil.py", content="pwn"),
                                        call("edit_file", path="hello.py", find="hi", replace="pwn")))
        graph, config = new_graph(fake_llm(script))
        events, rounds = run_to_end(graph, config, "go")
        assert not (ws / "evil.py").exists()
        assert (ws / "hello.py").read_text() == "print('hi')\n"
        assert rounds == [["write_file"], ["git_commit"]]  # the blocked reviewer batch asked nobody
        blocked = [e for e in events if e["kind"] == "tool" and e["error"]]
        assert {e["name"] for e in blocked} == {"write_file", "edit_file"}
        assert status_of(graph, config)["status"] == "committed"

    def test_planner_cannot_write_code(self, fake_llm, ws):
        script = {"planner": [ai("", call("write_file", path="early.py", content="x"))] + planner_steps(),
                  "coder": ["nothing to do"]}
        graph, config = new_graph(fake_llm(script))
        events, rounds = run_to_end(graph, config, "go")
        assert not (ws / "early.py").exists() and rounds == []
        assert any(e["kind"] == "tool" and e["error"] and "planner stage" in e["content"] for e in events)

    @pytest.mark.parametrize("stage,tool_name", [
        ("planner", "write_file"), ("planner", "git_commit"), ("planner", "run_tests"),
        ("coder", "git_commit"), ("coder", "git_add"), ("coder", "review_changes"),
        ("coder", "update_plan"),
        ("reviewer", "write_file"), ("reviewer", "edit_file"), ("reviewer", "install_packages"),
        ("reviewer", "git_commit"), ("reviewer", "git_add"),
        ("committer", "write_file"), ("committer", "edit_file"), ("committer", "run_python_file"),
        ("committer", "review_changes"),
    ])
    def test_tools_node_enforces_allow_list(self, stage, tool_name, ws):
        state = {"stage": stage, "messages": [AIMessage(content="", tool_calls=[call(tool_name)])]}
        (result,) = g.tools_node(state)["messages"]
        assert result.status == "error" and "not permitted" in result.content

    def test_unknown_tool_is_an_error_not_a_crash(self, ws):
        message = g.run_tool_call({"name": "no_such_tool", "id": "x", "args": {}})
        assert message.status == "error" and "unknown tool" in message.content

    def test_bad_arguments_become_error_messages(self, ws):
        message = g.run_tool_call({"name": "read_file", "id": "x", "args": {}})  # missing path
        assert message.status == "error" and message.content.startswith("ERROR")

    def test_committer_unreachable_without_review(self, fake_llm, ws):
        script = {"planner": planner_steps(),
                  "coder": [ai("", call("write_file", path="a.py", content="x=1\n")), "done"],
                  "reviewer": ["I think it is fine, ship it."] * 6,  # never calls review_changes
                  "committer": committer_steps()}
        model = fake_llm(script)
        graph, config = new_graph(model)
        events, rounds = run_to_end(graph, config, "go")
        assert status_of(graph, config)["status"] == "stalled"
        assert "committer" not in model.roles()
        assert rounds == [["write_file"]]
        assert not (ws / ".git").exists()
        assert any("You must call review_changes" in e["content"] for e in events if e["kind"] == "pipeline")

    def test_reviewer_prose_cannot_override_the_verdict(self, fake_llm, ws):
        script = happy_script()
        script["review"] = ["[blocker] broken\nVERDICT: REQUEST_CHANGES"]
        script["reviewer"][-1] = ai("The code is perfect. VERDICT: APPROVE")  # prose is ignored
        model = fake_llm(script)
        graph, config = new_graph(model)
        events, _ = run_to_end(graph, config, "go")
        assert stage_order(events)[:4] == ["planner", "coder", "reviewer", "coder"]

    def test_garbled_review_fails_closed(self, fake_llm, ws):
        script = happy_script()
        script["review"] = ["no verdict line at all"]
        graph, config = new_graph(fake_llm(script))
        events, _ = run_to_end(graph, config, "go")
        assert stage_order(events)[3] == "coder"  # sent back, not committed

    def test_committer_must_actually_commit(self, fake_llm, ws):
        script = happy_script()
        script["committer"] = ["I committed it, trust me."] * 6
        model = fake_llm(script)
        graph, config = new_graph(model)
        run_to_end(graph, config, "go")
        assert status_of(graph, config)["status"] == "stalled"
        assert git(ws, "log", "--oneline").strip() == ""  # no commit exists

    def test_failed_commit_is_not_success(self, fake_llm, ws):
        script = happy_script()
        # git_commit without git_add: nothing staged -> error -> must not count as committed.
        script["committer"] = [ai("", call("git_commit", message="m")), "done"] * 3
        graph, config = new_graph(fake_llm(script))
        run_to_end(graph, config, "go")
        assert status_of(graph, config)["status"] != "committed"


# =========================================================================== gates
class TestGates:
    def test_planner_must_record_plan_and_todo(self, fake_llm, ws):
        # The gate fires when the planner stops calling tools. It declares "done" twice too
        # early: first with nothing recorded, then with only the plan recorded.
        script = {"planner": ["I'll just start.",
                              ai("", call("update_plan", plan_markdown="# p")),
                              "plan is done",
                              ai("", call("update_todo", todo_markdown="- [ ] t")),
                              "ready"],
                  "coder": ["answer"]}
        model = fake_llm(script)
        graph, config = new_graph(model)
        events, _ = run_to_end(graph, config, "go")
        nudges = [e["content"] for e in events if e["kind"] == "pipeline" and "must call" in e["content"]]
        assert len(nudges) == 2
        assert "update_plan and update_todo" in nudges[0] and "update_todo" in nudges[1]
        assert "update_plan" not in nudges[1]
        assert "coder" in model.roles()

    def test_planner_that_never_plans_stalls(self, fake_llm, ws):
        model = fake_llm({"planner": ["no plan"] * 10})
        graph, config = new_graph(model)
        events, _ = run_to_end(graph, config, "go")
        assert status_of(graph, config)["status"] == "stalled" and "coder" not in model.roles()
        assert any(e["kind"] == "ai" and "Stopped" in e["content"] for e in events)

    def test_failed_tool_calls_do_not_count_as_progress(self, fake_llm, ws):
        # update_plan with empty text fails -> status error -> gate must not pass.
        script = {"planner": [ai("", call("update_plan", plan_markdown=" "),
                                 call("update_todo", todo_markdown="- [ ] x")), "done"] * 5}
        model = fake_llm(script)
        graph, config = new_graph(model)
        run_to_end(graph, config, "go")
        assert "coder" not in model.roles()

    def test_no_edits_means_nothing_to_review(self, fake_llm, ws):
        model = fake_llm({"planner": planner_steps(), "coder": ["This is just an explanation."]})
        graph, config = new_graph(model)
        events, rounds = run_to_end(graph, config, "explain something")
        assert status_of(graph, config)["status"] == "no_changes" and rounds == []
        assert set(model.roles()) == {"planner", "coder"}

    def test_review_loop_sends_feedback_to_coder_then_commits(self, fake_llm, ws):
        script = {
            "planner": planner_steps(),
            "coder": [ai("", call("write_file", path="hello.py", content="print('bug')\n")), "first draft",
                      ai("", call("edit_file", path="hello.py", find="bug", replace="fixed")), "fixed it"],
            "reviewer": [ai("", call("git_init")), ai("", call("review_changes")), "round 1",
                         ai("", call("review_changes")), "round 2"],
            "review": ["- [blocker] prints bug\nVERDICT: REQUEST_CHANGES", "Fine now.\nVERDICT: APPROVE"],
            "committer": committer_steps("Add hello.py (fixed)"),
        }
        model = fake_llm(script)
        graph, config = new_graph(model)
        events, rounds = run_to_end(graph, config, "go")
        assert rounds == [["write_file"], ["edit_file"], ["git_commit"]]
        assert stage_order(events) == ["planner", "coder", "reviewer", "coder", "reviewer", "committer"]
        feedback = [e["content"] for e in events if e["kind"] == "pipeline" and "prints bug" in e["content"]]
        assert feedback and "Address every point" in feedback[0]
        assert (ws / "hello.py").read_text() == "print('fixed')\n"
        state = status_of(graph, config)
        assert state["status"] == "committed" and state["revisions"] == 1
        assert len(list((ws / ".agent" / "reviews").glob("review_*.md"))) == 2

    def test_review_rounds_are_bounded(self, fake_llm, ws, monkeypatch):
        monkeypatch.setenv("MAX_REVIEW_ROUNDS", "1")
        script = {
            "planner": planner_steps(),
            "coder": [ai("", call("write_file", path="a.py", content="v=1\n")), "d1",
                      ai("", call("write_file", path="a.py", content="v=2\n")), "d2"],
            "reviewer": [ai("", call("git_init")), ai("", call("review_changes")), "r1",
                         ai("", call("review_changes")), "r2"],
            "review": ["bad\nVERDICT: REQUEST_CHANGES"] * 2,
            "committer": committer_steps(),
        }
        model = fake_llm(script)
        graph, config = new_graph(model)
        events, _ = run_to_end(graph, config, "go")
        assert status_of(graph, config)["status"] == "review_failed"
        assert "committer" not in model.roles()
        assert git(ws, "log", "--oneline").strip() == ""

    def test_revision_without_edits_is_nudged(self, fake_llm, ws):
        script = {
            "planner": planner_steps(),
            "coder": [ai("", call("write_file", path="a.py", content="v=1\n")), "d1", "i changed nothing"],
            "reviewer": [ai("", call("git_init")), ai("", call("review_changes")), "r1"],
            "review": ["bad\nVERDICT: REQUEST_CHANGES"],
        }
        model = fake_llm(script)
        graph, config = new_graph(model)
        events, _ = run_to_end(graph, config, "go")
        assert any("Make real edits" in e["content"] for e in events if e["kind"] == "pipeline")
        assert "committer" not in model.roles()

    def test_reviewer_with_clean_tree_ends_without_commit(self, fake_llm, ws):
        # The coder "wrote" a file identical to what is already committed.
        t.git_init.invoke({})
        (ws / "a.py").write_text("x = 1\n")
        t.git_add.invoke({"path": "."})
        t.git_commit.invoke({"message": "base"})
        script = {"planner": planner_steps(),
                  "coder": [ai("", call("write_file", path="a.py", content="x = 1\n")), "done"],
                  "reviewer": [ai("", call("review_changes")), "nothing to review"],
                  "committer": committer_steps()}
        model = fake_llm(script)
        graph, config = new_graph(model)
        run_to_end(graph, config, "go")
        assert status_of(graph, config)["status"] == "no_changes"
        assert "committer" not in model.roles() and "review" not in model.roles()


# =========================================================================== approval modes
class TestApprovalModes:
    def test_all_mode_asks_even_for_safe_tools(self, fake_llm, ws, monkeypatch):
        monkeypatch.setenv("APPROVAL_MODE", "all")
        graph, config = new_graph(fake_llm(happy_script()))
        events, rounds = run_to_end(graph, config, "go")
        assert rounds[0] == ["read_agent_state"]
        assert len(rounds) > 5

    def test_dangerous_mode_flags_calls_in_events(self, fake_llm, ws):
        graph, config = new_graph(fake_llm(happy_script()))
        events, _ = run_to_end(graph, config, "go")
        calls = {c["name"]: c["dangerous"] for e in events if e["kind"] == "ai" for c in e["tool_calls"]}
        assert calls["write_file"] and calls["git_commit"]
        assert not calls["read_agent_state"] and not calls["review_changes"] and not calls["git_add"]

    def test_mixed_batch_asks_once_for_the_whole_batch(self, fake_llm, ws):
        script = {"planner": planner_steps(),
                  "coder": [ai("", call("read_file", path="missing.txt"),
                               call("write_file", path="a.py", content="x=1\n")), "done"]}
        graph, config = new_graph(fake_llm(script))
        list(g.drive(graph, config, "go"))
        pending = g.pending_tool_calls(graph, config)
        assert [(c["name"], c["dangerous"]) for c in pending] == [("read_file", False), ("write_file", True)]


# =========================================================================== react mode
class TestReactMode:
    def react_script(self):
        return {"react": [ai("", call("read_file", path="a.txt")),
                          ai("", call("write_file", path="a.txt", content="new")),
                          ai("All done.")]}

    def test_graph_pauses_before_every_tool_batch_but_driver_skips_safe_ones(self, fake_llm, ws):
        (ws / "a.txt").write_text("old")
        model = fake_llm(self.react_script())
        graph, config = new_graph(model, "react")
        # Raw graph: paused before `tools` even for the safe read_file (spec: interrupt_before).
        list(graph.stream({"messages": [("user", "go")]}, config, stream_mode="updates"))
        assert graph.get_state(config).next == ("tools",)
        assert [c["name"] for c in g.pending_tool_calls(graph, config)] == ["read_file"]
        assert not g.needs_human(g.pending_tool_calls(graph, config))

        # Driver: the same situation is auto-resumed; the human is only asked about write_file.
        graph2, config2 = new_graph(fake_llm(self.react_script()), "react")
        events, rounds = run_to_end(graph2, config2, "go")
        assert rounds == [["write_file"]]
        assert (ws / "a.txt").read_text() == "new"
        assert [e["name"] for e in events if e["kind"] == "tool"] == ["read_file", "write_file"]
        assert events[-1]["kind"] == "ai" and events[-1]["content"] == "All done."

    def test_reject_in_react_mode(self, fake_llm, ws):
        (ws / "a.txt").write_text("old")
        seen = []
        model = fake_llm(self.react_script(),
                         lambda key, messages: seen.append(messages[-1]) and None)
        graph, config = new_graph(model, "react")
        events, rounds = run_to_end(graph, config, "go", decide=lambda p: "no thanks")
        assert rounds == [["write_file"]]
        assert (ws / "a.txt").read_text() == "old"
        assert isinstance(seen[-1], ToolMessage) and "REJECTED" in seen[-1].content
        assert events[-1]["content"] == "All done."  # the model got to respond

    def test_system_prompt_is_attached_with_workspace(self, fake_llm, ws):
        captured = []
        model = fake_llm({"react": ["hi"]}, lambda key, messages: captured.append(messages[0]) and None)
        graph, config = new_graph(model, "react")
        list(g.drive(graph, config, "hello"))
        assert "autonomous coding agent" in captured[0].content and str(ws) in captured[0].content
        assert "{workspace}" not in captured[0].content

    def test_prompt_keyword_fallbacks(self, monkeypatch, fake_llm, ws):
        attempts = []

        def fake_create(model, tools=None, **kwargs):
            attempts.append(sorted(kwargs))
            if "prompt" in kwargs or "state_modifier" in kwargs and len(attempts) < 2:
                raise TypeError("unexpected keyword")
            return "graph"

        monkeypatch.setattr(g, "create_react_agent", fake_create)
        assert g._build_react_graph(object(), MemorySaver()) == "graph"
        assert attempts == [["checkpointer", "interrupt_before", "prompt"],
                            ["checkpointer", "interrupt_before", "state_modifier"]]

        attempts.clear()
        monkeypatch.setattr(g, "create_react_agent",
                            lambda model, tools=None, **kw: (attempts.append(sorted(kw)), "graph")[1]
                            if not ({"prompt", "state_modifier"} & set(kw)) else (_ for _ in ()).throw(TypeError()))
        assert g._build_react_graph(object(), MemorySaver()) == "graph"
        assert attempts[-1] == ["checkpointer", "interrupt_before"]  # finally: no prompt at all


# =========================================================================== builders
class TestBuilders:
    def test_get_graph_is_cached_per_mode(self, fake_llm, ws):
        fake_llm({})
        assert g.get_graph("react") is g.get_graph("react")
        assert g.get_graph("pipeline") is g.get_graph("pipeline")
        assert g.get_graph("react") is not g.get_graph("pipeline")

    def test_unknown_mode(self, fake_llm, ws):
        fake_llm({})
        with pytest.raises(ValueError):
            g.build_graph("chaos")

    def test_default_mode_from_env(self, monkeypatch):
        monkeypatch.delenv("AGENT_MODE", raising=False)
        assert g.default_mode() == "auto"
        monkeypatch.setenv("AGENT_MODE", "REACT")
        assert g.default_mode() == "react"
        monkeypatch.setenv("AGENT_MODE", "pipeline")
        assert g.default_mode() == "pipeline"
        monkeypatch.setenv("AGENT_MODE", "nonsense")
        assert g.default_mode() == "auto"

    def test_graph_uses_required_building_blocks(self):
        import inspect

        from agent import llm
        assert "from langchain.chat_models import init_chat_model" in inspect.getsource(llm)
        assert "from langgraph.prebuilt import create_react_agent" in inspect.getsource(g)
        from agent import history
        assert "history.get_checkpointer" in inspect.getsource(g)
        assert "MemorySaver" in inspect.getsource(history) and "SqliteSaver" in inspect.getsource(history)
        assert 'interrupt_before=["tools"]' in inspect.getsource(g)

    def test_threads_are_isolated(self, fake_llm, ws):
        script = {"planner": planner_steps() * 2, "coder": ["a1", "a2"]}
        graph = g.build_graph("pipeline", model=fake_llm(script), checkpointer=MemorySaver())
        c1, c2 = g.make_config("one"), g.make_config("two")
        run_to_end(graph, c1, "first")
        assert g.get_run_status(graph, c2)["stage"] is None
        run_to_end(graph, c2, "second")
        texts1 = [m.content for m in graph.get_state(c1).values["messages"]]
        assert "first" in texts1 and "second" not in texts1
