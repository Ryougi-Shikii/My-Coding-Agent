"""Chat-first 'auto' mode: it chats by default and runs only the build steps that are needed."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import graph as g
from agent import sandbox
from agent import tools as t
from tests.conftest import git
from tests.fakes import ScriptedChatModel, ai, call
from tests.test_graph import committer_steps, new_graph, planner_steps, run_to_end, stage_order, status_of


def auto(model):
    return new_graph(model, "auto")


def hello_coder():
    return [ai("", call("write_file", path="hello.py", content="print('hi')\n")), "Implemented hello.py."]


def reviewer_steps():
    return [ai("", call("git_init")), ai("", call("review_changes")), "Reviewed."]


def build_script(steps=None, task="write hello.py", wrapup="All done.", **extra):
    args = {"task": task}
    if steps is not None:
        args["steps"] = steps
    script = {
        "chat": [ai("On it.", call("start_build", **args)), wrapup],
        "planner": planner_steps(),
        "coder": hello_coder(),
        "reviewer": reviewer_steps(),
        "review": ["Looks good.\nVERDICT: APPROVE"],
        "committer": committer_steps(),
    }
    script.update(extra)
    return script


# =========================================================================== it is a conversation
class TestChatsFirst:
    def test_hi_gets_a_reply_and_nothing_else(self, fake_llm, ws):
        model = fake_llm({"chat": ["Hey! What are we working on today?"]})
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "hi")
        assert rounds == []  # no approval prompt for a greeting
        assert model.roles() == ["chat"]
        assert [e["kind"] for e in events] == ["ai"] and events[0]["node"] == "chat"
        assert events[0]["content"] == "Hey! What are we working on today?"
        assert not (ws / "hello.py").exists() and not (ws / ".git").exists()
        assert status_of(graph, config)["stage"] == "chat"

    def test_discussion_and_brainstorming_need_no_tools(self, fake_llm, ws):
        model = fake_llm({"chat": ["Option A is simpler.", "Option B scales better.", "Go with A."]})
        graph, config = auto(model)
        for text in ("should I use A or B?", "what about scaling?", "ok decide"):
            events, rounds = run_to_end(graph, config, text)
            assert rounds == [] and events[-1]["kind"] == "ai"
        assert model.roles() == ["chat"] * 3
        # the conversation accumulates: the last call saw the whole discussion
        assert model.log[-1][1] > model.log[0][1]

    def test_it_can_read_the_project_without_asking(self, fake_llm, ws):
        (ws / "app.py").write_text("print('hello')\n")
        script = {"chat": [ai("", call("list_files"), call("read_file", path="app.py"),
                              call("git_status")),
                           "It prints hello."]}
        graph, config = auto(fake_llm(script))
        events, rounds = run_to_end(graph, config, "what does app.py do?")
        assert rounds == []
        results = {e["name"]: e["content"] for e in events if e["kind"] == "tool"}
        assert "app.py" in results["list_files"] and "print('hello')" in results["read_file"]
        assert events[-1]["content"] == "It prints hello."

    def test_chat_cannot_change_files_or_commit_itself(self, fake_llm, ws):
        script = {"chat": [ai("", call("write_file", path="x.py", content="1"),
                              call("edit_file", path="x.py", find="a", replace="b"),
                              call("git_commit", message="m"), call("install_packages", packages="rich")),
                           "I can't do that directly."]}
        graph, config = auto(fake_llm(script))
        events, rounds = run_to_end(graph, config, "just write it")
        assert rounds == []  # blocked calls never even ask for approval
        errors = [e for e in events if e["kind"] == "tool"]
        assert {e["name"] for e in errors} == {"write_file", "edit_file", "git_commit", "install_packages"}
        assert all(e["error"] and "not permitted in the chat stage" in e["content"] for e in errors)
        assert not (ws / "x.py").exists()

    def test_running_tests_in_chat_still_asks(self, fake_llm, ws, monkeypatch):
        monkeypatch.setattr(sandbox, "run_python_args_in_sandbox",
                            lambda root, args, timeout=None: sandbox.RunResult("docker", "img", 0, "3 passed", ""))
        script = {"chat": [ai("", call("run_tests")), "All 3 tests pass."]}
        graph, config = auto(fake_llm(script))
        events, rounds = run_to_end(graph, config, "do the tests pass?")
        assert rounds == [["run_tests"]]
        assert events[-1]["content"] == "All 3 tests pass."

    def test_rejecting_in_chat_lets_the_model_answer(self, fake_llm, ws):
        script = {"chat": [ai("", call("run_tests")), "No problem, I won't run them."]}
        graph, config = auto(fake_llm(script))
        events, rounds = run_to_end(graph, config, "run the tests", decide=lambda p: "not now")
        assert rounds == [["run_tests"]]
        assert any(e["kind"] == "ai" and "won't run" in e["content"] for e in events)

    def test_the_conversation_is_never_stuck_in_a_build(self, fake_llm, ws):
        script = build_script(["code"])
        script["chat"] += ["Sure, here is a thought.", "Another thought."]
        graph, config = auto(fake_llm(script))
        run_to_end(graph, config, "make hello.py")
        events, rounds = run_to_end(graph, config, "what would you improve?")
        assert rounds == [] and stage_order(events) == ["chat"]


# =========================================================================== flexible builds
class TestFlexibleBuilds:
    def test_default_build_plans_codes_and_reviews_but_does_not_commit(self, fake_llm, ws):
        model = fake_llm(build_script(wrapup="hello.py is written and reviewed (not committed)."))
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "please create hello.py")
        assert stage_order(events) == ["chat", "planner", "coder", "reviewer", "chat"]
        assert rounds == [["write_file"]]  # no commit approval: no commit step
        assert status_of(graph, config)["status"] == "reviewed"
        assert (ws / "hello.py").exists() and git(ws, "log", "--oneline").strip() == ""
        started = [e for e in events if e["kind"] == "tool" and e["name"] == "start_build"]
        assert started[0]["content"] == "Build started with steps: plan -> code -> review."
        wrapup = [e["content"] for e in events if e["kind"] == "pipeline" and "finished with status" in e["content"]]
        assert "status 'reviewed'" in wrapup[0] and "NOT committed" in wrapup[0]
        assert events[-1]["node"] == "chat" and "not committed" in events[-1]["content"]

    def test_commit_only_when_asked(self, fake_llm, ws):
        model = fake_llm(build_script(["plan", "code", "review", "commit"], wrapup="Committed."))
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "create hello.py and commit it")
        assert stage_order(events) == ["chat", "planner", "coder", "reviewer", "committer", "chat"]
        assert rounds == [["write_file"], ["git_commit"]]
        assert status_of(graph, config)["status"] == "committed"
        assert "Add hello.py" in git(ws, "log", "--oneline")

    def test_tiny_edit_skips_planning_review_and_commit(self, fake_llm, ws):
        model = fake_llm(build_script(["code"], wrapup="Done."))
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "add hello.py")
        assert stage_order(events) == ["chat", "coder", "chat"]
        assert set(model.roles()) == {"chat", "coder"}
        assert rounds == [["write_file"]]
        assert status_of(graph, config)["status"] == "coded"
        assert (ws / "hello.py").exists() and not (ws / ".git").exists()
        wrapup = [e["content"] for e in events if e["kind"] == "pipeline" and "finished with status" in e["content"]]
        assert "NOT reviewed and NOT committed" in wrapup[0]

    def test_review_and_commit_changes_that_already_exist(self, fake_llm, ws):
        (ws / "hello.py").write_text("print('already there')\n")
        script = build_script(["review", "commit"], task="review and commit hello.py", wrapup="Committed.")
        model = fake_llm(script)
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "commit what I have")
        assert stage_order(events) == ["chat", "reviewer", "committer", "chat"]
        assert "coder" not in model.roles() and rounds == [["git_commit"]]
        assert "Add hello.py" in git(ws, "log", "--oneline")

    def test_review_only_reports_and_does_not_loop_to_a_coder_that_was_not_requested(self, fake_llm, ws):
        (ws / "hello.py").write_text("print('x')\n")
        script = build_script(["review"], wrapup="The reviewer wants changes.")
        script["review"] = ["[blocker] bad\nVERDICT: REQUEST_CHANGES"]
        model = fake_llm(script)
        graph, config = auto(model)
        events, _ = run_to_end(graph, config, "review it")
        assert stage_order(events) == ["chat", "reviewer", "chat"]
        assert "coder" not in model.roles()
        assert status_of(graph, config)["status"] == "review_failed"

    def test_unchosen_steps_never_run_even_if_the_order_is_unusual(self, fake_llm, ws):
        model = fake_llm(build_script(["commit", "review", "code"], wrapup="ok"))  # given out of order
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "x")
        assert stage_order(events)[:5] == ["chat", "coder", "reviewer", "committer", "chat"]
        assert "planner" not in model.roles()

    def test_the_reviewed_tree_rule_still_applies_inside_a_build(self, fake_llm, ws, monkeypatch):
        def sandbox_writes_a_file(root, args, timeout=None):
            (ws / "generated.txt").write_text("written by the test run\n")
            return sandbox.RunResult("docker", "img", 0, "1 passed", "")

        monkeypatch.setattr(sandbox, "run_python_args_in_sandbox", sandbox_writes_a_file)
        script = build_script(["code", "review", "commit"], wrapup="stopped")
        script["reviewer"] = [ai("", call("git_init")), ai("", call("review_changes")),
                              ai("", call("run_tests")), "done"] * 3
        script["review"] = ["fine\nVERDICT: APPROVE"] * 3
        model = fake_llm(script)
        graph, config = auto(model)
        run_to_end(graph, config, "x")
        assert "committer" not in model.roles() and git(ws, "log", "--oneline").strip() == ""
        assert status_of(graph, config)["status"] == "stalled"

    def test_rejecting_the_write_ends_the_build_and_the_chat_reports_it(self, fake_llm, ws):
        script = build_script(["plan", "code", "review"], wrapup="You declined the edit; nothing changed.")
        script["coder"] = [ai("", call("write_file", path="hello.py", content="x")), "Understood."]
        model = fake_llm(script)
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "make hello.py", decide=lambda p: "no")
        assert not (ws / "hello.py").exists()
        assert "reviewer" not in model.roles()
        assert status_of(graph, config)["status"] == "no_changes"
        assert events[-1]["content"] == "You declined the edit; nothing changed."


# =========================================================================== handoff guard rails
class TestHandoffGuardRails:
    def tool_errors(self, events):
        return [e for e in events if e["kind"] == "tool" and e["error"]]

    @pytest.mark.parametrize("steps,fragment", [(["commit"], "at least"), (["code", "commit"], "requires 'review'"),
                                                (["plan"], "at least"), (["deploy"], "unknown step")])
    def test_invalid_steps_are_refused_and_chat_continues(self, fake_llm, ws, steps, fragment):
        script = {"chat": [ai("", call("start_build", task="x", steps=steps)), "Sorry, let me rethink."]}
        model = fake_llm(script)
        graph, config = auto(model)
        events, rounds = run_to_end(graph, config, "go")
        (error,) = self.tool_errors(events)
        assert fragment in error["content"] and error["name"] == "start_build"
        assert model.roles() == ["chat", "chat"] and events[-1]["content"] == "Sorry, let me rethink."
        assert status_of(graph, config)["stage"] == "chat"

    def test_empty_task_is_refused(self, fake_llm, ws):
        script = {"chat": [ai("", call("start_build", task="   ")), "ok"]}
        graph, config = auto(fake_llm(script))
        events, _ = run_to_end(graph, config, "go")
        assert "must not be empty" in self.tool_errors(events)[0]["content"]

    def test_start_build_must_be_alone(self, fake_llm, ws):
        (ws / "a.txt").write_text("x")
        script = {"chat": [ai("", call("read_file", path="a.txt"), call("start_build", task="x")), "ok"]}
        model = fake_llm(script)
        graph, config = auto(model)
        events, _ = run_to_end(graph, config, "go")
        errors = self.tool_errors(events)
        assert {e["name"] for e in errors} == {"read_file", "start_build"}  # every call is answered
        assert all("on its own" in e["content"] for e in errors)
        assert set(model.roles()) == {"chat"}

    def test_builds_per_message_are_limited(self, fake_llm, ws):
        def sb():
            return ai("", call("start_build", task="again", steps=["code"]))
        script = {"chat": [sb(), sb(), sb(), sb(), "I'll stop here."], "coder": ["nothing to do"] * 3}
        model = fake_llm(script)
        graph, config = auto(model)
        events, _ = run_to_end(graph, config, "go")
        assert model.roles().count("coder") == 3
        assert "already started" in self.tool_errors(events)[-1]["content"]
        assert events[-1]["content"] == "I'll stop here."
        # a new user message resets the allowance
        script["chat"] += [sb(), "fresh build done"]
        script["coder"] += ["nothing again"]
        events, _ = run_to_end(graph, config, "try again")
        assert model.roles().count("coder") == 4

    def test_start_build_is_not_a_tool_of_the_other_modes(self, fake_llm, ws):
        assert "start_build" not in {tool.name for tool in t.TOOLS}
        assert "start_build" in t.STAGE_TOOLS["chat"]
        for stage in t.STAGES:
            assert "start_build" not in t.STAGE_TOOLS[stage]
        assert not t.requires_approval("start_build")
        # even if a model in another stage emits it, it is just a refused tool call
        state = {"stage": "coder", "messages": [AIMessage(content="", tool_calls=[call("start_build", task="x")])]}
        (result,) = g.tools_node(state)["messages"]
        assert result.status == "error" and "not permitted" in result.content

    def test_direct_execution_of_start_build_is_an_error(self, ws):
        assert t.start_build.invoke({"task": "x"}).startswith("ERROR")


# =========================================================================== history and modes
class TestStoredConversationEvents:
    def test_history_events_rebuild_the_display(self, fake_llm, ws):
        model = fake_llm(build_script(["plan", "code", "review"], wrapup="Done."))
        graph, config = auto(model)
        live, _ = run_to_end(graph, config, "make hello.py")
        stored = g.history_events(graph, config)
        assert stored[0] == {"kind": "user", "content": "make hello.py"}
        assert [e["node"] for e in stored if e["kind"] == "ai"] == [e["node"] for e in live if e["kind"] == "ai"]
        assert [e["name"] for e in stored if e["kind"] == "tool"] == [e["name"] for e in live if e["kind"] == "tool"]
        assert any(e["kind"] == "pipeline" and "Build started" in e["content"] for e in stored)

    def test_messages_to_events_distinguishes_user_and_pipeline_notes(self):
        events = g.messages_to_events([HumanMessage(content="hello"), HumanMessage(content="[pipeline] note"),
                                       AIMessage(content="hi", response_metadata={"stage": "coder"}),
                                       ToolMessage(content="out", tool_call_id="1", name="read_file")])
        assert [e["kind"] for e in events] == ["user", "pipeline", "ai", "tool"]
        assert events[2]["node"] == "coder"

    def test_empty_thread_has_no_events(self, fake_llm, ws):
        graph, config = auto(fake_llm({}))
        assert g.history_events(graph, config) == []


class TestSwitchingModesInOneConversation:
    @pytest.mark.parametrize("first,second", [("auto", "react"), ("react", "auto"), ("auto", "pipeline"),
                                              ("pipeline", "auto"), ("react", "pipeline")])
    def test_messages_carry_over(self, fake_llm, ws, first, second):
        seen = []

        def responder(key, messages):
            seen.append(" ".join(str(m.content) for m in messages))
            return AIMessage(content="noted")

        model = ScriptedChatModel(responder=responder)
        saver = MemorySaver()
        graph_a = g.build_graph(first, model=model, checkpointer=saver)
        graph_b = g.build_graph(second, model=model, checkpointer=saver)
        config = g.make_config("shared")
        # strict pipeline needs a plan before it moves on; a plain answer is enough to end its stage
        list(g.drive(graph_a, config, "my favourite colour is teal"))
        list(g.drive(graph_b, config, "what did I say?"))
        assert "my favourite colour is teal" in seen[-1]


class TestModeConstants:
    def test_modes_and_help(self):
        assert set(g.MODES) == {"auto", "react", "pipeline"} == set(g.MODE_HELP)
        assert g.MODES[0] == "auto"
