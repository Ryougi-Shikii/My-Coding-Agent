"""Tests for: tool result status, read_file ranges, per-session workspaces, context/cost control,
token budget, image/file tools, run_python, sandbox limits and housekeeping, evals."""

from __future__ import annotations

import base64
import itertools
import subprocess
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import context as ctx
from agent import graph as g
from agent import history, usage
from agent import sandbox as sb
from agent import tools as t
from tests.fakes import ScriptedChatModel, ai, call
from tests.test_graph import happy_script, new_graph, run_to_end

# A real 1x1 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def run(tool, **args):
    return tool.invoke(args)


# ============================================================================ tool results
class TestToolResultStatus:
    def test_file_that_starts_with_error_is_not_an_error(self, ws):
        (ws / "log.txt").write_text("ERROR codes are documented below\n")
        message = g.run_tool_call(call("read_file", path="log.txt"))
        assert message.status == "success"
        assert message.content.startswith("ERROR codes")

    def test_real_failures_are_still_errors(self, ws):
        assert g.run_tool_call(call("read_file", path="missing.txt")).status == "error"
        assert g.run_tool_call(call("read_file", path="../outside")).status == "error"


class TestReadFile:
    def test_line_range(self, ws):
        (ws / "a.txt").write_text("".join(f"line {i}\n" for i in range(1, 101)))
        out = run(t.read_file, path="a.txt", start_line=10, end_line=12)
        assert out.splitlines()[0].startswith("[lines 10-12 of 100")
        assert "line 10" in out and "line 12" in out and "line 13" not in out

    def test_start_beyond_end_is_an_error(self, ws):
        (ws / "a.txt").write_text("one\n")
        assert run(t.read_file, path="a.txt", start_line=5).startswith("ERROR")

    def test_binary_files_are_refused_with_a_hint(self, ws):
        (ws / "x.bin").write_bytes(b"\x00\x01\x02" * 100)
        out = run(t.read_file, path="x.bin")
        assert out.startswith("ERROR") and "inspect_file" in out

    def test_truncation_mentions_line_ranges(self, ws, monkeypatch):
        monkeypatch.setattr(t, "MAX_READ_CHARS", 100)
        (ws / "big.txt").write_text("x" * 1000)
        assert "start_line" in run(t.read_file, path="big.txt")


# ============================================================================ view_image / inspect_file
class TestImages:
    def test_view_image_validates(self, ws):
        (ws / "a.png").write_bytes(PNG)
        out = run(t.view_image, path="a.png")
        assert out.startswith("Image attached: a.png") and "1x1" in out
        (ws / "t.txt").write_text("hi")
        assert run(t.view_image, path="t.txt").startswith("ERROR")
        assert run(t.view_image, path="nope.png").startswith("ERROR")
        assert run(t.view_image, path="../x.png").startswith("ERROR")

    def test_size_limit(self, ws, monkeypatch):
        monkeypatch.setattr(t, "MAX_IMAGE_BYTES", 10)
        (ws / "a.png").write_bytes(PNG)
        out = run(t.view_image, path="a.png")
        assert out.startswith("ERROR") and "smaller" in out

    def test_pipeline_attaches_the_image_after_tool_results(self, ws, fake_llm):
        (ws / "a.png").write_bytes(PNG)
        model = fake_llm({"chat": [ai("", call("view_image", path="a.png")), ai("I see it.")]})
        graph, config = new_graph(model, "auto")
        run_to_end(graph, config, "look at a.png")
        messages = graph.get_state(config).values["messages"]
        kinds = [type(m).__name__ for m in messages]
        tool_at = next(i for i, m in enumerate(messages) if isinstance(m, ToolMessage))
        image = messages[tool_at + 1]
        assert isinstance(image, HumanMessage) and ctx.has_image(image)
        assert image.content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert kinds[-1] == "AIMessage"

    def test_view_image_is_not_given_to_react_mode(self):
        assert "view_image" not in [x.name for x in t.TOOLS]
        assert "view_image" in t.TOOLS_BY_NAME


class TestInspectFile:
    def test_png_and_text_and_binary(self, ws):
        (ws / "a.png").write_bytes(PNG)
        (ws / "n.txt").write_text("a\nb\nc\n")
        (ws / "x.bin").write_bytes(b"\x00" * 64)
        assert "png" in run(t.inspect_file, path="a.png") and "1x1" in run(t.inspect_file, path="a.png")
        assert "lines: 3" in run(t.inspect_file, path="n.txt")
        assert "binary" in run(t.inspect_file, path="x.bin")
        assert run(t.inspect_file, path="missing").startswith("ERROR")
        assert "directory" in run(t.inspect_file, path=".")

    def test_media_uses_ffprobe_in_the_sandbox(self, ws, monkeypatch):
        (ws / "v.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        seen = {}

        def fake_run(workspace, code, timeout=None, script_args=()):
            seen.update(code=code, args=list(script_args))
            return sb.RunResult("docker", "img", 0, '{"format": {"duration": "5.1"}}', "")

        monkeypatch.setattr(sb, "run_code_in_sandbox", fake_run)
        out = run(t.inspect_file, path="v.mp4")
        assert "mp4" in out and "duration" in out and "ffprobe" in seen["code"]
        assert seen["args"] == ["./v.mp4"]


class TestOutputs:
    def test_lists_files_in_out_newest_first(self, ws):
        (ws / "out").mkdir()
        (ws / "out" / "a.txt").write_text("a")
        (ws / "other.txt").write_text("x")
        files = t.output_files(ws)
        assert [name for name, _ in files] == ["out/a.txt"]

    def test_symlinked_out_is_ignored(self, ws, tmp_path):
        from tests.conftest import symlink
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "secret").write_text("s")
        symlink(tmp_path / "elsewhere", ws / "out")
        assert t.output_files(ws) == []


# ============================================================================ run_python + sandbox
class TestRunPython:
    def test_classification(self):
        assert t.requires_approval("run_python")
        assert not t.requires_approval("inspect_file") and not t.requires_approval("view_image")
        assert "run_python" in t.STAGE_TOOLS["coder"] and "run_python" not in t.STAGE_TOOLS["reviewer"]
        assert "run_python" not in t.STAGE_TOOLS["planner"] + t.STAGE_TOOLS["committer"]

    def test_code_is_sent_on_stdin_not_written_or_passed_as_an_argument(self, ws, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"], seen["input"] = cmd, kw.get("input")
            return subprocess.CompletedProcess(cmd, 0, "42\n", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = run(t.run_python, code="print(6*7)")
        assert "42" in out and "exit code: 0" in out
        assert seen["input"] == "print(6*7)"
        assert "-i" in seen["cmd"] and seen["cmd"][-2:] == ["python", "-"]
        assert "print(6*7)" not in " ".join(seen["cmd"])
        assert list(ws.iterdir()) == [ws / ".agent"]  # nothing written to the workspace

    def test_empty_and_huge_code_are_rejected(self, ws):
        assert run(t.run_python, code="  ").startswith("ERROR")
        assert run(t.run_python, code="x" * (sb.MAX_SNIPPET_CHARS + 1)).startswith("ERROR")

    def test_no_docker_no_execution(self, ws):
        assert "Docker is not available" in run(t.run_python, code="print(1)")


class TestSandboxSettings:
    def test_defaults_are_unchanged(self, ws):
        cmd = sb.build_docker_command(ws, "img", ["x.py"], "n")
        assert "-i" not in cmd
        assert cmd[cmd.index("--memory") + 1] == "512m" and cmd[cmd.index("--cpus") + 1] == "1"
        assert cmd[cmd.index("--pids-limit") + 1] == "256"
        assert cmd[cmd.index("--tmpfs") + 1] == "/tmp:rw,size=128m"

    def test_limits_are_configurable_and_validated(self, ws, monkeypatch):
        monkeypatch.setenv("SANDBOX_MEMORY", "2g")
        monkeypatch.setenv("SANDBOX_CPUS", "2")
        monkeypatch.setenv("SANDBOX_PIDS", "64")
        monkeypatch.setenv("SANDBOX_TMP_MB", "512")
        cmd = sb.build_docker_command(ws, "img", ["x.py"], "n")
        assert cmd[cmd.index("--memory") + 1] == "2g" and cmd[cmd.index("--memory-swap") + 1] == "2g"
        assert cmd[cmd.index("--cpus") + 1] == "2" and cmd[cmd.index("--pids-limit") + 1] == "64"
        assert cmd[cmd.index("--tmpfs") + 1] == "/tmp:rw,size=512m"
        for bad in ("2g --privileged", "-1", "abc", "1;rm", ""):
            monkeypatch.setenv("SANDBOX_MEMORY", bad)
            assert sb.build_docker_command(ws, "img", ["x.py"], "n")[
                sb.build_docker_command(ws, "img", ["x.py"], "n").index("--memory") + 1] == "512m"
        monkeypatch.setenv("SANDBOX_CPUS", "1 --network host")
        assert sb.sandbox_cpus() == "1"

    def test_network_is_always_off(self, ws, monkeypatch):
        monkeypatch.setenv("SANDBOX_MEMORY", "4g")
        cmd = sb.build_docker_command(ws, "img", ["x.py"], "n", interactive=True)
        assert cmd[cmd.index("--network") + 1] == "none" and "--read-only" in cmd
        assert cmd[cmd.index("--cap-drop") + 1] == "ALL"


class TestInstallHardening:
    def test_allowlist(self, monkeypatch):
        monkeypatch.setenv("PACKAGE_ALLOWLIST", "numpy, Pillow ,scikit_learn")
        assert sb.validate_requirements(["numpy>=1.26", "pillow", "scikit-learn[x]"])
        with pytest.raises(ValueError, match="allowlist"):
            sb.validate_requirements(["requests"])
        monkeypatch.setenv("PACKAGE_ALLOWLIST", "*")
        assert sb.validate_requirements(["requests"]) == ["requests"]

    def test_source_builds_off_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        seen = {}

        def fake_run(cmd, **kw):
            ctx_dir = __import__("pathlib").Path(cmd[-1])
            seen["flag"] = (ctx_dir / "source_builds.json").read_text()
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        root = tmp_path / "w"
        root.mkdir()
        sb.install_packages_into_sandbox_image(root, ["numpy"])
        assert seen["flag"] == "false"
        monkeypatch.setenv("ALLOW_SOURCE_BUILDS", "true")
        sb.install_packages_into_sandbox_image(root, ["numpy"])
        assert seen["flag"] == "true"

    def test_prune_removes_only_unreferenced_images(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        w1, w2 = tmp_path / "w1", tmp_path / "w2"
        for w, tag in ((w1, "lca-sandbox:aaa"), (w2, "lca-sandbox:bbb")):
            (w / ".agent").mkdir(parents=True)
            (w / ".agent" / "sandbox_image.txt").write_text(tag + "\n")
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[:2] == ["docker", "images"]:
                out = "lca-sandbox:aaa\nlca-sandbox:bbb\nlca-sandbox:old1\nlca-sandbox:old2\n"
                return subprocess.CompletedProcess(cmd, 0, out, "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        removed = sb.prune_unused_images([w1, w2])
        assert removed == ["lca-sandbox:old1", "lca-sandbox:old2"]
        assert [c for c in calls if c[:2] == ["docker", "rmi"]] == [
            ["docker", "rmi", "lca-sandbox:old1"], ["docker", "rmi", "lca-sandbox:old2"]]

    def test_prune_without_docker_does_nothing(self, tmp_path):
        assert sb.prune_unused_images([tmp_path]) == []


# ============================================================================ per-session workspaces
class TestSessionWorkspaces:
    @pytest.fixture
    def session_mode(self, tmp_path, monkeypatch, ws):
        monkeypatch.setenv("WORKSPACE_MODE", "session")
        monkeypatch.setenv("WORKSPACES_DIR", str(tmp_path / "sessions"))
        return tmp_path / "sessions"

    def test_each_thread_gets_its_own_directory(self, session_mode):
        a, b = t.workspace_for("thread-a"), t.workspace_for("thread-b")
        assert a != b and a.parent == b.parent == session_mode.resolve()
        assert t.workspace_for("thread-a") == a

    def test_unsafe_thread_ids_cannot_escape(self, session_mode):
        for bad in ("../../etc", "a/b", "..", "x" * 200, "ünï"):
            path = t.workspace_for(bad)
            assert path.parent == session_mode.resolve(), bad

    def test_shared_mode_keeps_one_workspace(self, ws, monkeypatch):
        monkeypatch.setenv("WORKSPACE_MODE", "shared")
        assert t.workspace_for("a") == t.workspace_for("b") == ws

    def test_two_conversations_do_not_see_each_others_files(self, session_mode, fake_llm):
        script = {"chat": [ai("", call("start_build", task="write", steps=["code"])), "done"] * 2,
                  "coder": [ai("", call("write_file", path="who.txt", content="A")), "ok",
                            ai("", call("write_file", path="who.txt", content="B")), "ok"]}
        model = fake_llm(script)
        graph = g.build_graph("auto", model=model, checkpointer=MemorySaver())
        for thread in ("conv-a", "conv-b"):
            config = g.make_config(thread, "auto")
            list(g.drive(graph, config, "write"))
            while g.pending_tool_calls(graph, config):
                list(g.drive(graph, config, None))
        assert (t.workspace_for("conv-a") / "who.txt").read_text() == "A"
        assert (t.workspace_for("conv-b") / "who.txt").read_text() == "B"

    def test_workspace_does_not_leak_out_of_drive(self, session_mode, fake_llm):
        model = fake_llm({"chat": ["hello"]})
        graph = g.build_graph("auto", model=model, checkpointer=MemorySaver())
        before = t.current_workspace()
        iterator = g.drive(graph, g.make_config("leak", "auto"), "hi")
        next(iterator)  # paused between events, as a UI would be while rendering
        assert t.current_workspace() == before
        list(iterator)
        assert t.current_workspace() == before

    def test_concurrent_conversations_stay_apart(self, session_mode):
        seen = {}
        barrier = threading.Barrier(2)

        def work(name):
            with t.use_workspace(t.workspace_for(name)):
                barrier.wait()
                seen[name] = t.current_workspace().name

        threads = [threading.Thread(target=work, args=(n,)) for n in ("p", "q")]
        [x.start() for x in threads]
        [x.join() for x in threads]
        assert seen == {"p": "p", "q": "q"}

    def test_history_lists_all_session_conversations(self, session_mode):
        for name in ("one", "two"):
            history.record(name, "auto", str(t.workspace_for(name)), f"title {name}")
        history.record("elsewhere", "auto", "/somewhere/else", "other project")
        titles = {r["title"] for r in history.list_conversations(t.workspace_scope())}
        assert titles == {"title one", "title two"}

    def test_sandbox_image_pointer_is_per_workspace(self, session_mode):
        a, b = t.workspace_for("a"), t.workspace_for("b")
        for w in (a, b):
            t.ensure_workspace(w)
        (a / ".agent" / "sandbox_image.txt").write_text("lca-sandbox:abc\n")
        assert sb.get_sandbox_image(a) == "lca-sandbox:abc"
        assert sb.get_sandbox_image(b) == "python:3.12-slim"


# ============================================================================ context control
def tool_pair(i, size):
    cid = f"c{i}"
    return [AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": f"f{i}"},
                                                "id": cid, "type": "tool_call"}]),
            ToolMessage(content="x" * size, tool_call_id=cid, name="read_file")]


class TestCompaction:
    def test_old_tool_results_are_shortened_recent_ones_kept(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_OLD_TOOL_CHARS", "200")
        monkeypatch.setenv("CONTEXT_KEEP_RECENT", "4")
        messages = [HumanMessage(content="go")] + list(itertools.chain(*[tool_pair(i, 5000) for i in range(10)]))
        out = ctx.compact(messages)
        tools_out = [m for m in out if isinstance(m, ToolMessage)]
        assert len(tools_out[0].content) < 400 and "omitted" in tools_out[0].content
        assert len(tools_out[-1].content) == 5000 and len(tools_out[-2].content) == 5000
        assert len(messages[2].content) == 5000  # the stored conversation is untouched

    def test_prefix_is_stable_between_calls(self, monkeypatch):
        """Prompt caches need the start of the conversation to stay identical as it grows."""
        monkeypatch.setenv("CONTEXT_OLD_TOOL_CHARS", "100")
        monkeypatch.setenv("CONTEXT_KEEP_RECENT", "4")
        base = [HumanMessage(content="go")] + list(itertools.chain(*[tool_pair(i, 3000) for i in range(12)]))
        first = ctx.compact(base)
        more = ctx.compact(base + tool_pair(99, 3000))
        same = sum(1 for a, b in zip(first, more) if a.content == b.content)
        assert same >= len(first) - 6  # only the newest few messages differ

    def test_old_images_are_replaced(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_KEEP_IMAGES", "1")

        def image(n):
            return HumanMessage(content=[{"type": "text", "text": f"[pipeline] image {n}"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}])

        out = ctx.compact([image(1), image(2), image(3)])
        assert [ctx.has_image(m) for m in out] == [False, False, True]
        assert "removed" in ctx.text_of(out[0].content)


class TestFitAndCollapse:
    def test_fit_starts_at_a_user_turn_and_keeps_the_latest(self):
        turns = []
        for i in range(10):
            turns += [HumanMessage(content=f"q{i} " + "y" * 1000), AIMessage(content="a" * 1000)]
        out = ctx.fit(turns, max_chars=5000)
        assert isinstance(out[0], HumanMessage) and out[-1].content == "a" * 1000
        assert ctx.total_chars(out) <= 5000 and out[0].content.startswith("q")

    def test_fit_never_returns_an_orphan_tool_result(self):
        messages = [HumanMessage(content="old " * 500), *tool_pair(1, 3000),
                    HumanMessage(content="new"), *tool_pair(2, 100)]
        out = ctx.fit(messages, max_chars=1000)
        assert isinstance(out[0], HumanMessage) and out[0].content == "new"

    def test_single_huge_latest_turn_is_kept_whole(self):
        out = ctx.fit([HumanMessage(content="a"), AIMessage(content="b"),
                       HumanMessage(content="z" * 9000)], max_chars=100)
        assert len(out) == 1 and len(out[0].content) == 9000

    def test_finished_builds_collapse_to_their_outcome(self):
        done = "[pipeline] The build finished with status 'committed'. The changes were committed."
        messages = [HumanMessage(content="build it"),
                    AIMessage(content="", tool_calls=[{"name": "start_build", "args": {}, "id": "s",
                                                       "type": "tool_call"}]),
                    ToolMessage(content="Build started", tool_call_id="s", name="start_build"),
                    HumanMessage(content="[pipeline] Build started. You are the CODER"),
                    *tool_pair(1, 4000), AIMessage(content="coded"),
                    HumanMessage(content=done), AIMessage(content="All done.")]
        out = ctx.collapse_builds(messages)
        assert len(out) < len(messages) and any("committed" in ctx.text_of(m.content) for m in out)
        assert not any(isinstance(m, ToolMessage) and m.name == "read_file" for m in out)
        assert any(isinstance(m, ToolMessage) and m.name == "start_build" for m in out)

    def test_unfinished_build_is_not_collapsed(self):
        messages = [HumanMessage(content="x"),
                    AIMessage(content="", tool_calls=[{"name": "start_build", "args": {}, "id": "s",
                                                       "type": "tool_call"}]),
                    ToolMessage(content="Build started", tool_call_id="s", name="start_build"),
                    *tool_pair(1, 100)]
        assert len(ctx.collapse_builds(messages)) == len(messages)


class TestStageIsolation:
    def test_coder_does_not_see_the_planners_transcript(self, ws, fake_llm):
        sizes = {}

        def responder(key, messages):
            sizes.setdefault(key, []).append(
                (len(messages), " ".join(ctx.text_of(m.content) for m in messages)))
            return None

        model = fake_llm(happy_script())
        model.responder = responder
        graph, config = new_graph(model)
        run_to_end(graph, config, "write hello.py please")
        coder_text = sizes["coder"][0][1]
        assert "write hello.py please" in coder_text          # the task is in the briefing
        assert "Saved plan" in coder_text and "write hello.py" in coder_text
        assert "read_agent_state" not in coder_text            # planner's tool calls are not
        assert "Plan ready." not in coder_text

    def test_committer_gets_plan_but_not_the_review_transcript(self, ws, fake_llm):
        seen = {}

        def responder(key, messages):
            seen.setdefault(key, " ".join(ctx.text_of(m.content) for m in messages))
            return None

        model = fake_llm(happy_script())
        model.responder = responder
        graph, config = new_graph(model)
        run_to_end(graph, config, "hello")
        assert "Looks good." not in seen["committer"] and "VERDICT" not in seen["committer"]
        assert "Review approved" in seen["committer"]

    def test_message_count_stays_flat_across_stages(self, ws, fake_llm):
        model = fake_llm(happy_script())
        graph, config = new_graph(model)
        run_to_end(graph, config, "hello")
        full = len(graph.get_state(config).values["messages"])
        biggest = max(n for key, n in model.log if key != "review")
        assert biggest < full  # no stage was sent the entire transcript

    def test_chat_context_stays_bounded_over_many_turns(self, ws, fake_llm, monkeypatch):
        monkeypatch.setenv("CONTEXT_MAX_CHARS", "6000")
        sizes = []

        def responder(key, messages):
            sizes.append(ctx.total_chars(messages[1:]))
            return AIMessage(content="r" * 800)

        model = ScriptedChatModel(responder=responder)
        graph = g.build_graph("auto", model=model, checkpointer=MemorySaver())
        config = g.make_config("long", "auto")
        for i in range(30):
            list(g.drive(graph, config, f"message {i} " + "q" * 400))
        assert max(sizes) <= 6000 + 1300  # cap, plus the newest turn
        assert sizes[-1] < 7500 < 30 * 1200  # and far below the unbounded size


# ============================================================================ usage + budget
def usage_reply(text="ok", total=100, calls=()):
    return AIMessage(content=text, tool_calls=list(calls),
                     usage_metadata={"input_tokens": total - 10, "output_tokens": 10, "total_tokens": total})


class TestUsage:
    def test_sum_and_turn(self):
        messages = [HumanMessage(content="old"), usage_reply(total=100),
                    HumanMessage(content="new"), usage_reply(total=30), usage_reply(total=20)]
        assert usage.sum_usage(messages)["total"] == 150
        assert usage.turn_usage(messages)["total"] == 50
        assert usage.turn_usage(messages)["calls"] == 2

    def test_pipeline_notes_do_not_start_a_new_turn(self):
        messages = [HumanMessage(content="task"), usage_reply(total=40),
                    HumanMessage(content="[pipeline] next stage"), usage_reply(total=60)]
        assert usage.turn_usage(messages)["total"] == 100

    def test_extra_usage_from_tool_side_calls(self):
        usage.record_extra("k1", {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})
        assert usage.summary([usage_reply(total=100)], "k1")["total"] == 110

    def test_budget_setting(self, monkeypatch):
        assert usage.token_budget() == usage.DEFAULT_BUDGET
        monkeypatch.setenv("RUN_TOKEN_BUDGET", "0")
        assert usage.token_budget() == 0
        monkeypatch.setenv("RUN_TOKEN_BUDGET", "junk")
        assert usage.token_budget() == usage.DEFAULT_BUDGET

    def test_status_reports_usage(self, ws, fake_llm):
        model = fake_llm({"chat": [usage_reply("hello", 123)]})
        graph, config = new_graph(model, "auto")
        list(g.drive(graph, config, "hi"))
        assert g.get_run_status(graph, config)["usage"]["total"] == 123


class TestBudget:
    def test_build_stops_when_the_budget_is_reached(self, ws, fake_llm, monkeypatch):
        monkeypatch.setenv("RUN_TOKEN_BUDGET", "250")
        script = happy_script()
        # every planner/coder reply costs 100 tokens
        model = fake_llm({k: [usage_reply(v.content, 100, v.tool_calls) if isinstance(v, AIMessage) else v
                              for v in vals] for k, vals in script.items()})
        graph, config = new_graph(model)
        events, _ = run_to_end(graph, config, "hello")
        status = g.get_run_status(graph, config)
        assert status["status"] == "stalled" and status["stage"] == "done"
        assert any("[budget]" in e.get("content", "") for e in events)
        assert "reviewer" not in model.roles() and "committer" not in model.roles()

    def test_chat_says_so_and_stops(self, ws, fake_llm, monkeypatch):
        monkeypatch.setenv("RUN_TOKEN_BUDGET", "50")
        model = fake_llm({"chat": [usage_reply("first", 100, [call("list_files")]), "never sent"]})
        graph, config = new_graph(model, "auto")
        events = list(g.drive(graph, config, "hi"))
        assert any("[budget]" in e.get("content", "") for e in events)
        assert model.roles().count("chat") == 1  # the second model call was never made

    def test_zero_means_unlimited(self, ws, fake_llm, monkeypatch):
        monkeypatch.setenv("RUN_TOKEN_BUDGET", "0")
        model = fake_llm({"chat": [usage_reply("fine", 10_000_000)]})
        graph, config = new_graph(model, "auto")
        events = list(g.drive(graph, config, "hi"))
        assert not any("[budget]" in e.get("content", "") for e in events)


# ============================================================================ evals
class TestEvals:
    def test_tasks_are_well_formed(self):
        from evals.tasks import BY_ID, TASKS
        assert len(TASKS) == len(BY_ID) >= 8
        for task in TASKS:
            assert task.id and task.prompt and callable(task.check)
            for name in task.files:
                assert not name.startswith("/") and ".." not in name
        assert any(not task.expects_changes for task in TASKS)

    def test_check_helper_needs_pass_and_exit_zero(self, ws, monkeypatch):
        from evals import tasks as et
        monkeypatch.setattr(sb, "run_code_in_sandbox",
                            lambda *a, **k: sb.RunResult("docker", "i", 0, "PASS\n", ""))
        assert et.run_check(ws, "x")[0] is True
        monkeypatch.setattr(sb, "run_code_in_sandbox",
                            lambda *a, **k: sb.RunResult("docker", "i", 1, "PASS\n", "boom"))
        assert et.run_check(ws, "x")[0] is False
        monkeypatch.setattr(sb, "run_code_in_sandbox",
                            lambda *a, **k: sb.RunResult("docker", "i", 0, "no marker\n", ""))
        assert et.run_check(ws, "x")[0] is False

    def test_runner_end_to_end_with_a_scripted_model(self, tmp_path, monkeypatch, fake_llm):
        """The eval runner drives the real graph; here the model is scripted and checks run locally."""
        from evals import run as er
        from evals.tasks import BY_ID
        monkeypatch.setenv("WORKSPACE_MODE", "session")
        monkeypatch.setenv("WORKSPACES_DIR", str(tmp_path / "evalws"))
        monkeypatch.setenv("ALLOW_UNSAFE_LOCAL_EXECUTION", "true")
        script = happy_script()
        script["coder"] = [ai("", call("write_file", path="fizzbuzz.py", content=(
            "def fizzbuzz(n):\n    return ['FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else "
            "'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)]\n"))), ai("Done.")]
        fake_llm(script)
        result = er.run_task(BY_ID["fizzbuzz"], "pipeline", tmp_path)
        assert result["passed"], result["detail"]
        assert result["build_status"] == "committed" and result["docker"] is False

    def test_question_task_fails_if_files_are_touched(self, tmp_path, monkeypatch, fake_llm):
        from evals import run as er
        from evals.tasks import BY_ID
        monkeypatch.setenv("WORKSPACE_MODE", "session")
        monkeypatch.setenv("WORKSPACES_DIR", str(tmp_path / "evalws"))
        script = happy_script()
        script["coder"] = [ai("", call("write_file", path="secret.py", content="changed\n")), ai("ok")]
        fake_llm(script)
        result = er.run_task(BY_ID["question_only"], "pipeline", tmp_path)
        assert result["passed"] is False and "modified" in result["detail"]

    def test_question_task_passes_when_nothing_changes(self, tmp_path, monkeypatch, fake_llm):
        from evals import run as er
        from evals.tasks import BY_ID
        monkeypatch.setenv("WORKSPACE_MODE", "session")
        monkeypatch.setenv("WORKSPACES_DIR", str(tmp_path / "evalws"))
        script = {"planner": happy_script()["planner"], "coder": ["It squares x and adds one."]}
        fake_llm(script)
        result = er.run_task(BY_ID["question_only"], "pipeline", tmp_path)
        assert result["passed"] and result["build_status"] == "no_changes"
