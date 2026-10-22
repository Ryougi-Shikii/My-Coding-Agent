"""File, git, planning/memory, review tools and the approval policy."""

from __future__ import annotations

import os

import pytest

from agent import sandbox
from agent import tools as t
from tests.conftest import git
from tests.fakes import ScriptedChatModel


def run(tool, **args):
    return tool.invoke(args)


# --------------------------------------------------------------------------- path safety
class TestPathSafety:
    @pytest.mark.parametrize("bad", ["../escape.txt", "a/../../escape.txt", "/etc/passwd"])
    def test_outside_paths_rejected(self, ws, bad):
        with pytest.raises(t.PathError):
            t.safe_path(bad)
        assert run(t.read_file, path=bad).startswith("ERROR")
        assert run(t.write_file, path=bad, content="x").startswith("ERROR")

    def test_absolute_path_inside_workspace_is_fine(self, ws):
        assert t.safe_path(str(ws / "a.txt")) == ws / "a.txt"

    def test_symlink_escape_rejected(self, ws, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("top secret")
        os.symlink(outside, ws / "link.txt")
        assert run(t.read_file, path="link.txt").startswith("ERROR")
        assert run(t.write_file, path="link.txt", content="pwn").startswith("ERROR")
        assert outside.read_text() == "top secret"

    def test_symlinked_directory_escape_rejected(self, ws, tmp_path):
        outside = tmp_path / "outdir"
        outside.mkdir()
        os.symlink(outside, ws / "d")
        assert run(t.write_file, path="d/x.txt", content="pwn").startswith("ERROR")
        assert not (outside / "x.txt").exists()

    def test_nul_byte_rejected(self, ws):
        assert run(t.read_file, path="a\x00b").startswith("ERROR")

    def test_git_dir_and_image_pointer_protected(self, git_ready):
        assert "protected" in run(t.write_file, path=".git/config", content="x")
        assert "protected" in run(t.write_file, path="sub/.git/hooks/pre-commit", content="x")
        assert "cannot be edited" in run(t.write_file, path=".agent/sandbox_image.txt", content="evil")
        assert not (git_ready / ".agent" / "sandbox_image.txt").exists()
        assert "protected" in run(t.create_directory, path=".git/x")
        assert "protected" in run(t.edit_file, path=".git/config", find="a", replace="b")


# --------------------------------------------------------------------------- file tools
class TestFileTools:
    def test_write_read_roundtrip_creates_parents(self, ws):
        out = run(t.write_file, path="pkg/mod.py", content="x = 1\n")
        assert out.startswith("Wrote")
        assert run(t.read_file, path="pkg/mod.py") == "x = 1\n"

    def test_read_missing_and_directory(self, ws):
        assert run(t.read_file, path="nope.txt").startswith("ERROR")
        (ws / "d").mkdir()
        assert run(t.read_file, path="d").startswith("ERROR")

    def test_edit_file_semantics(self, ws):
        (ws / "f.txt").write_text("a b a")
        assert run(t.edit_file, path="f.txt", find="zzz", replace="y").startswith("ERROR")
        assert "matches 2" in run(t.edit_file, path="f.txt", find="a", replace="y")
        assert (ws / "f.txt").read_text() == "a b a"  # untouched after the failed edit
        assert "Replaced 2" in run(t.edit_file, path="f.txt", find="a", replace="y", replace_all=True)
        assert (ws / "f.txt").read_text() == "y b y"
        assert "Replaced 1" in run(t.edit_file, path="f.txt", find="b", replace="c")
        assert run(t.edit_file, path="f.txt", find="", replace="c").startswith("ERROR")

    def test_create_directory_and_list_files(self, ws):
        run(t.create_directory, path="src/pkg")
        run(t.write_file, path="src/pkg/a.py", content="")
        (ws / "__pycache__").mkdir()
        listing = run(t.list_files).splitlines()
        assert "src/" in listing and "src/pkg/" in listing and "src/pkg/a.py" in listing
        assert not any("__pycache__" in line for line in listing)
        assert run(t.list_files, path="src") == "src/pkg/\nsrc/pkg/a.py"
        assert run(t.list_files, path="missing").startswith("ERROR")

    def test_search_text(self, ws):
        run(t.write_file, path="a.py", content="def Hello():\n    pass\n")
        run(t.write_file, path="b.txt", content="nothing here\nhello again\n")
        (ws / "bin.dat").write_bytes(b"\x00\x01hello")
        hits = run(t.search_text, query="HELLO").splitlines()
        assert "a.py:1: def Hello():" in hits and "b.txt:2: hello again" in hits
        assert not any("bin.dat" in h for h in hits)
        assert run(t.search_text, query="zzz") == "No matches."
        assert run(t.search_text, query="hello", path="b.txt") == "b.txt:2: hello again"
        assert run(t.search_text, query="").startswith("ERROR")

    def test_check_python_syntax(self, ws):
        run(t.write_file, path="ok.py", content="x = 1\n")
        run(t.write_file, path="sub/bad.py", content="def f(:\n")
        assert run(t.check_python_syntax, path="ok.py").startswith("Syntax OK")
        bad = run(t.check_python_syntax, path="sub/bad.py")
        assert bad.startswith("ERROR") and "sub/bad.py:1" in bad
        assert run(t.check_python_syntax, path=".").startswith("ERROR")  # finds the bad file
        assert run(t.check_python_syntax, path="nope.py").startswith("ERROR")


# --------------------------------------------------------------------------- planning / memory
class TestAgentState:
    def test_plan_todo_memory_and_state(self, ws):
        assert "(empty)" in run(t.read_agent_state)
        run(t.update_plan, plan_markdown="# Plan\nstep")
        run(t.update_todo, todo_markdown="- [ ] one")
        run(t.save_memory, note="uses stdlib only")
        run(t.save_memory, note="second")
        assert (ws / ".agent" / "PLAN.md").read_text() == "# Plan\nstep\n"
        assert (ws / ".agent" / "TODO.md").read_text() == "- [ ] one\n"
        memory = (ws / ".agent" / "MEMORY.md").read_text()
        assert memory.count("- [") == 2 and "uses stdlib only" in memory
        state = run(t.read_agent_state)
        assert "## PLAN.md\n# Plan" in state and "- [ ] one" in state and "second" in state

    def test_update_plan_overwrites(self, ws):
        run(t.update_plan, plan_markdown="old")
        run(t.update_plan, plan_markdown="new")
        assert (ws / ".agent" / "PLAN.md").read_text() == "new\n"

    @pytest.mark.parametrize("tool,arg", [(t.update_plan, "plan_markdown"),
                                          (t.update_todo, "todo_markdown"),
                                          (t.save_memory, "note")])
    def test_empty_input_rejected(self, ws, tool, arg):
        assert run(tool, **{arg: "  "}).startswith("ERROR")


# --------------------------------------------------------------------------- git
class TestGit:
    def test_not_a_repo_messages(self, ws):
        for tool in (t.git_status, t.git_diff, t.git_log):
            assert "not a git repository" in tool.invoke({})
        assert "not a git repository" in run(t.git_add, path=".")
        assert "not a git repository" in run(t.git_commit, message="m")

    def test_does_not_touch_parent_repository(self, ws, tmp_path):
        # The workspace sits inside another repo, like workspace/ inside this project.
        os.system(f"git -C {tmp_path} init -q")
        assert "not a git repository" in t.git_status.invoke({})
        assert not (ws / ".git").exists()

    def test_full_cycle_with_identity_fallback(self, ws):
        assert not t.git_init.invoke({}).startswith("ERROR")
        assert (ws / ".git").is_dir()
        assert (ws / ".gitignore").read_text().startswith(".agent/")
        assert "Repository already initialised" in t.git_init.invoke({})
        assert t.git_log.invoke({}) == "No commits yet."
        run(t.write_file, path="a.py", content="print(1)\n")
        assert "a.py" in t.git_status.invoke({})
        assert run(t.git_commit, message="nothing staged").startswith("ERROR")
        assert not run(t.git_add, path="a.py").startswith("ERROR")
        assert "A  a.py" in git(ws, "status", "--short")
        out = run(t.git_commit, message="Add a.py")
        assert not out.startswith("ERROR"), out
        assert "Add a.py" in t.git_log.invoke({})
        assert git(ws, "config", "--local", "user.name").strip() == "Coding Agent"
        run(t.edit_file, path="a.py", find="1", replace="2")
        assert "+print(2)" in t.git_diff.invoke({})

    def test_agent_state_is_gitignored(self, git_ready):
        run(t.update_plan, plan_markdown="x")
        run(t.git_add, path=".")
        assert ".agent" not in git(git_ready, "status", "--short")

    def test_hooks_are_disabled(self, git_ready):
        hook = git_ready / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\ntouch hook-ran\nexit 1\n")
        hook.chmod(0o755)
        run(t.write_file, path="a.txt", content="x")
        run(t.git_add, path=".")
        assert not run(t.git_commit, message="m").startswith("ERROR")
        assert not (git_ready / "hook-ran").exists()

    def test_commit_message_validation(self, git_ready):
        assert run(t.git_commit, message="   ").startswith("ERROR")
        assert run(t.git_commit, message="x" * 6000).startswith("ERROR")

    def test_cannot_add_dot_git(self, git_ready):
        assert run(t.git_add, path=".git").startswith("ERROR")


# --------------------------------------------------------------------------- review
class TestReview:
    def test_verdict_parsing_fails_closed(self):
        assert t.parse_verdict("ok\nVERDICT: APPROVE") == "APPROVE"
        assert t.parse_verdict("verdict: request_changes") == "REQUEST_CHANGES"
        assert t.parse_verdict("VERDICT: REQUEST_CHANGES\n...\nVERDICT: APPROVE") == "APPROVE"
        assert t.parse_verdict("looks fine to me") == "REQUEST_CHANGES"
        assert t.parse_verdict("") == "REQUEST_CHANGES"

    def test_requires_repo(self, ws):
        assert "not a git repository" in t.review_changes.invoke({})

    def test_no_changes(self, git_ready):
        # git_init created a .gitignore (an untracked change); commit it so the tree is clean.
        run(t.git_add, path=".")
        assert not run(t.git_commit, message="init").startswith("ERROR")
        assert t.review_changes.invoke({}).endswith("VERDICT: NO_CHANGES")

    def test_review_sees_new_and_modified_files_and_is_saved(self, git_ready, monkeypatch):
        seen = {}

        def responder(key, messages):
            seen["request"] = str(messages[-1].content)
            return "Fine overall.\nVERDICT: APPROVE"

        monkeypatch.setattr(t, "get_chat_model", lambda: ScriptedChatModel(responder=responder))
        run(t.write_file, path="new.py", content="print('brand new')\n")
        review = t.review_changes.invoke({})
        assert t.parse_verdict(review) == "APPROVE"
        assert t.parse_reviewed_tree(review) == t.tree_fingerprint()
        assert "brand new" in seen["request"] and "new.py" in seen["request"]
        saved = list((git_ready / ".agent" / "reviews").glob("review_*.md"))
        assert len(saved) == 1 and "VERDICT: APPROVE" in saved[0].read_text()
        t.review_changes.invoke({})  # same second: must not overwrite the first file
        assert len(list((git_ready / ".agent" / "reviews").glob("review_*.md"))) == 2

    def test_model_without_verdict_means_request_changes(self, git_ready, monkeypatch):
        monkeypatch.setattr(t, "get_chat_model",
                            lambda: ScriptedChatModel(responder=lambda k, m: "no verdict here"))
        run(t.write_file, path="a.py", content="x=1\n")
        assert t.parse_verdict(t.review_changes.invoke({})) == "REQUEST_CHANGES"

    def test_model_failure_is_reported_not_raised(self, git_ready, monkeypatch):
        def boom(key, messages):
            raise RuntimeError("api down")

        monkeypatch.setattr(t, "get_chat_model", lambda: ScriptedChatModel(responder=boom))
        run(t.write_file, path="a.py", content="x=1\n")
        assert t.review_changes.invoke({}).startswith("ERROR: review model call failed")


# --------------------------------------------------------------------------- execution tools
class TestExecutionTools:
    def test_run_tests_uses_pytest_in_sandbox(self, ws, monkeypatch):
        captured = {}

        def fake_run(root, args, timeout=None):
            captured.update(root=root, args=list(args), timeout=timeout)
            return sandbox.RunResult("docker", "img", 1, "", "No module named pytest")

        monkeypatch.setattr(sandbox, "run_python_args_in_sandbox", fake_run)
        out = run(t.run_tests, path="tests")
        assert captured["args"][:5] == ["-m", "pytest", "-o", "cache_dir=/tmp/pytest_cache", "-q"]
        assert captured["args"][-1] == "./tests" and captured["root"] == ws
        assert captured["timeout"] == sandbox.test_timeout()
        assert "install_packages('pytest')" in out

    def test_run_tests_rejects_outside_path(self, ws):
        assert run(t.run_tests, path="../x").startswith("ERROR")

    def test_install_packages_splits_and_delegates(self, ws, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            sandbox, "install_packages_into_sandbox_image",
            lambda root, pkgs, timeout=None: captured.update(root=root, pkgs=list(pkgs)) or "ok")
        assert run(t.install_packages, packages="pytest  requests>=2.0 ") == "ok"
        assert captured == {"root": ws, "pkgs": ["pytest", "requests>=2.0"]}

    def test_run_python_file_without_docker_refuses(self, ws):
        run(t.write_file, path="a.py", content="print(1)")
        out = run(t.run_python_file, path="a.py")
        assert out.startswith("ERROR") and "Docker is not available" in out


# --------------------------------------------------------------------------- policy
class TestApprovalPolicy:
    def test_every_tool_is_classified_exactly_once(self):
        assert t.SAFE_TOOLS | t.DANGEROUS_TOOLS == set(t.TOOLS_BY_NAME)
        assert not t.SAFE_TOOLS & t.DANGEROUS_TOOLS

    @pytest.mark.parametrize("name", ["write_file", "edit_file", "create_directory",
                                      "run_python_file", "run_tests", "install_packages",
                                      "git_commit"])
    def test_dangerous_tools_need_approval(self, name):
        assert t.requires_approval(name)

    @pytest.mark.parametrize("name", ["list_files", "read_file", "search_text",
                                      "check_python_syntax", "git_status", "git_diff", "git_log",
                                      "read_agent_state", "review_changes"])
    def test_safe_tools_are_auto_approved(self, name):
        assert not t.requires_approval(name)

    def test_unknown_tool_fails_closed(self):
        assert t.requires_approval("rm_rf_everything")

    def test_all_mode(self, monkeypatch):
        monkeypatch.setenv("APPROVAL_MODE", "all")
        assert t.requires_approval("read_file")
        monkeypatch.setenv("APPROVAL_MODE", "bogus")
        assert not t.requires_approval("read_file")  # invalid value falls back to the default

    def test_stage_allow_lists(self):
        everything = set(t.TOOLS_BY_NAME)
        assert set(t.STAGE_TOOLS) == set(t.STAGES) == {"planner", "coder", "reviewer", "committer"}
        mutating = {"write_file", "edit_file", "create_directory", "install_packages"}
        assert not mutating & set(t.STAGE_TOOLS["planner"])
        assert not mutating & set(t.STAGE_TOOLS["reviewer"])
        assert not mutating & set(t.STAGE_TOOLS["committer"])
        assert "git_commit" in t.STAGE_TOOLS["committer"]
        for stage in ("planner", "coder", "reviewer"):
            assert "git_commit" not in t.STAGE_TOOLS[stage]
        assert "review_changes" in t.STAGE_TOOLS["reviewer"]
        assert "review_changes" not in t.STAGE_TOOLS["coder"]
        assert "update_plan" in t.STAGE_TOOLS["planner"] and "update_plan" not in t.STAGE_TOOLS["coder"]
        assert set().union(*t.STAGE_TOOLS.values()) <= everything

    def test_workspace_top_level(self, ws):
        (ws / "z.txt").write_text("")
        (ws / "adir").mkdir()
        names = t.workspace_top_level()
        assert [n for n, is_dir in names if is_dir] == [".agent", "adir"]
        assert names[-1] == ("z.txt", False)  # directories first, then files
