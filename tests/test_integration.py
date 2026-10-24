"""REAL integration tests: real Docker containers and a real LLM API.

Off by default (the normal suite is offline). Run them with:

    python -m pytest --integration                 # everything that can run on this machine
    python -m pytest --integration -m docker       # Docker only (no API key needed)
    python -m pytest --integration -m llm          # LLM only
    python -m pytest --integration -k "not build"  # skip the slow image-build test

Tests skip themselves when Docker is not running or the API key is missing.
The LLM tests send a few short prompts (a few cents at most).
"""

from __future__ import annotations

import os

import pytest

from agent import graph as g
from agent import llm, sandbox
from agent import tools as t
from tests.conftest import git

pytestmark = pytest.mark.integration

KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "google_genai": "GOOGLE_API_KEY", "xai": "XAI_API_KEY"}


@pytest.fixture
def real_ws(tmp_path, monkeypatch):
    """Workspace like `ws`, but WITHOUT faking Docker."""
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    monkeypatch.setattr(t, "WORKSPACE", root)
    t.ensure_workspace(root)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("ALLOW_UNSAFE_LOCAL_EXECUTION", raising=False)
    monkeypatch.delenv("SANDBOX_IMAGE", raising=False)
    return root


@pytest.fixture
def docker(real_ws):
    if not sandbox.docker_available(force=True):
        pytest.skip("Docker is not running (start Docker Desktop) or not installed")
    image = sandbox.get_sandbox_image(real_ws)
    import subprocess

    have = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    if not have:
        pytest.skip(f"image {image} not pulled yet: run `docker pull {image}`")
    return real_ws


@pytest.fixture
def real_llm():
    provider = llm.get_provider()
    var = KEY_VARS.get(provider)
    if var and not os.getenv(var):
        pytest.skip(f"{var} is not set (provider {provider})")
    llm.reset_chat_model_cache()
    return llm.get_chat_model()


# ============================================================================= Docker
@pytest.mark.docker
class TestRealDocker:
    def test_runs_python_and_captures_output(self, docker):
        (docker / "hello.py").write_text("print('hello from the sandbox')\n")
        result = sandbox.run_python_in_sandbox(docker, "hello.py")
        assert result.mode == "docker" and result.ok, str(result)
        assert "hello from the sandbox" in result.stdout

    def test_reports_failures(self, docker):
        (docker / "boom.py").write_text("raise SystemExit(3)\n")
        result = sandbox.run_python_in_sandbox(docker, "boom.py")
        assert result.exit_code == 3 and not result.ok

    def test_no_network(self, docker):
        (docker / "net.py").write_text(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
            "    print('NETWORK OPEN')\n"
            "except OSError:\n"
            "    print('network blocked')\n")
        result = sandbox.run_python_in_sandbox(docker, "net.py")
        assert "network blocked" in result.stdout and "NETWORK OPEN" not in result.stdout, str(result)

    def test_root_filesystem_is_read_only(self, docker):
        (docker / "ro.py").write_text(
            "try:\n"
            "    open('/etc/pwned', 'w').write('x')\n"
            "    print('WRITABLE')\n"
            "except OSError:\n"
            "    print('read-only')\n")
        result = sandbox.run_python_in_sandbox(docker, "ro.py")
        assert "read-only" in result.stdout, str(result)

    def test_infinite_loop_is_killed(self, docker):
        (docker / "loop.py").write_text("while True:\n    pass\n")
        result = sandbox.run_python_in_sandbox(docker, "loop.py", timeout=4)
        assert result.timed_out

    def test_workspace_files_can_be_created_by_the_script(self, docker):
        (docker / "w.py").write_text("open('made_in_sandbox.txt', 'w').write('ok')\n")
        result = sandbox.run_python_in_sandbox(docker, "w.py")
        assert result.ok, str(result)
        assert (docker / "made_in_sandbox.txt").read_text() == "ok"

    @pytest.mark.slow
    def test_build_install_then_run_tests(self, docker):
        """Slow: builds a new image (needs internet inside `docker build`)."""
        out = t.install_packages.invoke({"packages": "pytest"})
        assert out.startswith("Installed pytest"), out
        assert sandbox.get_sandbox_image(docker).startswith("lca-sandbox:")
        (docker / "test_math.py").write_text("def test_add():\n    assert 1 + 1 == 2\n")
        out = t.run_tests.invoke({})
        assert "1 passed" in out, out


# ============================================================================= LLM
@pytest.mark.llm
class TestRealModel:
    def test_the_model_answers(self, real_llm):
        reply = real_llm.invoke("Reply with exactly one word: pong")
        assert "pong" in str(reply.content).lower()

    def test_the_model_can_call_a_tool(self, real_llm):
        bound = real_llm.bind_tools([t.list_files, t.read_file])
        reply = bound.invoke("List the files in the project using the list_files tool.")
        names = [c["name"] for c in reply.tool_calls]
        assert "list_files" in names, (reply.content, reply.tool_calls)

    def test_hi_in_auto_mode_is_just_a_reply(self, real_llm, real_ws):
        graph = g.build_graph("auto", model=real_llm)
        config = g.make_config("it-hi", "auto")
        events = list(g.drive(graph, config, "hi"))
        assert g.pending_tool_calls(graph, config) == []
        assert not [e for e in events if e["kind"] == "tool"], events
        assert any(e["kind"] == "ai" and e["content"].strip() for e in events)
        assert not list(real_ws.glob("*.py"))

    def approve_everything(self, graph, config, text, max_rounds=15):
        events = list(g.drive(graph, config, text))
        approvals: list[str] = []
        for _ in range(max_rounds):
            pending = g.pending_tool_calls(graph, config)
            if not pending:
                break
            approvals += [c["name"] for c in pending]
            events += list(g.drive(graph, config, None))
        return events, approvals

    @pytest.mark.docker
    def test_end_to_end_build_review_commit(self, real_llm, docker):
        """The whole thing with a real model: it may behave a little differently every run,
        so assert on outcomes (file exists, runs, reviewed, committed), not exact wording."""
        graph = g.build_graph("auto", model=real_llm)
        config = g.make_config("it-build", "auto")
        events, approvals = self.approve_everything(
            graph, config,
            "Create hello.py that prints 'hello from the agent'. Review it and commit it. "
            "Do not ask me questions.")
        assert (docker / "hello.py").exists(), [e.get("content", "")[:200] for e in events][-5:]
        assert "write_file" in approvals or "edit_file" in approvals
        status = g.get_run_status(graph, config)
        assert status["status"] in ("committed", "reviewed", "coded"), status
        if status["status"] == "committed":
            assert git(docker, "log", "--oneline").strip(), "status says committed but git has no commit"
        out = sandbox.run_python_in_sandbox(docker, "hello.py")
        assert "hello from the agent" in out.stdout, str(out)
