"""Sandbox command construction, validation, image handling (Docker itself is mocked)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent import sandbox as sb
from tests.conftest import symlink


@pytest.fixture
def root(tmp_path) -> Path:
    r = (tmp_path / "w").resolve()
    r.mkdir()
    return r


def flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class TestDockerCommand:
    def test_locked_down_flags(self, root, monkeypatch):
        monkeypatch.delenv("SANDBOX_USER", raising=False)
        cmd = sb.build_docker_command(root, "img:1", ["-m", "pytest"], "name1")
        assert cmd[:3] == ["docker", "run", "--rm"]
        assert flag_value(cmd, "--network") == "none"
        assert flag_value(cmd, "--memory") == "512m"
        assert flag_value(cmd, "--memory-swap") == "512m"
        assert flag_value(cmd, "--cpus") == "1"
        assert flag_value(cmd, "--pids-limit") == "256"
        assert flag_value(cmd, "--cap-drop") == "ALL"
        assert flag_value(cmd, "--security-opt") == "no-new-privileges"
        assert "--read-only" in cmd
        assert flag_value(cmd, "--tmpfs") == "/tmp:rw,size=128m"
        assert "HOME=/tmp" in cmd and "PYTHONDONTWRITEBYTECODE=1" in cmd
        assert f"{root}:/workspace" in cmd
        assert flag_value(cmd, "-w") == "/workspace"
        assert flag_value(cmd, "--name") == "name1"
        assert cmd[-4:] == ["img:1", "python", "-m", "pytest"]  # image, then command

    def test_runs_as_non_root(self, root, monkeypatch):
        monkeypatch.delenv("SANDBOX_USER", raising=False)
        cmd = sb.build_docker_command(root, "img", ["x.py"], "n")
        if hasattr(__import__("os"), "getuid"):
            user = flag_value(cmd, "--user")
            assert user.split(":")[0] != "0"

    def test_root_host_falls_back_to_nobody(self, root, monkeypatch):
        monkeypatch.delenv("SANDBOX_USER", raising=False)
        monkeypatch.setattr(sb.os, "getuid", lambda: 0, raising=False)
        monkeypatch.setattr(sb.os, "getgid", lambda: 0, raising=False)
        assert flag_value(sb.build_docker_command(root, "i", ["a"], "n"), "--user") == "65534:65534"

    def test_user_override_is_validated(self, root, monkeypatch):
        monkeypatch.setenv("SANDBOX_USER", "1234:5678")
        assert flag_value(sb.build_docker_command(root, "i", ["a"], "n"), "--user") == "1234:5678"
        monkeypatch.setenv("SANDBOX_USER", "root; rm -rf /")
        assert "--user" not in sb.build_docker_command(root, "i", ["a"], "n")

    def test_git_and_agent_dirs_mounted_read_only(self, root):
        plain = sb.build_docker_command(root, "i", ["a"], "n")
        assert not any(arg.endswith(":ro") for arg in plain)
        (root / ".git").mkdir()
        (root / ".agent").mkdir()
        cmd = sb.build_docker_command(root, "i", ["a"], "n")
        assert f"{root / '.git'}:/workspace/.git:ro" in cmd
        assert f"{root / '.agent'}:/workspace/.agent:ro" in cmd


class TestArgValidation:
    def test_allowed_forms(self, root):
        (root / "a.py").write_text("")
        assert sb.validate_python_args(root, ["-m", "pytest", "-q"]) == ["-m", "pytest", "-q"]
        assert sb.validate_python_args(root, ["a.py", "x"]) == ["./a.py", "x"]
        assert sb.validate_python_args(root, ["--version"]) == ["--version"]

    @pytest.mark.parametrize("args", [[], ["-c", "print(1)"], ["-"], ["-m"], ["-m", "a b"],
                                      ["-m", "../x"], ["-X", "dev"], ["../x.py"], ["a\x00.py"]])
    def test_rejected_forms(self, root, args):
        with pytest.raises(ValueError):
            sb.validate_python_args(root, args)

    def test_non_py_and_missing_files_rejected(self, root):
        (root / "a.txt").write_text("")
        for name in ("a.txt", "missing.py", "/etc/passwd"):
            with pytest.raises(ValueError):
                sb.validate_python_args(root, [name])

    def test_run_python_in_sandbox_validates_before_running(self, root, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
        (root / "a.txt").write_text("")
        for target in ("a.txt", "nope.py", "../evil.py", "/etc/hosts"):
            result = sb.run_python_in_sandbox(root, target)
            assert str(result).startswith("ERROR"), target

    def test_symlink_to_outside_py_rejected(self, root, tmp_path, monkeypatch):
        outside = tmp_path / "evil.py"
        outside.write_text("print('pwn')")
        symlink(outside, root / "link.py")
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
        assert str(sb.run_python_in_sandbox(root, "link.py")).startswith("ERROR")


class TestExecutionDispatch:
    def test_docker_run_is_used_when_available(self, root, monkeypatch):
        (root / "a.py").write_text("print(1)")
        calls = []
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, **kw: calls.append((cmd, kw)) or subprocess.CompletedProcess(cmd, 0, "1\n", ""))
        result = sb.run_python_in_sandbox(root, "a.py")
        cmd, kwargs = calls[0]
        assert cmd[:2] == ["docker", "run"] and cmd[-2:] == ["python", "./a.py"]
        assert kwargs["timeout"] == sb.execution_timeout() and "shell" not in kwargs
        assert result.ok and "exit code: 0" in str(result) and "1" in str(result)

    def test_timeout_removes_the_container(self, root, monkeypatch):
        (root / "a.py").write_text("while True: pass")
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[:2] == ["docker", "run"]:
                raise subprocess.TimeoutExpired(cmd, 1, output=b"partial", stderr=b"")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = sb.run_python_in_sandbox(root, "a.py", timeout=1)
        assert result.timed_out and "TIMED OUT" in str(result) and "partial" in str(result)
        run_cmd = calls[0]
        name = run_cmd[run_cmd.index("--name") + 1]
        assert ["docker", "rm", "-f", name] in calls

    def test_no_docker_means_no_execution(self, root, monkeypatch):
        (root / "a.py").write_text("print('ran')")
        monkeypatch.setattr(sb, "docker_available", lambda force=False: False)
        monkeypatch.delenv("ALLOW_UNSAFE_LOCAL_EXECUTION", raising=False)
        out = str(sb.run_python_in_sandbox(root, "a.py"))
        assert out.startswith("ERROR") and "ran" not in out.replace("Docker", "")

    def test_unsafe_local_fallback_needs_explicit_opt_in(self, root, monkeypatch):
        (root / "a.py").write_text("import os\nprint('ran', 'SECRET_KEY' in os.environ)")
        monkeypatch.setattr(sb, "docker_available", lambda force=False: False)
        monkeypatch.setenv("SECRET_KEY", "shh")
        monkeypatch.setenv("ALLOW_UNSAFE_LOCAL_EXECUTION", "true")
        result = sb.run_python_in_sandbox(root, "a.py")
        assert result.ok and "ran False" in result.stdout  # secrets are not inherited
        assert "UNSAFE" in str(result)

    def test_local_args_and_failure_exit_code(self, root, monkeypatch):
        (root / "t.py").write_text("import sys; print(sys.argv[1:]); sys.exit(3)")
        monkeypatch.setattr(sb, "docker_available", lambda force=False: False)
        monkeypatch.setenv("ALLOW_UNSAFE_LOCAL_EXECUTION", "true")
        result = sb.run_python_args_in_sandbox(root, ["t.py", "a", "b"])
        assert result.exit_code == 3 and "['a', 'b']" in result.stdout

    def test_local_timeout(self, root, monkeypatch):
        (root / "s.py").write_text("import time; time.sleep(30)")
        monkeypatch.setattr(sb, "docker_available", lambda force=False: False)
        monkeypatch.setenv("ALLOW_UNSAFE_LOCAL_EXECUTION", "true")
        assert sb.run_python_in_sandbox(root, "s.py", timeout=1).timed_out


class TestImageSelection:
    def test_default_then_env_then_file(self, root, monkeypatch):
        monkeypatch.delenv("SANDBOX_IMAGE", raising=False)
        assert sb.get_sandbox_image(root) == "python:3.12-slim"
        monkeypatch.setenv("SANDBOX_IMAGE", "python:3.11-slim")
        assert sb.get_sandbox_image(root) == "python:3.11-slim"
        (root / ".agent").mkdir()
        (root / ".agent" / "sandbox_image.txt").write_text("lca-sandbox:abc123\n")
        assert sb.get_sandbox_image(root) == "lca-sandbox:abc123"

    def test_invalid_values_are_ignored(self, root, monkeypatch):
        (root / ".agent").mkdir()
        (root / ".agent" / "sandbox_image.txt").write_text("bad image; rm -rf /\n")
        monkeypatch.setenv("SANDBOX_IMAGE", "also bad\nFROM evil")
        assert sb.get_sandbox_image(root) == sb.DEFAULT_IMAGE

    def test_timeouts_from_env(self, monkeypatch):
        monkeypatch.setenv("EXECUTION_TIMEOUT", "7")
        monkeypatch.setenv("TEST_TIMEOUT", "oops")
        assert sb.execution_timeout() == 7 and sb.test_timeout() == 180


class TestInstallPackages:
    @pytest.mark.parametrize("good", ["pytest", "requests>=2.31", "numpy==1.26.4", "pkg[extra,more]>=1,<2",
                                      "a", "my-pkg_name.v2", "x~=1.4"])
    def test_valid_requirements(self, good):
        assert sb.validate_requirements([good]) == [good]

    @pytest.mark.parametrize("bad", ["", "--index-url=http://evil", "-r req.txt", "git+https://x/y.git",
                                     "http://evil/pkg.whl", "./local", "../x", "pkg @ http://x", "a;b",
                                     "a b", "a$(id)", "a`id`", "pkg; python_version<'3'", "*", "pkg\n--evil"])
    def test_invalid_requirements(self, bad):
        with pytest.raises(ValueError):
            sb.validate_requirements([bad] if bad else [])

    def test_too_many(self):
        with pytest.raises(ValueError):
            sb.validate_requirements([f"p{i}" for i in range(sb.MAX_PACKAGES + 1)])

    def test_invalid_input_never_reaches_docker(self, root, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
        assert sb.install_packages_into_sandbox_image(root, ["--evil"]).startswith("ERROR")

    def test_requires_docker(self, root, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: False)
        assert "Docker is not available" in sb.install_packages_into_sandbox_image(root, ["pytest"])
        assert not (root / ".agent" / "sandbox_image.txt").exists()

    def test_builds_new_image_and_switches_to_it(self, root, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        monkeypatch.delenv("SANDBOX_IMAGE", raising=False)
        seen = {}

        def fake_run(cmd, **kw):
            ctx = Path(cmd[-1])
            seen["cmd"], seen["kw"] = cmd, kw
            seen["files"] = sorted(p.name for p in ctx.iterdir())
            seen["packages"] = json.loads((ctx / "packages.json").read_text())
            seen["dockerfile"] = (ctx / "Dockerfile").read_text()
            seen["install"] = (ctx / "install.py").read_text()
            return subprocess.CompletedProcess(cmd, 0, "built", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        msg = sb.install_packages_into_sandbox_image(root, ["pytest", "requests>=2.0"])
        cmd = seen["cmd"]
        assert cmd[:2] == ["docker", "build"] and "shell" not in seen["kw"]
        assert "BASE_IMAGE=python:3.12-slim" in cmd
        tag = cmd[cmd.index("-t") + 1]
        assert tag.startswith("lca-sandbox:") and tag in msg
        assert seen["files"] == ["Dockerfile", "install.py", "packages.json", "source_builds.json"]
        assert seen["packages"] == ["pytest", "requests>=2.0"]
        assert "FROM ${BASE_IMAGE}" in seen["dockerfile"] and "install.py" in seen["dockerfile"]
        assert "shell=True" not in seen["install"] and "check_call" in seen["install"]
        assert "--only-binary=:all:" in seen["install"]  # no setup.py runs during the build
        # The new tag becomes the active image for every later run...
        assert (root / ".agent" / "sandbox_image.txt").read_text().strip() == tag
        assert sb.get_sandbox_image(root) == tag
        assert sb.build_docker_command(root, sb.get_sandbox_image(root), ["x.py"], "n")[-3] == tag
        # ...and a second install builds on top of it.
        sb.install_packages_into_sandbox_image(root, ["rich"])
        assert f"BASE_IMAGE={tag}" in seen["cmd"]

    def test_build_failure_keeps_current_image(self, root, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)
        monkeypatch.setattr(subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "no such package"))
        out = sb.install_packages_into_sandbox_image(root, ["doesnotexist"])
        assert out.startswith("ERROR") and "no such package" in out
        assert not (root / ".agent" / "sandbox_image.txt").exists()

    def test_build_timeout(self, root, monkeypatch):
        monkeypatch.setattr(sb, "docker_available", lambda force=False: True)

        def boom(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 1)

        monkeypatch.setattr(subprocess, "run", boom)
        assert "timed out" in sb.install_packages_into_sandbox_image(root, ["pytest"])


def test_docker_available_is_false_without_cli(monkeypatch):
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    assert sb.docker_available(force=True) is False
    assert sys.version_info >= (3, 9)