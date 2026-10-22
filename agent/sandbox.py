"""Sandboxed Python execution.

Code the agent writes is executed inside a locked-down Docker container (no network,
read-only root filesystem, dropped capabilities, resource limits). A local, unsafe
fallback exists only when ALLOW_UNSAFE_LOCAL_EXECUTION=true.

Packages are never installed on the host. ``install_packages_into_sandbox_image`` builds a
NEW Docker image on top of the current sandbox image and records its tag in
``<workspace>/.agent/sandbox_image.txt`` so that later runs use it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_IMAGE = "python:3.12-slim"
IMAGE_FILE_PARTS = (".agent", "sandbox_image.txt")
MAX_OUTPUT_CHARS = 20_000
MAX_PACKAGES = 20

# Docker image reference: conservative character set, no whitespace or newlines.
_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}$")

# One requirement: name, optional [extras], optional comma-separated version specifiers.
# No URLs, paths, '@' direct references, environment markers or leading '-' (pip options).
_SPEC = r"(?:===|==|>=|<=|~=|!=|>|<)[A-Za-z0-9.*+!_-]+"
_REQ_RE = re.compile(
    rf"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    rf"(?:\[[A-Za-z0-9._,-]+\])?"
    rf"(?:{_SPEC}(?:,{_SPEC})*)?$"
)

_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_USER_RE = re.compile(r"^\d+(?::\d+)?$")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
        return value if value > 0 else default
    except ValueError:
        return default


def execution_timeout() -> int:
    return _env_int("EXECUTION_TIMEOUT", 60)


def test_timeout() -> int:
    return _env_int("TEST_TIMEOUT", 180)


def build_timeout() -> int:
    return _env_int("BUILD_TIMEOUT", 600)


def unsafe_local_allowed() -> bool:
    return os.getenv("ALLOW_UNSAFE_LOCAL_EXECUTION", "false").strip().lower() == "true"


# ---------------------------------------------------------------------------
# Docker detection and image selection
# ---------------------------------------------------------------------------
_docker_checked_at: float | None = None
_docker_ok = False


def docker_available(force: bool = False) -> bool:
    """True if the docker CLI exists and the daemon answers. Cached for 20 seconds."""
    global _docker_checked_at, _docker_ok
    now = time.monotonic()
    if not force and _docker_checked_at is not None and now - _docker_checked_at < 20:
        return _docker_ok
    ok = False
    if shutil.which("docker"):
        try:
            ok = subprocess.run(["docker", "info"], capture_output=True, timeout=8).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
    _docker_checked_at, _docker_ok = now, ok
    return ok


def get_sandbox_image(workspace_root: str | os.PathLike) -> str:
    """Image used for execution.

    Order: workspace/.agent/sandbox_image.txt, then $SANDBOX_IMAGE, then python:3.12-slim.
    """
    image_file = Path(workspace_root).joinpath(*IMAGE_FILE_PARTS)
    if image_file.is_file():
        tag = image_file.read_text(encoding="utf-8", errors="replace").strip()
        if tag and _IMAGE_RE.match(tag):
            return tag
    env_image = os.getenv("SANDBOX_IMAGE", "").strip()
    if env_image and _IMAGE_RE.match(env_image):
        return env_image
    return DEFAULT_IMAGE


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@dataclass
class RunResult:
    mode: str  # "docker", "local" or "none"
    image: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.timed_out and self.exit_code == 0

    def __str__(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        where = f"docker image={self.image}" if self.mode == "docker" else "UNSAFE local python"
        if self.timed_out:
            head = f"[{where}] TIMED OUT"
        else:
            head = f"[{where}] exit code: {self.exit_code}"
        parts = [head]
        if self.stdout.strip():
            parts.append("--- stdout ---\n" + _truncate(self.stdout.rstrip()))
        if self.stderr.strip():
            parts.append("--- stderr ---\n" + _truncate(self.stderr.rstrip()))
        if not self.stdout.strip() and not self.stderr.strip():
            parts.append("(no output)")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _resolve_inside(workspace_root: Path, file_path: str) -> Path:
    root = Path(workspace_root).resolve()
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(f"path '{file_path}' is outside the workspace")
    return resolved


def _container_path(workspace_root: Path, resolved: Path) -> str:
    """Path relative to /workspace, prefixed with ./ so it can never look like an option."""
    return "./" + resolved.relative_to(Path(workspace_root).resolve()).as_posix()


def validate_python_args(workspace_root: Path, python_args: Sequence[str]) -> list[str]:
    """Allow `-m <module> ...`, `--version`, or a workspace `.py` file; reject anything else."""
    args = [str(a) for a in python_args]
    if not args:
        raise ValueError("no python arguments given")
    if any("\x00" in a for a in args):
        raise ValueError("NUL byte in arguments")
    first = args[0]
    if first == "-m":
        if len(args) < 2 or not _MODULE_RE.match(args[1]):
            raise ValueError("'-m' must be followed by a module name")
        return args
    if first in ("--version", "-V"):
        return args[:1]
    if first.startswith("-"):
        raise ValueError(f"python option '{first}' is not allowed (use '-m module' or a .py file)")
    resolved = _resolve_inside(workspace_root, first)
    if resolved.suffix != ".py":
        raise ValueError("only .py files can be run")
    if not resolved.is_file():
        raise ValueError(f"file not found: {first}")
    return [_container_path(workspace_root, resolved), *args[1:]]


# ---------------------------------------------------------------------------
# Docker execution
# ---------------------------------------------------------------------------
def _sandbox_user() -> str | None:
    """User to run as inside the container (non-root where possible)."""
    override = os.getenv("SANDBOX_USER", "").strip()
    if override:
        return override if _USER_RE.match(override) else None
    if hasattr(os, "getuid"):
        uid, gid = os.getuid(), os.getgid()
        # Never run as root in the container; 65534 is 'nobody'.
        return "65534:65534" if uid == 0 else f"{uid}:{gid}"
    return None  # e.g. Windows hosts: fall back to the image's default user


def build_docker_command(
    workspace_root: str | os.PathLike,
    image: str,
    python_args: Sequence[str],
    container_name: str,
) -> list[str]:
    """Build the locked-down `docker run` command (pure function, easy to test)."""
    root = Path(workspace_root).resolve()
    cmd = [
        "docker", "run",
        "--rm",
        "--name", container_name,
        "--network", "none",
        "--memory", "512m",
        "--memory-swap", "512m",
        "--cpus", "1",
        "--pids-limit", "256",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=128m",
        "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{root}:/workspace",
    ]
    # The agent must not be able to tamper with git internals or its own control files
    # from inside the sandbox, so mount them read-only on top of the writable workspace.
    for protected in (".git", ".agent"):
        if (root / protected).is_dir():
            cmd += ["-v", f"{root / protected}:/workspace/{protected}:ro"]
    cmd += ["-w", "/workspace"]
    user = _sandbox_user()
    if user:
        cmd += ["--user", user]
    cmd += [image, "python", *python_args]
    return cmd


def _run_in_docker(workspace_root: Path, python_args: list[str], timeout: int) -> RunResult:
    image = get_sandbox_image(workspace_root)
    name = f"lca-run-{uuid.uuid4().hex[:12]}"
    cmd = build_docker_command(workspace_root, image, python_args, name)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # Killing the docker client does not stop the container: remove it explicitly.
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            pass
        return RunResult("docker", image, None, _to_text(exc.stdout), _to_text(exc.stderr),
                         timed_out=True)
    except OSError as exc:
        return RunResult("docker", image, error=f"could not start docker: {exc}")
    return RunResult("docker", image, proc.returncode, proc.stdout, proc.stderr)


def _run_locally(workspace_root: Path, python_args: list[str], timeout: int) -> RunResult:
    """UNSAFE: runs on the host. Only reachable with ALLOW_UNSAFE_LOCAL_EXECUTION=true."""
    # Minimal environment so API keys from the host are not exposed to agent-written code.
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": str(workspace_root)}
    if os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    # Local runs see host paths, so `./file.py` relative to the workspace cwd is right.
    try:
        proc = subprocess.run(
            [sys.executable, *python_args], cwd=workspace_root, env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult("local", "", None, _to_text(exc.stdout), _to_text(exc.stderr),
                         timed_out=True)
    except OSError as exc:
        return RunResult("local", "", error=f"could not start python: {exc}")
    return RunResult("local", "", proc.returncode, proc.stdout, proc.stderr)


def _execute(workspace_root: Path, python_args: list[str], timeout: int) -> RunResult:
    if docker_available():
        return _run_in_docker(workspace_root, python_args, timeout)
    if unsafe_local_allowed():
        return _run_locally(workspace_root, python_args, timeout)
    return RunResult(
        "none",
        error=(
            "Docker is not available, so nothing was executed. Start Docker (and pull the "
            "sandbox image), or set ALLOW_UNSAFE_LOCAL_EXECUTION=true to run directly on "
            "this machine (NOT sandboxed)."
        ),
    )


def run_python_in_sandbox(
    workspace_root: str | os.PathLike,
    file_path: str,
    timeout: int | None = None,
    script_args: Sequence[str] = (),
) -> RunResult:
    """Run one workspace .py file in the sandbox."""
    root = Path(workspace_root).resolve()
    try:
        resolved = _resolve_inside(root, file_path)
        if resolved.suffix != ".py":
            raise ValueError("only .py files can be run")
        if not resolved.is_file():
            raise ValueError(f"file not found: {file_path}")
    except ValueError as exc:
        return RunResult("none", error=str(exc))
    args = [_container_path(root, resolved), *[str(a) for a in script_args]]
    return _execute(root, args, timeout or execution_timeout())


def run_python_args_in_sandbox(
    workspace_root: str | os.PathLike,
    python_args: Sequence[str],
    timeout: int | None = None,
) -> RunResult:
    """Run `python <args>` in the sandbox, e.g. ['-m', 'pytest', '-q']."""
    root = Path(workspace_root).resolve()
    try:
        args = validate_python_args(root, python_args)
    except ValueError as exc:
        return RunResult("none", error=str(exc))
    return _execute(root, args, timeout or execution_timeout())


# ---------------------------------------------------------------------------
# Package installation = build a new image
# ---------------------------------------------------------------------------
_INSTALL_SCRIPT = '''\
"""Runs inside `docker build`: installs the packages listed in packages.json."""
import json
import subprocess
import sys
from pathlib import Path

packages = json.loads(Path(__file__).with_name("packages.json").read_text(encoding="utf-8"))
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir",
     "--disable-pip-version-check", *packages]
)
'''

_DOCKERFILE = """\
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY packages.json install.py /opt/lca-build/
RUN python /opt/lca-build/install.py
"""


def validate_requirements(packages: Sequence[str]) -> list[str]:
    """Validate pip requirement strings. Raises ValueError with a helpful message."""
    cleaned = [str(p).strip() for p in packages if str(p).strip()]
    if not cleaned:
        raise ValueError("no packages given")
    if len(cleaned) > MAX_PACKAGES:
        raise ValueError(f"too many packages (max {MAX_PACKAGES})")
    for req in cleaned:
        if not _REQ_RE.match(req):
            raise ValueError(
                f"invalid package requirement '{req}'. Use plain names with optional extras "
                f"and version specifiers, e.g. requests, numpy>=1.26, pytest==8.2.0 "
                f"(no spaces, URLs, paths or pip options)."
            )
    return cleaned


def install_packages_into_sandbox_image(
    workspace_root: str | os.PathLike,
    packages: Sequence[str],
    timeout: int | None = None,
) -> str:
    """Build a new sandbox image with `packages` added and make it the active image."""
    root = Path(workspace_root).resolve()
    try:
        requirements = validate_requirements(packages)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not docker_available():
        return ("ERROR: Docker is not available, so packages cannot be installed. "
                "(Packages are never installed on the host.)")

    base = get_sandbox_image(root)
    digest = hashlib.sha1(("|".join([base, *sorted(requirements)])).encode()).hexdigest()[:12]
    tag = f"lca-sandbox:{digest}"

    with tempfile.TemporaryDirectory(prefix="lca-build-") as tmp:
        ctx = Path(tmp)
        (ctx / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
        (ctx / "packages.json").write_text(json.dumps(requirements), encoding="utf-8")
        (ctx / "install.py").write_text(_INSTALL_SCRIPT, encoding="utf-8")
        # Argument list, no shell: package strings never touch a shell.
        cmd = ["docker", "build", "--build-arg", f"BASE_IMAGE={base}", "-t", tag, str(ctx)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout or build_timeout(),
            )
        except subprocess.TimeoutExpired:
            return "ERROR: docker build timed out."
        except OSError as exc:
            return f"ERROR: could not start docker: {exc}"

    if proc.returncode != 0:
        tail = _truncate((proc.stdout + "\n" + proc.stderr).strip()[-4000:], 4000)
        return f"ERROR: docker build failed (exit {proc.returncode}).\n{tail}"

    image_file = root.joinpath(*IMAGE_FILE_PARTS)
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_text(tag + "\n", encoding="utf-8")
    return (f"Installed {' '.join(requirements)} by building image {tag} (from {base}). "
            f"It is now the sandbox image for all future runs.")
