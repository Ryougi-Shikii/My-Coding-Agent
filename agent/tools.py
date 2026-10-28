"""Agent tools, approval policy and per-stage tool allow-lists.

Every file operation is confined to WORKSPACE. Tools return plain strings; a failure is a
string starting with "ERROR" (the graph marks such ToolMessages with status="error").
"""

from __future__ import annotations

import ast
import base64
import functools
import hashlib
import os
import re
import stat
import struct
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agent import PROJECT_ROOT, sandbox, usage
from agent.llm import content_to_text, get_chat_model
from agent.prompts import REVIEW_PROMPT

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
        return value if value > 0 else default
    except ValueError:
        return default


def _resolve_dir(env_name: str, default: str) -> Path:
    path = Path(os.getenv(env_name, default)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path  # relative paths are relative to the project root
    return path.resolve()


def _resolve_workspace() -> Path:
    return _resolve_dir("WORKSPACE_DIR", "./workspace")


# The shared workspace (WORKSPACE_MODE=shared) and the fallback when no session is active.
WORKSPACE: Path = _resolve_workspace()

IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}
# Smaller read limit = smaller context = lower cost. read_file can read a line range instead.
MAX_READ_CHARS = _env_int("MAX_READ_CHARS", 30_000)
MAX_WRITE_CHARS = 1_000_000
MAX_OUTPUT_CHARS = 20_000
MAX_LIST_ENTRIES = 300
MAX_SEARCH_MATCHES = 100
OUTPUT_DIR = "out"  # deliverables (videos, images, reports) go here; the UIs offer them for download


# ---------------------------------------------------------------------------
# Which workspace is active?
#
# WORKSPACE_MODE=session (default): every conversation gets its own directory
#   <WORKSPACES_DIR>/<thread_id> (default ./workspaces), with its own git repository, plan,
#   memory and sandbox image. Conversations can no longer overwrite each other's files.
# WORKSPACE_MODE=shared: all conversations use WORKSPACE_DIR (the old behaviour).
#
# The active workspace is a ContextVar set by the graph driver (graph.drive) while a conversation
# runs, so several web sessions can work at the same time without sharing state.
# ---------------------------------------------------------------------------
_active_workspace: ContextVar[Path | None] = ContextVar("active_workspace", default=None)
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def current_workspace() -> Path:
    """The workspace of the running conversation (or the shared default outside one)."""
    return _active_workspace.get() or WORKSPACE


@contextmanager
def use_workspace(path: Path):
    """Make `path` the active workspace inside the `with` block (this thread/task only)."""
    token = _active_workspace.set(Path(path).resolve())
    try:
        yield
    finally:
        _active_workspace.reset(token)


def workspace_mode() -> str:
    mode = os.getenv("WORKSPACE_MODE", "session").strip().lower()
    return mode if mode in ("session", "shared") else "session"


def workspaces_root() -> Path:
    """Parent directory of all per-conversation workspaces."""
    return _resolve_dir("WORKSPACES_DIR", "./workspaces")


def workspace_for(thread_id: str | None) -> Path:
    """The workspace of one conversation (the shared workspace in `shared` mode)."""
    if workspace_mode() == "shared" or not thread_id:
        return WORKSPACE
    safe = thread_id if _THREAD_ID_RE.match(thread_id) else hashlib.sha1(
        thread_id.encode("utf-8", "replace")).hexdigest()[:24]
    return workspaces_root() / safe


def workspace_scope() -> str:
    """Key used to group conversations in the history list: the shared workspace, or the
    directory that holds all session workspaces."""
    return str(WORKSPACE if workspace_mode() == "shared" else workspaces_root())


def all_workspaces() -> list[Path]:
    """Every workspace directory that exists (used to find sandbox images still in use)."""
    roots = [WORKSPACE]
    base = workspaces_root()
    if base.is_dir():
        roots += [p for p in base.iterdir() if p.is_dir()]
    return [p for p in roots if p.is_dir()]


def ensure_workspace(root: Path | None = None) -> Path:
    """Create the workspace, its .agent state directory and the reviews directory."""
    root = root or current_workspace()
    (root / ".agent" / "reviews").mkdir(parents=True, exist_ok=True)
    return root


ensure_workspace(WORKSPACE)


class PathError(ValueError):
    """Raised when a path escapes the workspace or targets a protected file."""


def safe_path(path: str | None = "") -> Path:
    """Resolve `path` inside the workspace or raise PathError.

    Symlinks are resolved before the check, so a link that points outside is rejected.
    """
    path = path or ""
    if "\x00" in path:
        raise PathError("invalid path")
    root = current_workspace().resolve()
    candidate = Path(path).expanduser() if path.startswith("~") else Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise PathError(f"path '{path}' is outside the workspace")
    return resolved


def _rel(path: Path) -> str:
    rel = path.relative_to(current_workspace().resolve()).as_posix()
    return rel or "."


def is_git_name(name: str) -> bool:
    """True for '.git' however a filesystem may spell it (case-insensitive FS, NTFS/HFS quirks)."""
    lowered = name.lower().rstrip(". ")
    return lowered in (".git", "git~1")


def _is_regular_file(path: Path) -> bool:
    """A real file (not a symlink, FIFO, device or directory). lstat never follows links."""
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _read_capped(path: Path, max_bytes: int) -> tuple[str, bool]:
    """Read at most `max_bytes` from a regular file; returns (text, was_truncated)."""
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    return raw[:max_bytes].decode("utf-8", errors="replace"), len(raw) > max_bytes


def assert_writable(path: Path) -> None:
    """Refuse writes to git internals and to the sandbox image pointer."""
    rel = Path(_rel(path))
    if any(is_git_name(part) for part in rel.parts):
        raise PathError("the .git directory is protected; use the git tools instead")
    if rel.as_posix() == ".agent/sandbox_image.txt":
        raise PathError("sandbox_image.txt is managed by install_packages and cannot be edited")


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


def _safe_tool(fn):
    """Turn path/OS errors into 'ERROR: ...' strings instead of exceptions."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PathError as exc:
            return f"ERROR: {exc}"
        except OSError as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    return wrapper


class Content(str):
    """File or program content returned by a tool. It is never an error, even if the text itself
    happens to start with "ERROR" (a log file, for example)."""


def is_error_text(text: str) -> bool:
    if isinstance(text, Content):
        return False
    return str(text).lstrip().startswith("ERROR")


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------
@tool
@_safe_tool
def list_files(path: str = "") -> str:
    """List files and directories under `path` (relative to the workspace, default: root).
    Directories end with '/'. Skips .git, __pycache__ and similar noise."""
    base = safe_path(path)
    if not base.exists():
        return f"ERROR: '{path}' does not exist"
    if base.is_file():
        return _rel(base)
    entries: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        here = Path(dirpath)
        for name in dirnames:
            entries.append(_rel(here / name) + "/")
        for name in sorted(filenames):
            entries.append(_rel(here / name))
        if len(entries) > MAX_LIST_ENTRIES:
            break
    entries.sort()
    if not entries:
        return "(empty)"
    if len(entries) > MAX_LIST_ENTRIES:
        return "\n".join(entries[:MAX_LIST_ENTRIES]) + "\n... (truncated)"
    return "\n".join(entries)


@tool
@_safe_tool
def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a UTF-8 text file from the workspace. Large files are cut off at a size limit; for those
    pass start_line / end_line (1-based, inclusive) to read just the part you need. Images and
    other binary files cannot be read here: use view_image or inspect_file."""
    target = safe_path(path)
    if not _is_regular_file(target):  # also rejects FIFOs/devices that could block forever
        return f"ERROR: '{path}' is not a regular file"
    # Read a bounded amount: sandboxed code can create arbitrarily large files.
    text, truncated = _read_capped(target, MAX_READ_CHARS * 4)
    if "\x00" in text[:4096]:
        return (f"ERROR: '{path}' looks like a binary file. Use inspect_file for details, or "
                f"view_image if it is an image.")
    if start_line or end_line:
        lines = text.splitlines(keepends=True)
        first = max(1, start_line or 1)
        last = min(len(lines), end_line) if end_line else len(lines)
        if first > len(lines):
            return f"ERROR: '{path}' has only {len(lines)} lines"
        chunk = "".join(lines[first - 1:last])
        note = f"[lines {first}-{last} of {len(lines)}{'+' if truncated else ''}]\n"
        if len(chunk) > MAX_READ_CHARS:
            chunk = chunk[:MAX_READ_CHARS] + "\n... [truncated: choose a smaller line range]"
        return Content(note + chunk)
    if len(text) > MAX_READ_CHARS or truncated:
        return Content(text[:MAX_READ_CHARS] + "\n... [truncated: the file is larger than the read "
                       "limit; pass start_line/end_line to read a specific part]")
    return Content(text)


@tool
@_safe_tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file in the workspace with `content`. Parent directories are
    created. Requires human approval."""
    target = safe_path(path)
    assert_writable(target)
    if target.is_dir():
        return f"ERROR: '{path}' is a directory"
    if target.exists() and not _is_regular_file(target):  # e.g. a FIFO would block the write
        return f"ERROR: '{path}' exists and is not a regular file"
    if len(content) > MAX_WRITE_CHARS:
        return f"ERROR: content too large (max {MAX_WRITE_CHARS} characters)"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:  # no CRLF translation on Windows
        handle.write(content)
    return f"Wrote {len(content)} characters to {_rel(target)}"


@tool
@_safe_tool
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> str:
    """Replace the text `find` with `replace` in an existing file. `find` must match exactly
    once unless replace_all=True. Read the file first so `find` is exact. Requires approval."""
    target = safe_path(path)
    assert_writable(target)
    if not _is_regular_file(target):
        return f"ERROR: '{path}' is not a regular file"
    if target.stat().st_size > MAX_WRITE_CHARS * 2:
        return f"ERROR: '{path}' is too large to edit"
    if not find:
        return "ERROR: 'find' must not be empty"
    with target.open("r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    count = text.count(find)
    if count == 0:
        return "ERROR: 'find' text not found. Re-read the file and copy the text exactly."
    if count > 1 and not replace_all:
        return (f"ERROR: 'find' matches {count} places. Add more surrounding context "
                f"or pass replace_all=true.")
    updated = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    return f"Replaced {count if replace_all else 1} occurrence(s) in {_rel(target)}"


@tool
@_safe_tool
def create_directory(path: str) -> str:
    """Create a directory (and parents) inside the workspace. Requires human approval."""
    target = safe_path(path)
    assert_writable(target)
    target.mkdir(parents=True, exist_ok=True)
    return f"Created directory {_rel(target)}"


@tool
@_safe_tool
def search_text(query: str, path: str = "") -> str:
    """Case-insensitive text search in files under `path` (default: whole workspace).
    Returns up to 100 matches as 'file:line: text'."""
    if not query:
        return "ERROR: query must not be empty"
    base = safe_path(path)
    if not base.exists():
        return f"ERROR: '{path}' does not exist"
    needle = query.lower()
    matches: list[str] = []

    def files():
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                yield Path(dirpath) / name

    for file in files():
        # Regular files only: symlinks, FIFOs and devices (which sandboxed code can create)
        # could block forever or read unbounded data.
        if not _is_regular_file(file):
            continue
        try:
            text, too_big = _read_capped(file, 1_000_000)
        except OSError:
            continue
        if too_big or "\x00" in text[:2048]:  # large or binary
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if needle in line.lower():
                try:
                    shown = _rel(file.resolve())
                except ValueError:  # symlink pointing outside: skip
                    break
                matches.append(f"{shown}:{number}: {line.strip()[:200]}")
                if len(matches) >= MAX_SEARCH_MATCHES:
                    return "\n".join(matches) + "\n... (stopped at 100 matches)"
    return "\n".join(matches) if matches else "No matches."


# ---------------------------------------------------------------------------
# Execution tools
# ---------------------------------------------------------------------------
@tool
@_safe_tool
def run_python_file(path: str) -> str:
    """Run a workspace .py file inside the Docker sandbox (no network, 512 MB, timeout).
    Returns exit code, stdout and stderr. Requires human approval."""
    return str(sandbox.run_python_in_sandbox(current_workspace(), path))


@tool
@_safe_tool
def check_python_syntax(path: str) -> str:
    """Check Python syntax of a file, or of every .py file under a directory, without running
    it. Safe: only parses the code."""
    target = safe_path(path)
    if not target.exists():
        return f"ERROR: '{path}' does not exist"
    if target.is_file():
        files = [target]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            files += [Path(dirpath) / n for n in sorted(filenames) if n.endswith(".py")]
    # Regular files only (no symlinks that could point outside, no FIFOs/devices).
    files = [f for f in files if _is_regular_file(f)]
    if not files:
        return "No Python files found."
    problems: list[str] = []
    for file in files:
        source, too_big = _read_capped(file, 2_000_000)
        if too_big:
            problems.append(f"{_rel(file)}: file too large to check")
            continue
        try:
            ast.parse(source, filename=str(file))
        except (SyntaxError, ValueError) as exc:  # ValueError: e.g. null bytes in source
            line = getattr(exc, "lineno", None) or 1
            message = getattr(exc, "msg", None) or str(exc)
            problems.append(f"{_rel(file)}:{line}: SyntaxError: {message}")
    if problems:
        return "ERROR: syntax problems found:\n" + "\n".join(problems)
    return f"Syntax OK ({len(files)} file(s) checked)."


@tool
@_safe_tool
def run_tests(path: str = "") -> str:
    """Run pytest inside the Docker sandbox, optionally limited to `path`. Requires approval.
    If pytest is missing in the sandbox image, install it with install_packages('pytest')."""
    args = ["-m", "pytest", "-o", "cache_dir=/tmp/pytest_cache", "-q"]
    if path:
        rel = _rel(safe_path(path))
        args.append("." if rel == "." else "./" + rel)
    result = sandbox.run_python_args_in_sandbox(current_workspace(), args, timeout=sandbox.test_timeout())
    text = str(result)
    if "No module named pytest" in text:
        text += "\nHint: pytest is not installed in the sandbox image. Call install_packages('pytest')."
    return text


@tool
@_safe_tool
def install_packages(packages: str) -> str:
    """Install pip packages by building a NEW Docker sandbox image from the current one.
    `packages` is space-separated requirements, e.g. 'pytest requests>=2.31'. Requires approval."""
    result = sandbox.install_packages_into_sandbox_image(current_workspace(), packages.split())
    if not is_error_text(result):  # the old image of this workspace is no longer referenced
        try:
            sandbox.prune_unused_images(all_workspaces())
        except Exception:  # housekeeping must never turn a successful install into an error
            pass
    return result


@tool
@_safe_tool
def run_python(code: str) -> str:
    """Run a Python snippet inside the Docker sandbox (no network, resource limits, timeout) and
    return its output. The working directory is the workspace, so the snippet can read and write
    workspace files (put deliverables in out/) and start tools installed in the image, for example
    `subprocess.run(["ffmpeg", ...])`. Prefer write_file + run_python_file for code you want to keep.
    Requires human approval."""
    return str(sandbox.run_code_in_sandbox(current_workspace(), code))


# ---------------------------------------------------------------------------
# Looking at binary files: images and media
# ---------------------------------------------------------------------------
MAX_IMAGE_BYTES = _env_int("MAX_IMAGE_BYTES", 2_000_000)
_IMAGE_MIME = {"png": "image/png", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
_MEDIA_KINDS = {"mp4", "webm", "mkv", "mov", "wav", "mp3", "ogg", "flac", "aac"}


def sniff_type(head: bytes) -> str:
    """File type from the first bytes ('png', 'jpeg', 'mp4', ..., 'text' or 'binary')."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[4:8] == b"ftyp":
        return "mov" if head[8:12] == b"qt  " else "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "mp3"
    if head[:5] == b"%PDF-":
        return "pdf"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    return "binary" if b"\x00" in head[:4096] else "text"


def image_size(kind: str, data: bytes) -> tuple[int, int] | None:
    """(width, height) read from the header; no image library is used on untrusted data."""
    try:
        if kind == "png" and len(data) >= 24:
            return struct.unpack(">II", data[16:24])
        if kind == "gif" and len(data) >= 10:
            return struct.unpack("<HH", data[6:10])
        if kind == "jpeg":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                length = struct.unpack(">H", data[i + 2:i + 4])[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    height, width = struct.unpack(">HH", data[i + 5:i + 9])
                    return width, height
                i += 2 + length
    except (struct.error, IndexError):
        return None
    return None


def load_image(path: str) -> tuple[str, bytes, str]:
    """Validated image for view_image: (mime type, bytes, workspace-relative path).
    Raises PathError / ValueError with a message the model can act on."""
    target = safe_path(path)
    if not _is_regular_file(target):
        raise ValueError(f"'{path}' is not a regular file")
    size = target.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"'{path}' is {size // 1024} KB, over the {MAX_IMAGE_BYTES // 1024} KB limit. Make a "
            f"smaller copy first (for example with Pillow: img.thumbnail((1280, 1280)); save as "
            f"JPEG or PNG) and view that.")
    data = target.read_bytes()
    kind = sniff_type(data[:64])
    if kind not in _IMAGE_MIME:
        raise ValueError(f"'{path}' is not a PNG, JPEG, GIF or WebP image (looks like: {kind})")
    return _IMAGE_MIME[kind], data, _rel(target)


def image_content_block(mime: str, data: bytes) -> dict:
    """A provider-neutral image block (LangChain converts data URIs for every provider)."""
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}}


@tool
@_safe_tool
def view_image(path: str) -> str:
    """Look at an image file in the workspace (PNG, JPEG, GIF or WebP, up to about 2 MB): the image
    is attached to the conversation so you can see it. Use it to check charts, rendered pages, video
    frames and contact sheets. Keep images small (about 1280 px wide) to save cost; each image you
    look at costs tokens, and only the most recent ones stay in context."""
    try:
        mime, data, rel = load_image(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    size = image_size(mime.split("/")[1].replace("jpg", "jpeg"), data)
    dims = f", {size[0]}x{size[1]} px" if size else ""
    return f"Image attached: {rel} ({mime}{dims}, {len(data) // 1024} KB)"


_FFPROBE_SNIPPET = """\
import json, shutil, subprocess, sys
if not shutil.which("ffprobe"):
    print("ffprobe is not installed in the sandbox image (see Dockerfile.sandbox)")
    sys.exit(0)
r = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries",
     "format=format_name,duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,"
     "r_frame_rate,sample_rate,channels", "-of", "json", sys.argv[1]],
    capture_output=True, text=True, timeout=30)
print((r.stdout or r.stderr).strip())
"""


@tool
@_safe_tool
def inspect_file(path: str) -> str:
    """Describe any file in the workspace without reading all of it: size, detected type, image
    dimensions, line count for text, and for audio/video the duration, codecs, resolution and
    streams (via ffprobe in the sandbox). Use it before touching binary or media files."""
    target = safe_path(path)
    if not target.exists():
        return f"ERROR: '{path}' does not exist"
    if target.is_dir():
        count = sum(1 for _ in target.iterdir())
        return f"{_rel(target)}/ is a directory with {count} entries (use list_files)"
    if not _is_regular_file(target):
        return f"ERROR: '{path}' is not a regular file"
    size = target.stat().st_size
    with target.open("rb") as handle:
        head = handle.read(65536)
    kind = sniff_type(head)
    lines = [f"{_rel(target)}: {kind}, {size:,} bytes"]
    dims = image_size(kind, head)
    if dims:
        lines.append(f"image size: {dims[0]}x{dims[1]} px")
    if kind == "text":
        text, truncated = _read_capped(target, 2_000_000)
        lines.append(f"lines: {text.count(chr(10)) + (1 if text and not text.endswith(chr(10)) else 0)}"
                     + ("+ (file is larger than 2 MB)" if truncated else ""))
    if kind in _MEDIA_KINDS:
        result = sandbox.run_code_in_sandbox(
            current_workspace(), _FFPROBE_SNIPPET, timeout=45,
            script_args=[sandbox._container_path(current_workspace(), target)])
        lines.append(("ffprobe: " + (result.stdout.strip() or result.stderr.strip() or "(no output)"))
                     if result.ok else f"ffprobe not run: {result.error or result.stderr.strip() or 'failed'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deliverables: files the agent leaves in <workspace>/out/ are offered to the user
# ---------------------------------------------------------------------------
def output_files(root: Path | None = None, limit: int = 100) -> list[tuple[str, int]]:
    """(path relative to the workspace, size in bytes) of regular files in out/, newest first."""
    root = (root or current_workspace()).resolve()
    base = root / OUTPUT_DIR
    if not base.is_dir() or base.is_symlink():
        return []
    found: list[tuple[float, str, int]] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for name in filenames:
            path = Path(dirpath) / name
            if _is_regular_file(path):
                info = path.lstat()
                found.append((info.st_mtime, path.relative_to(root).as_posix(), info.st_size))
    found.sort(reverse=True)
    return [(rel, size) for _, rel, size in found[:limit]]


# ---------------------------------------------------------------------------
# Git tools (subprocess without a shell, always inside WORKSPACE)
# ---------------------------------------------------------------------------
def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    # Never walk up into a parent repository (the workspace may live inside another repo).
    env["GIT_CEILING_DIRECTORIES"] = str(current_workspace().resolve().parent)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


# Config keys a repository created by git_init (plus our identity fallback) may contain.
# Anything else (filter.*, diff.*.command, alias.*, core.sshCommand, include.path, ...) can make
# git run programs, and sandboxed code can create a .git directory before git_init ever runs.
_SAFE_CONFIG_KEYS = frozenset({
    "core.repositoryformatversion", "core.filemode", "core.bare", "core.logallrefupdates",
    "core.ignorecase", "core.precomposeunicode", "core.symlinks",
    "user.name", "user.email",
})


def _repo_problem() -> str | None:
    """None if there is no .git or it looks like a plain, trustworthy repository."""
    gitdir = current_workspace().resolve() / ".git"
    if not os.path.lexists(gitdir):
        return None
    if gitdir.is_symlink() or not gitdir.is_dir():
        return ".git must be a real directory inside the workspace"
    for extra in ("commondir", "objects/info/alternates"):
        if (gitdir / extra).exists():
            return f".git/{extra} is not allowed"
    config = gitdir / "config"
    if config.exists():
        try:
            # Reads the file only; nothing in it is executed.
            proc = subprocess.run(
                ["git", "config", "--file", str(config), "--list", "--name-only"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=10, env=_git_env())
        except (OSError, subprocess.SubprocessError):
            return "cannot read .git/config"
        if proc.returncode != 0:
            return "cannot parse .git/config"
        for key in proc.stdout.splitlines():
            key = key.strip().lower()
            if key and key not in _SAFE_CONFIG_KEYS and not key.startswith("branch."):
                return f".git/config contains the unsupported setting '{key}'"
    return None


def _git(*args: str, timeout: int = 60, raw: bool = False) -> tuple[int, str]:
    """Run git in WORKSPACE. Returns (exit_code, combined output).

    With raw=True returns stdout unmodified (for NUL-separated output).
    """
    problem = _repo_problem()
    if problem:
        return 126, f"unsafe .git directory: {problem}"
    cmd = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
           "-c", "commit.gpgsign=false", *args]
    try:
        proc = subprocess.run(cmd, cwd=current_workspace().resolve(), env=_git_env(), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError:
        return 127, "git is not installed"
    except subprocess.TimeoutExpired:
        return 124, "git command timed out"
    if raw:
        return proc.returncode, proc.stdout
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _is_repo() -> bool:
    code, out = _git("rev-parse", "--show-toplevel")
    return code == 0 and Path(out).resolve() == current_workspace().resolve()


def _has_head() -> bool:
    return _git("rev-parse", "--verify", "HEAD")[0] == 0


_NOT_A_REPO = "ERROR: the workspace is not a git repository. Call git_init first."


def _require_repo() -> str | None:
    """Error text if git tools must not run, else None."""
    problem = _repo_problem()
    if problem:
        return (f"ERROR: refusing to use the .git directory: {problem}. "
                f"It was not created by git_init; ask the user to delete it.")
    return None if _is_repo() else _NOT_A_REPO


@tool
@_safe_tool
def git_init() -> str:
    """Initialise a git repository in the workspace (adds a .gitignore for agent state)."""
    problem = _repo_problem()
    if problem:
        return f"ERROR: refusing to use the existing .git directory: {problem}."
    if _is_repo():
        return "Repository already initialised."
    code, out = _git("init")
    if code != 0:
        return f"ERROR: {out}"
    gitignore = current_workspace() / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(".agent/\n__pycache__/\n*.pyc\n.pytest_cache/\n", encoding="utf-8")
    return out or "Initialised repository."


@tool
@_safe_tool
def git_status() -> str:
    """Show `git status` for the workspace repository."""
    error = _require_repo()
    if error:
        return error
    code, out = _git("status", "--short", "--branch")
    return out if code == 0 else f"ERROR: {out}"


@tool
@_safe_tool
def git_add(path: str = ".") -> str:
    """Stage `path` (default: everything) for commit."""
    error = _require_repo()
    if error:
        return error
    target = safe_path(path)
    if any(is_git_name(part) for part in Path(_rel(target)).parts):
        return "ERROR: cannot add .git"
    code, out = _git("add", "--", _rel(target))
    return (out or f"Staged {_rel(target)}") if code == 0 else f"ERROR: {out}"


@tool
@_safe_tool
def git_commit(message: str) -> str:
    """Commit staged changes with `message`. A local git identity is configured if missing.
    Requires human approval."""
    message = (message or "").strip()
    if not message:
        return "ERROR: commit message must not be empty"
    if len(message) > 5000:
        return "ERROR: commit message too long"
    error = _require_repo()
    if error:
        return error
    if not _git("config", "user.name")[1]:
        _git("config", "--local", "user.name", "Coding Agent")
    if not _git("config", "user.email")[1]:
        _git("config", "--local", "user.email", "coding-agent@localhost")
    code, out = _git("commit", "-m", message)
    if code != 0:
        if "nothing to commit" in out or "no changes added" in out:
            return "ERROR: nothing to commit. Stage changes with git_add first."
        return f"ERROR: {out}"
    return out


def _diff_text() -> str:
    args = ["diff", "HEAD"] if _has_head() else ["diff"]
    return _git(*args)[1]


@tool
@_safe_tool
def git_diff() -> str:
    """Show uncommitted changes (staged and unstaged) in the workspace repository."""
    error = _require_repo()
    if error:
        return error
    return _truncate(_diff_text()) or "(no changes)"


@tool
@_safe_tool
def git_log() -> str:
    """Show the last 20 commits (one line each)."""
    error = _require_repo()
    if error:
        return error
    if not _has_head():
        return "No commits yet."
    code, out = _git("log", "--oneline", "-n", "20")
    return out if code == 0 else f"ERROR: {out}"


# ---------------------------------------------------------------------------
# Planning and memory tools (all live in workspace/.agent)
# ---------------------------------------------------------------------------
def _agent_file(name: str) -> Path:
    ensure_workspace()
    return current_workspace() / ".agent" / name


@tool
@_safe_tool
def update_plan(plan_markdown: str) -> str:
    """Overwrite .agent/PLAN.md with the current plan (Markdown)."""
    if not plan_markdown.strip():
        return "ERROR: plan must not be empty"
    _agent_file("PLAN.md").write_text(plan_markdown.rstrip() + "\n", encoding="utf-8")
    return f"Plan updated ({len(plan_markdown)} characters)."


@tool
@_safe_tool
def update_todo(todo_markdown: str) -> str:
    """Overwrite .agent/TODO.md with the current todo checklist (Markdown, '- [ ] item')."""
    if not todo_markdown.strip():
        return "ERROR: todo list must not be empty"
    _agent_file("TODO.md").write_text(todo_markdown.rstrip() + "\n", encoding="utf-8")
    return f"Todo list updated ({len(todo_markdown)} characters)."


@tool
@_safe_tool
def save_memory(note: str) -> str:
    """Append a timestamped note to the persistent memory .agent/MEMORY.md."""
    note = note.strip()
    if not note:
        return "ERROR: note must not be empty"
    memory = _agent_file("MEMORY.md")
    if not memory.exists():
        memory.write_text("# Agent memory\n\n", encoding="utf-8")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with memory.open("a", encoding="utf-8") as fh:
        fh.write(f"- [{stamp}] {note}\n")
    return "Memory saved."


@tool
@_safe_tool
def read_agent_state() -> str:
    """Read the saved plan, todo list and memory (.agent/PLAN.md, TODO.md, MEMORY.md)."""
    sections = []
    for name in ("PLAN.md", "TODO.md", "MEMORY.md"):
        file = _agent_file(name)
        body = file.read_text(encoding="utf-8", errors="replace").strip() if file.exists() else ""
        sections.append(f"## {name}\n{body or '(empty)'}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Review tool
# ---------------------------------------------------------------------------
# The verdict must be the whole line (markdown bold/quote/period around it is tolerated).
# 'VERDICT: APPROVE_WITH_NITS', 'APPROVE-ish', 'Approved with concerns' or 'APPROVE (but ...)'
# do NOT count as APPROVE: they fail closed.
_VERDICT_RE = re.compile(
    r"^[ \t>*_`#-]*VERDICT:[ \t]*\**[ \t]*(APPROVE|REQUEST_CHANGES|NO_CHANGES)[ \t*_`.]*$",
    re.IGNORECASE | re.MULTILINE,
)
_TREE_RE = re.compile(r"^REVIEWED-TREE:[ \t]*([0-9a-f]{64})[ \t]*$", re.MULTILINE)

REVIEW_CHUNK_CHARS = 50_000  # diff characters per reviewer call
MAX_REVIEW_CHUNKS = 8        # beyond this the change is too large to review: request changes


def parse_verdict(text: str) -> str:
    """Return the LAST verdict in `text`. A missing verdict fails closed (REQUEST_CHANGES)."""
    found = _VERDICT_RE.findall(text or "")
    return found[-1].upper() if found else "REQUEST_CHANGES"


def parse_reviewed_tree(text: str) -> str:
    """Return the LAST 'REVIEWED-TREE: <sha256>' fingerprint in a review ('' if absent)."""
    found = _TREE_RE.findall(text or "")
    return found[-1] if found else ""


def tree_fingerprint() -> str:
    """SHA-256 over the content of every non-ignored file in the workspace (index-independent).

    The reviewer approves a specific tree; the pipeline refuses to commit any other one.
    `.agent/` is excluded because the agent legitimately updates its notes at any time.
    """
    code, listing = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard", raw=True)
    if code != 0:
        return ""
    root = current_workspace().resolve()
    digest = hashlib.sha256()
    for name in sorted({n for n in listing.split("\0") if n}):
        if name == ".agent" or name.startswith(".agent/"):
            continue
        path = root / name
        if path.is_symlink():
            marker = b"link:" + os.readlink(path).encode("utf-8", "surrogateescape")
        elif _is_regular_file(path):
            inner = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    inner.update(block)
            marker = b"file:" + inner.digest()
        else:
            marker = b"absent"  # deleted tracked file (or something that is not a file)
        digest.update(name.encode("utf-8", "surrogateescape") + b"\0" + marker + b"\0")
    return digest.hexdigest()


def _split_chunks(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most `limit` characters, preferring line boundaries."""
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:  # a single enormous line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if current and len(current) + len(line) > limit:
            chunks.append(current)
            current = ""
        current += line
    if current:
        chunks.append(current)
    return chunks


@tool
@_safe_tool
def review_changes() -> str:
    """Review uncommitted changes: sends `git status` and the COMPLETE `git diff` (in parts if it
    is large) to the chat model, saves the review to .agent/reviews/review_<timestamp>.md and
    returns it. The output contains a 'VERDICT:' line and a 'REVIEWED-TREE:' fingerprint of exactly
    what was reviewed; only that tree can be committed."""
    error = _require_repo()
    if error:
        return error
    # Make brand-new files show up in `git diff` without staging their content.
    _git("add", "--intent-to-add", "-A")
    fingerprint = tree_fingerprint()
    status = _truncate(_git("status", "--short")[1], 5_000)
    diff = _diff_text()
    if not status.strip() and not diff.strip():
        return "No changes to review.\nVERDICT: NO_CHANGES"
    if not fingerprint:
        return "ERROR: could not fingerprint the workspace tree"

    parts = _split_chunks(diff, REVIEW_CHUNK_CHARS) or [""]
    if len(parts) > MAX_REVIEW_CHUNKS:
        review = (f"The change is too large to review in full ({len(diff)} characters of diff, "
                  f"limit {MAX_REVIEW_CHUNKS * REVIEW_CHUNK_CHARS}).\n"
                  f"- [blocker] Split the work into smaller changes.\n\nVERDICT: REQUEST_CHANGES")
    else:
        model = get_chat_model()
        outputs: list[str] = []
        for number, part in enumerate(parts, 1):
            header = f"(part {number} of {len(parts)} of the diff)\n" if len(parts) > 1 else ""
            request = f"## git status\n{status}\n\n{header}## git diff\n{part}"
            try:
                reply = model.invoke(
                    [SystemMessage(content=REVIEW_PROMPT), HumanMessage(content=request)])
            except Exception as exc:  # network/API problems must not crash the graph
                return f"ERROR: review model call failed: {type(exc).__name__}: {exc}"
            usage.record_extra(str(current_workspace()), getattr(reply, "usage_metadata", None))
            outputs.append(content_to_text(reply.content).strip() or "(empty review)")
        if len(outputs) == 1:
            review = outputs[0]
            if not _VERDICT_RE.search(review):
                review += "\n\nVERDICT: REQUEST_CHANGES"  # no verdict given: fail closed
        else:
            verdicts = [parse_verdict(o) for o in outputs]
            body = "\n\n".join(f"### Part {i}\n{_VERDICT_RE.sub('', o).strip()}"
                                for i, o in enumerate(outputs, 1))
            overall = "APPROVE" if all(v == "APPROVE" for v in verdicts) else "REQUEST_CHANGES"
            review = f"{body}\n\nVERDICT: {overall}"
    # Appended by this tool, after the model's text, so model output cannot forge it.
    review += f"\n\nREVIEWED-TREE: {fingerprint}"

    reviews_dir = ensure_workspace() / ".agent" / "reviews"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = reviews_dir / f"review_{stamp}.md"
    counter = 1
    while out.exists():
        counter += 1
        out = reviews_dir / f"review_{stamp}_{counter}.md"
    out.write_text(review + "\n", encoding="utf-8")
    return review


# ---------------------------------------------------------------------------
# Build handoff (chat stage only; intercepted by the graph, never executed as a normal tool)
# ---------------------------------------------------------------------------
STEP_TO_STAGE = {"plan": "planner", "code": "coder", "review": "reviewer", "commit": "committer"}
DEFAULT_BUILD_STEPS = ("plan", "code", "review")


def normalize_build_steps(steps) -> list[str]:
    """Validate the steps chosen for a build and return stage names in pipeline order.

    The ORDER is fixed (planner, coder, reviewer, committer); the caller only chooses which
    steps run. Rules: a build needs 'code' or 'review', and 'commit' requires 'review', so
    nothing can be committed without a review of exactly what is committed.
    """
    if steps is None or (not isinstance(steps, str) and len(steps) == 0):
        steps = DEFAULT_BUILD_STEPS
    if isinstance(steps, str):
        steps = [part for part in re.split(r"[\s,>/+-]+", steps) if part]
    chosen: set[str] = set()
    for raw in steps:
        key = str(raw).strip().lower()
        stage = STEP_TO_STAGE.get(key) or (key if key in STEP_TO_STAGE.values() else None)
        if stage is None:
            raise ValueError(f"unknown step '{raw}'. Valid steps: plan, code, review, commit")
        chosen.add(stage)
    if not chosen & {"coder", "reviewer"}:
        raise ValueError("a build needs at least 'code' or 'review'")
    if "committer" in chosen and "reviewer" not in chosen:
        raise ValueError("'commit' requires 'review': only reviewed changes can be committed")
    return [stage for stage in ("planner", "coder", "reviewer", "committer") if stage in chosen]


@tool
def start_build(task: str, steps: list[str] | None = None) -> str:
    """Hand a concrete coding task to the build pipeline (the only way to change project files).
    `task`: a complete, self-contained description of what to build or change.
    `steps`: which steps to run, any of 'plan', 'code', 'review', 'commit' (they always run in
    that order). Default ['plan','code','review']. Use ['code'] for a tiny edit; add 'commit' only
    when the user wants it committed ('commit' needs 'review'). Use ['review','commit'] to review
    and commit changes that already exist. Call it alone, never together with other tools."""
    return "ERROR: start_build only works from the chat stage of the auto agent."


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
TOOLS = [
    # file tools
    list_files, read_file, write_file, edit_file, create_directory, search_text,
    # execution tools
    run_python_file, run_python, check_python_syntax, run_tests, install_packages, inspect_file,
    # git tools
    git_init, git_status, git_add, git_commit, git_diff, git_log,
    # planning and memory tools
    update_plan, update_todo, save_memory, read_agent_state,
    # review tool
    review_changes,
]
# TOOLS is what the classic react agent gets. start_build exists only in the auto agent, and so
# does view_image: its image has to be attached by the pipeline's tools node (a plain ToolNode only
# returns text).
ALL_TOOLS = [*TOOLS, start_build, view_image]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

# ---------------------------------------------------------------------------
# Approval policy: only dangerous tools need a human
# ---------------------------------------------------------------------------
# Dangerous = changes project files, executes code, installs software or records history.
DANGEROUS_TOOLS = frozenset({
    "write_file", "edit_file", "create_directory",
    "run_python_file", "run_python", "run_tests", "install_packages",
    "git_commit",
})
# Safe = reads, listings, parsing, staging, and the agent's own notes in .agent/.
SAFE_TOOLS = frozenset({
    "list_files", "read_file", "search_text", "check_python_syntax", "inspect_file", "view_image",
    "git_init", "git_status", "git_add", "git_diff", "git_log",
    "update_plan", "update_todo", "save_memory", "read_agent_state", "review_changes",
    "start_build",  # a handoff only: it starts the build stages, whose dangerous tools still ask
})
assert DANGEROUS_TOOLS | SAFE_TOOLS == set(TOOLS_BY_NAME), "every tool must be classified"
assert not DANGEROUS_TOOLS & SAFE_TOOLS


def approval_mode() -> str:
    """'dangerous' (default): only dangerous tools need approval. 'all': every tool does."""
    mode = os.getenv("APPROVAL_MODE", "dangerous").strip().lower()
    return mode if mode in ("dangerous", "all", "none") else "dangerous"


def requires_approval(tool_name: str) -> bool:
    """True if a call to `tool_name` must be approved by a human.
    Unknown tools always require approval (fail closed)."""
    mode = approval_mode()
    if mode == "none":
        return False
    if mode == "all":
        return True
    return tool_name not in SAFE_TOOLS


# ---------------------------------------------------------------------------
# Per-stage allow-lists for the planner -> coder -> reviewer -> committer pipeline
# ---------------------------------------------------------------------------
STAGES = ("planner", "coder", "reviewer", "committer")
CHAT_STAGE = "chat"

STAGE_TOOLS: dict[str, tuple[str, ...]] = {
    # Conversation: can look at everything, run code/tests (with approval) and keep notes, but
    # cannot change project files or commit. Changes go through start_build.
    "chat": ("read_agent_state", "list_files", "read_file", "search_text", "check_python_syntax",
             "inspect_file", "view_image", "run_python_file", "run_python", "run_tests",
             "git_status", "git_diff", "git_log",
             "update_plan", "update_todo", "save_memory", "start_build"),
    # Understand and plan; cannot touch project files or run code.
    "planner": ("read_agent_state", "list_files", "read_file", "search_text",
                "git_status", "git_log", "update_plan", "update_todo"),
    # The only stage that can change files, run code or install packages. Cannot review or commit.
    "coder": ("read_agent_state", "list_files", "read_file", "search_text",
              "write_file", "edit_file", "create_directory", "check_python_syntax",
              "inspect_file", "view_image", "run_python_file", "run_python", "run_tests",
              "install_packages", "update_todo", "git_status", "git_diff", "save_memory"),
    # Read-only judge: no write/edit tools at all.
    # git_init is allowed only so review_changes can work on a brand-new workspace.
    "reviewer": ("read_agent_state", "list_files", "read_file", "search_text",
                 "check_python_syntax", "inspect_file", "view_image", "run_tests", "git_init",
                 "git_status", "git_diff", "git_log", "review_changes"),
    # The only stage that can stage and commit.
    "committer": ("read_agent_state", "git_init", "git_status", "git_diff", "git_log",
                  "git_add", "git_commit", "save_memory"),
}
for _stage, _names in STAGE_TOOLS.items():
    assert set(_names) <= set(TOOLS_BY_NAME), f"unknown tool in {_stage}"
assert "git_commit" not in sum((STAGE_TOOLS[s] for s in ("chat", "planner", "coder", "reviewer")), ())
assert not {"write_file", "edit_file", "create_directory", "install_packages", "git_add"} & set(STAGE_TOOLS["chat"])


def workspace_top_level(limit: int = 50, root: Path | None = None) -> list[tuple[str, bool]]:
    """Top-level workspace entries as (name, is_dir), directories first (for the UI sidebar)."""
    root = ensure_workspace(root)
    items = [(p.name, p.is_dir()) for p in root.iterdir() if p.name not in IGNORED_DIRS]
    items.sort(key=lambda item: (not item[1], item[0].lower()))
    return items[:limit]