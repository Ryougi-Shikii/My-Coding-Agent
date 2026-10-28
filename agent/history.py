"""Durable conversations.

* The LangGraph checkpointer is a SQLite file, so a conversation (messages, plan state and even a
  pending approval) survives restarts of the CLI or the web app.
* A small index table remembers each conversation's title, mode and timestamps so the user can list,
  resume, switch and delete them. Conversations are grouped by workspace.

Where: $CONVERSATION_DB, default ``<project>/.data/conversations.sqlite``. Set it to ``memory`` to
disable persistence (conversations then live only as long as the process).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from agent import PROJECT_ROOT

_lock = threading.RLock()
_saver = None
_saver_key: str | None = None
_memory_index: dict[str, dict] = {}  # used instead of SQLite when persistence is off

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    thread_id  TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    mode       TEXT NOT NULL,
    workspace  TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""


def db_path() -> str:
    """Resolved database path, or ':memory:' when persistence is disabled."""
    raw = os.getenv("CONVERSATION_DB", "").strip()
    if raw.lower() in (":memory:", "memory", "off", "none"):
        return ":memory:"
    path = Path(raw).expanduser() if raw else PROJECT_ROOT / ".data" / "conversations.sqlite"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def persistent() -> bool:
    return db_path() != ":memory:"


def reset() -> None:
    """Forget cached handles (tests, or after changing CONVERSATION_DB)."""
    global _saver, _saver_key
    with _lock:
        saver = _saver
        _saver, _saver_key = None, None
        _memory_index.clear()
    conn = getattr(saver, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def get_checkpointer():
    """The process-wide checkpointer: SQLite when persistent, else MemorySaver."""
    global _saver, _saver_key
    path = db_path()
    with _lock:
        if _saver is not None and _saver_key == path:
            return _saver
        if path == ":memory:":
            _saver = MemorySaver()
        else:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError as exc:
                raise RuntimeError(
                    "Saving conversations needs the 'langgraph-checkpoint-sqlite' package. "
                    "Run: pip install -r requirements.txt   (or set CONVERSATION_DB=memory to "
                    "run without saving)") from exc

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: Streamlit runs scripts on several threads.
            conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            _saver = SqliteSaver(conn)
        _saver_key = path
        return _saver


# ---------------------------------------------------------------------------
# Conversation index
# ---------------------------------------------------------------------------
def new_thread_id() -> str:
    return str(uuid.uuid4())


def make_title(text: str, limit: int = 60) -> str:
    title = re.sub(r"\s+", " ", text or "").strip() or "(untitled)"
    return title if len(title) <= limit else title[: limit - 1] + "…"


def _connect() -> sqlite3.Connection:
    path = db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    return conn


def record(thread_id: str, mode: str, workspace: str, first_message: str | None = None) -> None:
    """Create or touch a conversation. The title comes from its first user message."""
    now = time.time()
    if not persistent():
        with _lock:
            row = _memory_index.setdefault(thread_id, {
                "thread_id": thread_id, "title": make_title(first_message or ""),
                "mode": mode, "workspace": workspace, "created_at": now, "updated_at": now})
            row.update(mode=mode, updated_at=now)
        return
    with _lock:
        conn = _connect()
        try:
            with conn:  # commits (the context manager does not close the connection)
                conn.execute(
                    "INSERT INTO conversations(thread_id, title, mode, workspace, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(thread_id) DO UPDATE SET mode=excluded.mode, updated_at=excluded.updated_at",
                    (thread_id, make_title(first_message or ""), mode, workspace, now, now))
        finally:
            conn.close()


def _in_scope(row_workspace: str, scope: str) -> bool:
    """A conversation belongs to `scope` if its workspace IS the scope (shared mode) or lives
    directly inside it (one directory per conversation)."""
    if row_workspace == scope:
        return True
    try:
        return Path(row_workspace).parent == Path(scope)
    except (OSError, ValueError):
        return False


def list_conversations(workspace: str, limit: int = 50) -> list[dict]:
    """Most recently used first. `workspace` is the scope (see tools.workspace_scope)."""
    if not persistent():
        with _lock:
            rows = [dict(r) for r in _memory_index.values() if _in_scope(r["workspace"], workspace)]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)[:limit]
    with _lock:
        conn = _connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()]
        finally:
            conn.close()
    return [r for r in rows if _in_scope(r["workspace"], workspace)][:limit]


def get_conversation(thread_id: str) -> dict | None:
    if not persistent():
        with _lock:
            row = _memory_index.get(thread_id)
        return dict(row) if row else None
    with _lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM conversations WHERE thread_id = ?",
                               (thread_id,)).fetchone()
        finally:
            conn.close()
    return dict(row) if row else None


def find_conversation(workspace: str, ref: str) -> dict | None:
    """Resolve '3' (position in the list), a full id, or a unique id prefix."""
    ref = (ref or "").strip()
    rows = list_conversations(workspace, limit=500)
    if ref.isdigit() and 1 <= int(ref) <= len(rows):
        return rows[int(ref) - 1]
    matches = [r for r in rows if r["thread_id"] == ref] or \
              [r for r in rows if ref and r["thread_id"].startswith(ref)]
    return matches[0] if len(matches) == 1 else None


def latest_conversation(workspace: str) -> dict | None:
    rows = list_conversations(workspace, limit=1)
    return rows[0] if rows else None


def delete_conversation(thread_id: str) -> None:
    """Remove the index entry and all checkpoints of the conversation."""
    if not persistent():
        with _lock:
            _memory_index.pop(thread_id, None)
    else:
        with _lock:
            conn = _connect()
            try:
                with conn:
                    conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
            finally:
                conn.close()
    try:
        get_checkpointer().delete_thread(thread_id)
    except Exception:  # best effort: an index-less thread is invisible anyway
        pass


def format_age(timestamp: float, now: float | None = None) -> str:
    seconds = max(0, int((now or time.time()) - timestamp))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 7 * 86400:
        return f"{seconds // 86400}d ago"
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))