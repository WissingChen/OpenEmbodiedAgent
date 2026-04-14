"""
hal/action_queue.py

ACTION.md queue manager — replaces the single-entry overwrite pattern with a
proper status-tracked queue.

Queue format (JSON array inside a ```json … ``` fence in ACTION.md):

    ```json
    [
      {
        "action_id":   "act_1712567890_001",
        "action_type": "semantic_navigate",
        "parameters":  {"robot_id": "go2_edu_001", "target_id": "apple_01"},
        "status":      "pending",   // pending | running | completed | failed
        "result_msg":  null,
        "created_at":  "2026-04-08T15:00:00Z",
        "updated_at":  "2026-04-08T15:00:00Z"
      }
    ]
    ```

Concurrency safety
------------------
All reads and writes go through ``ActionQueueLock``, which acquires a
``filelock.FileLock`` on ``<action_file>.lock`` before touching the file.
If the ``filelock`` package is not installed the lock degrades gracefully to
a no-op (single-process usage is still safe).
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── optional file-lock dependency ────────────────────────────────────────────
try:
    from filelock import FileLock as _FileLock  # type: ignore[import-untyped]
    _FILELOCK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FILELOCK_AVAILABLE = False

_FENCE_OPEN = "```json"
_FENCE_CLOSE = "```"
_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)

# Valid status transitions
_VALID_STATUSES = {"pending", "running", "completed", "failed"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _generate_action_id() -> str:
    ts = int(time.time() * 1000)
    return f"act_{ts}"


@contextmanager
def _file_lock(lock_path: Path):
    """Acquire a file-system lock if filelock is available, else no-op."""
    if _FILELOCK_AVAILABLE:
        lock = _FileLock(str(lock_path), timeout=10)
        with lock:
            yield
    else:
        yield


# ── low-level read / write ────────────────────────────────────────────────────

def _read_queue_raw(action_file: Path) -> list[dict[str, Any]]:
    """Parse the JSON array from ACTION.md. Returns [] on any error."""
    if not action_file.exists():
        return []
    content = action_file.read_text(encoding="utf-8").strip()
    if not content:
        return []
    match = _BLOCK_RE.search(content)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _write_queue_raw(action_file: Path, queue: list[dict[str, Any]]) -> None:
    """Serialise *queue* back to ACTION.md."""
    action_file.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(queue, indent=2, ensure_ascii=False)
    content = (
        "# Action Queue\n\n"
        "Managed by PhyAgentOS. Do not edit manually while the system is running.\n\n"
        f"{_FENCE_OPEN}\n{body}\n{_FENCE_CLOSE}\n"
    )
    action_file.write_text(content, encoding="utf-8")


# ── public API ────────────────────────────────────────────────────────────────

def enqueue_action(
    action_file: Path,
    action_type: str,
    parameters: dict[str, Any],
    *,
    action_id: str | None = None,
) -> str:
    """Append a new *pending* action to the queue and return its ``action_id``.

    Thread/process safe when ``filelock`` is installed.
    """
    aid = action_id or _generate_action_id()
    now = _utcnow()
    entry: dict[str, Any] = {
        "action_id":   aid,
        "action_type": action_type,
        "parameters":  parameters,
        "status":      "pending",
        "result_msg":  None,
        "created_at":  now,
        "updated_at":  now,
    }
    lock_path = action_file.with_suffix(".lock")
    with _file_lock(lock_path):
        queue = _read_queue_raw(action_file)
        queue.append(entry)
        _write_queue_raw(action_file, queue)
    return aid


def get_action_status(
    action_file: Path,
    action_id: str,
) -> tuple[str, str | None]:
    """Return ``(status, result_msg)`` for *action_id*.

    Returns ``("not_found", None)`` if the entry does not exist.
    """
    lock_path = action_file.with_suffix(".lock")
    with _file_lock(lock_path):
        queue = _read_queue_raw(action_file)
    for entry in queue:
        if entry.get("action_id") == action_id:
            return entry.get("status", "unknown"), entry.get("result_msg")
    return "not_found", None


def update_action_status(
    action_file: Path,
    action_id: str,
    status: str,
    result_msg: str | None = None,
) -> bool:
    """Update the status (and optionally result_msg) of an existing action.

    Returns ``True`` if the entry was found and updated, ``False`` otherwise.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}. Must be one of {_VALID_STATUSES}")
    lock_path = action_file.with_suffix(".lock")
    with _file_lock(lock_path):
        queue = _read_queue_raw(action_file)
        updated = False
        for entry in queue:
            if entry.get("action_id") == action_id:
                entry["status"] = status
                entry["updated_at"] = _utcnow()
                if result_msg is not None:
                    entry["result_msg"] = result_msg
                updated = True
                break
        if updated:
            _write_queue_raw(action_file, queue)
    return updated


def pop_next_pending(
    action_file: Path,
) -> dict[str, Any] | None:
    """Atomically claim the oldest *pending* action by setting it to *running*.

    Returns the action dict (with updated status) or ``None`` if the queue is
    empty or has no pending entries.
    """
    lock_path = action_file.with_suffix(".lock")
    with _file_lock(lock_path):
        queue = _read_queue_raw(action_file)
        for entry in queue:
            if entry.get("status") == "pending":
                entry["status"] = "running"
                entry["updated_at"] = _utcnow()
                _write_queue_raw(action_file, queue)
                return dict(entry)
    return None


def purge_completed(action_file: Path) -> int:
    """Remove all *completed* and *failed* entries from the queue.

    Returns the number of entries removed.
    """
    lock_path = action_file.with_suffix(".lock")
    with _file_lock(lock_path):
        queue = _read_queue_raw(action_file)
        before = len(queue)
        queue = [e for e in queue if e.get("status") not in ("completed", "failed")]
        _write_queue_raw(action_file, queue)
    return before - len(queue)


def read_queue(action_file: Path) -> list[dict[str, Any]]:
    """Return a snapshot of the full queue (read-only, no lock held after return)."""
    lock_path = action_file.with_suffix(".lock")
    with _file_lock(lock_path):
        return list(_read_queue_raw(action_file))
