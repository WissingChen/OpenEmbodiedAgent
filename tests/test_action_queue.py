"""Tests for hal/action_queue.py — queue-based ACTION.md management."""

from __future__ import annotations

from pathlib import Path

import pytest

from hal.action_queue import (
    enqueue_action,
    get_action_status,
    pop_next_pending,
    purge_completed,
    read_queue,
    update_action_status,
)


# ── enqueue ───────────────────────────────────────────────────────────────────

def test_enqueue_creates_file_with_pending_entry(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    aid = enqueue_action(action_file, "move_to", {"x": 1.0, "y": 2.0})

    assert action_file.exists()
    queue = read_queue(action_file)
    assert len(queue) == 1
    entry = queue[0]
    assert entry["action_id"] == aid
    assert entry["action_type"] == "move_to"
    assert entry["parameters"] == {"x": 1.0, "y": 2.0}
    assert entry["status"] == "pending"
    assert entry["result_msg"] is None


def test_enqueue_multiple_actions_preserves_order(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    id1 = enqueue_action(action_file, "move_to", {"x": 1.0})
    id2 = enqueue_action(action_file, "pick_up", {"target_id": "apple"})
    id3 = enqueue_action(action_file, "move_to", {"x": 3.0})

    queue = read_queue(action_file)
    assert [e["action_id"] for e in queue] == [id1, id2, id3]


def test_enqueue_accepts_explicit_action_id(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    aid = enqueue_action(action_file, "stop", {}, action_id="my_custom_id")
    assert aid == "my_custom_id"
    queue = read_queue(action_file)
    assert queue[0]["action_id"] == "my_custom_id"


# ── get_action_status ─────────────────────────────────────────────────────────

def test_get_action_status_returns_pending_for_new_entry(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    aid = enqueue_action(action_file, "move_to", {})
    status, result = get_action_status(action_file, aid)
    assert status == "pending"
    assert result is None


def test_get_action_status_returns_not_found_for_unknown_id(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    status, result = get_action_status(action_file, "nonexistent_id")
    assert status == "not_found"
    assert result is None


def test_get_action_status_returns_not_found_when_file_missing(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    status, result = get_action_status(action_file, "any_id")
    assert status == "not_found"


# ── update_action_status ──────────────────────────────────────────────────────

def test_update_action_status_to_completed(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    aid = enqueue_action(action_file, "move_to", {})
    updated = update_action_status(action_file, aid, "completed", result_msg="Done.")
    assert updated is True
    status, result = get_action_status(action_file, aid)
    assert status == "completed"
    assert result == "Done."


def test_update_action_status_to_failed(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    aid = enqueue_action(action_file, "pick_up", {})
    update_action_status(action_file, aid, "failed", result_msg="Object not reachable.")
    status, result = get_action_status(action_file, aid)
    assert status == "failed"
    assert result == "Object not reachable."


def test_update_action_status_returns_false_for_unknown_id(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    result = update_action_status(action_file, "ghost_id", "completed")
    assert result is False


def test_update_action_status_rejects_invalid_status(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    aid = enqueue_action(action_file, "move_to", {})
    with pytest.raises(ValueError, match="Invalid status"):
        update_action_status(action_file, aid, "flying")


# ── pop_next_pending ──────────────────────────────────────────────────────────

def test_pop_next_pending_claims_oldest_entry(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    id1 = enqueue_action(action_file, "move_to", {"x": 1.0})
    id2 = enqueue_action(action_file, "pick_up", {})

    claimed = pop_next_pending(action_file)
    assert claimed is not None
    assert claimed["action_id"] == id1
    assert claimed["status"] == "running"

    # The file should reflect the running status
    status, _ = get_action_status(action_file, id1)
    assert status == "running"

    # id2 is still pending
    status2, _ = get_action_status(action_file, id2)
    assert status2 == "pending"


def test_pop_next_pending_returns_none_when_queue_empty(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    assert pop_next_pending(action_file) is None


def test_pop_next_pending_skips_running_entries(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    id1 = enqueue_action(action_file, "move_to", {})
    # Manually set id1 to running
    update_action_status(action_file, id1, "running")
    id2 = enqueue_action(action_file, "pick_up", {})

    claimed = pop_next_pending(action_file)
    assert claimed is not None
    assert claimed["action_id"] == id2


# ── purge_completed ───────────────────────────────────────────────────────────

def test_purge_completed_removes_done_entries(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    id1 = enqueue_action(action_file, "move_to", {})
    id2 = enqueue_action(action_file, "pick_up", {})
    id3 = enqueue_action(action_file, "stop", {})

    update_action_status(action_file, id1, "completed")
    update_action_status(action_file, id3, "failed")

    removed = purge_completed(action_file)
    assert removed == 2

    queue = read_queue(action_file)
    assert len(queue) == 1
    assert queue[0]["action_id"] == id2
    assert queue[0]["status"] == "pending"


def test_purge_completed_on_empty_file_returns_zero(tmp_path: Path) -> None:
    action_file = tmp_path / "ACTION.md"
    assert purge_completed(action_file) == 0


# ── full lifecycle ────────────────────────────────────────────────────────────

def test_full_action_lifecycle(tmp_path: Path) -> None:
    """Simulate: enqueue → watchdog claims → watchdog completes → agent reads result."""
    action_file = tmp_path / "ACTION.md"

    # Agent enqueues
    aid = enqueue_action(action_file, "semantic_navigate", {"robot_id": "go2_edu_001", "target_id": "apple_01"})

    # Watchdog claims
    action = pop_next_pending(action_file)
    assert action["action_id"] == aid
    assert action["status"] == "running"

    # Watchdog completes
    update_action_status(action_file, aid, "completed", result_msg="Arrived at apple_01.")

    # Agent reads result
    status, result = get_action_status(action_file, aid)
    assert status == "completed"
    assert result == "Arrived at apple_01."

    # Purge
    removed = purge_completed(action_file)
    assert removed == 1
    assert read_queue(action_file) == []
