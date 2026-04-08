"""Tests for TaskPlanningTool (agent/tools/task.py)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from PhyAgentOS.agent.tools.task import TaskPlanningTool, _load_task_md


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── create ────────────────────────────────────────────────────────────────────

def test_create_writes_task_md(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    result = _run(tool.execute(
        operation="create",
        goal="Navigate to the kitchen and pick up the apple",
        steps=["Navigate to kitchen", "Locate apple", "Pick up apple"],
    ))
    assert "3 step" in result
    task_file = tmp_path / "TASK.md"
    assert task_file.exists()
    plan = _load_task_md(task_file)
    assert plan is not None
    assert plan["goal"] == "Navigate to the kitchen and pick up the apple"
    assert plan["status"] == "in_progress"
    assert len(plan["steps"]) == 3
    assert plan["steps"][0]["id"] == 1
    assert plan["steps"][0]["status"] == "pending"


def test_create_requires_goal(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    result = _run(tool.execute(operation="create", steps=["step1"]))
    assert "Error" in result


def test_create_requires_steps(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    result = _run(tool.execute(operation="create", goal="Do something"))
    assert "Error" in result


# ── update_step ───────────────────────────────────────────────────────────────

def test_update_step_marks_completed(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(
        operation="create",
        goal="Test goal",
        steps=["Step A", "Step B"],
    ))
    result = _run(tool.execute(
        operation="update_step",
        step_id=1,
        step_status="completed",
        step_result="Step A done successfully.",
    ))
    assert "completed" in result

    plan = _load_task_md(tmp_path / "TASK.md")
    assert plan["steps"][0]["status"] == "completed"
    assert plan["steps"][0]["result"] == "Step A done successfully."
    assert plan["steps"][1]["status"] == "pending"


def test_update_step_auto_completes_plan_when_all_done(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(operation="create", goal="G", steps=["S1", "S2"]))
    _run(tool.execute(operation="update_step", step_id=1, step_status="completed"))
    _run(tool.execute(operation="update_step", step_id=2, step_status="completed"))

    plan = _load_task_md(tmp_path / "TASK.md")
    assert plan["status"] == "completed"


def test_update_step_auto_fails_plan_when_any_step_failed(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(operation="create", goal="G", steps=["S1", "S2"]))
    _run(tool.execute(operation="update_step", step_id=1, step_status="failed", step_result="Obstacle."))
    _run(tool.execute(operation="update_step", step_id=2, step_status="completed"))

    plan = _load_task_md(tmp_path / "TASK.md")
    assert plan["status"] == "failed"


def test_update_step_returns_error_for_missing_plan(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    result = _run(tool.execute(operation="update_step", step_id=1, step_status="completed"))
    assert "Error" in result


def test_update_step_returns_error_for_invalid_step_id(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(operation="create", goal="G", steps=["S1"]))
    result = _run(tool.execute(operation="update_step", step_id=99, step_status="completed"))
    assert "Error" in result


# ── complete ──────────────────────────────────────────────────────────────────

def test_complete_marks_plan_as_completed(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(operation="create", goal="G", steps=["S1"]))
    result = _run(tool.execute(operation="complete", plan_status="completed"))
    assert "completed" in result
    plan = _load_task_md(tmp_path / "TASK.md")
    assert plan["status"] == "completed"


def test_complete_marks_plan_as_failed(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(operation="create", goal="G", steps=["S1"]))
    result = _run(tool.execute(operation="complete", plan_status="failed"))
    assert "failed" in result


def test_complete_returns_error_for_invalid_status(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(operation="create", goal="G", steps=["S1"]))
    result = _run(tool.execute(operation="complete", plan_status="unknown"))
    assert "Error" in result


# ── get ───────────────────────────────────────────────────────────────────────

def test_get_returns_formatted_plan(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    _run(tool.execute(operation="create", goal="Pick up apple", steps=["Navigate", "Grasp"]))
    result = _run(tool.execute(operation="get"))
    assert "Pick up apple" in result
    assert "Navigate" in result
    assert "Grasp" in result


def test_get_returns_message_when_no_plan(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    result = _run(tool.execute(operation="get"))
    assert "No active task plan" in result


# ── unknown operation ─────────────────────────────────────────────────────────

def test_unknown_operation_returns_error(tmp_path: Path) -> None:
    tool = TaskPlanningTool(tmp_path)
    result = _run(tool.execute(operation="fly"))
    assert "Error" in result
