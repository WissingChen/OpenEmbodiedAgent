"""Task planning tool — lets the LLM decompose complex goals into sub-tasks.

The tool writes a structured task plan to ``TASK.md`` in the workspace.
``ContextBuilder`` already loads ``TASK.md`` when it exists, so the LLM
will see the current plan on every subsequent turn without any extra work.

TASK.md format
--------------
The file contains a YAML-fenced block with the following structure:

    ```yaml
    goal: "Make a cup of coffee"
    status: in_progress   # pending | in_progress | completed | failed
    created_at: "2026-04-08T15:00:00Z"
    updated_at: "2026-04-08T15:01:00Z"
    steps:
      - id: 1
        description: "Navigate to the kitchen"
        status: completed
        result: "Arrived at kitchen zone."
      - id: 2
        description: "Pick up the coffee mug"
        status: in_progress
        result: null
      - id: 3
        description: "Place mug under coffee machine"
        status: pending
        result: null
    ```
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PhyAgentOS.agent.tools.base import Tool

_FENCE_OPEN  = "```yaml"
_FENCE_CLOSE = "```"
_BLOCK_RE    = re.compile(r"```yaml\s*\n(.*?)\n```", re.DOTALL)

_VALID_STATUSES = {"pending", "in_progress", "completed", "failed"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _render_task_md(plan: dict[str, Any]) -> str:
    """Serialise *plan* to TASK.md content (YAML-fenced Markdown)."""
    import yaml  # type: ignore[import-untyped]

    body = yaml.dump(plan, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return (
        "# Task Plan\n\n"
        "Managed by PhyAgentOS TaskPlanningTool. "
        "Edit via the `manage_task` tool, not manually.\n\n"
        f"{_FENCE_OPEN}\n{body.rstrip()}\n{_FENCE_CLOSE}\n"
    )


def _load_task_md(task_file: Path) -> dict[str, Any] | None:
    """Parse TASK.md and return the plan dict, or None if absent/invalid."""
    if not task_file.exists():
        return None
    content = task_file.read_text(encoding="utf-8")
    match = _BLOCK_RE.search(content)
    if not match:
        return None
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(match.group(1))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


class TaskPlanningTool(Tool):
    """Create, update, and complete structured task plans in TASK.md.

    Supported operations
    --------------------
    ``create``
        Start a new plan (overwrites any existing one).
    ``update_step``
        Mark a step as completed/failed and record its result.
    ``complete``
        Mark the whole plan as completed or failed.
    ``get``
        Return the current plan as a formatted string (read-only).
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "manage_task"

    @property
    def description(self) -> str:
        return (
            "Create and manage a structured task plan in TASK.md. "
            "Use 'create' to decompose a complex goal into ordered steps. "
            "Use 'update_step' after each physical action to record the result. "
            "Use 'complete' when the whole goal is done. "
            "Use 'get' to read the current plan."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "update_step", "complete", "get"],
                    "description": "The operation to perform.",
                },
                "goal": {
                    "type": "string",
                    "description": "[create] High-level goal description.",
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "[create] Ordered list of step descriptions.",
                },
                "step_id": {
                    "type": "integer",
                    "description": "[update_step] 1-based step index to update.",
                },
                "step_status": {
                    "type": "string",
                    "enum": ["completed", "failed", "in_progress"],
                    "description": "[update_step] New status for the step.",
                },
                "step_result": {
                    "type": "string",
                    "description": "[update_step] Result message for the step.",
                },
                "plan_status": {
                    "type": "string",
                    "enum": ["completed", "failed"],
                    "description": "[complete] Final status for the whole plan.",
                },
            },
            "required": ["operation"],
        }

    async def execute(
        self,
        operation: str,
        goal: str | None = None,
        steps: list[str] | None = None,
        step_id: int | None = None,
        step_status: str | None = None,
        step_result: str | None = None,
        plan_status: str | None = None,
    ) -> str:
        task_file = self.workspace / "TASK.md"

        if operation == "create":
            return self._create(task_file, goal, steps)
        if operation == "update_step":
            return self._update_step(task_file, step_id, step_status, step_result)
        if operation == "complete":
            return self._complete(task_file, plan_status)
        if operation == "get":
            return self._get(task_file)
        return f"Error: Unknown operation '{operation}'."

    # ── operations ────────────────────────────────────────────────────────────

    def _create(
        self,
        task_file: Path,
        goal: str | None,
        steps: list[str] | None,
    ) -> str:
        if not goal:
            return "Error: 'goal' is required for the 'create' operation."
        if not steps:
            return "Error: 'steps' list is required for the 'create' operation."

        now = _utcnow()
        plan: dict[str, Any] = {
            "goal":       goal,
            "status":     "in_progress",
            "created_at": now,
            "updated_at": now,
            "steps": [
                {
                    "id":          i + 1,
                    "description": desc,
                    "status":      "pending",
                    "result":      None,
                }
                for i, desc in enumerate(steps)
            ],
        }
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(_render_task_md(plan), encoding="utf-8")
        return (
            f"Task plan created with {len(steps)} step(s). "
            f"Goal: '{goal}'. TASK.md updated."
        )

    def _update_step(
        self,
        task_file: Path,
        step_id: int | None,
        step_status: str | None,
        step_result: str | None,
    ) -> str:
        if step_id is None:
            return "Error: 'step_id' is required for 'update_step'."
        if step_status not in ("completed", "failed", "in_progress"):
            return "Error: 'step_status' must be 'completed', 'failed', or 'in_progress'."

        plan = _load_task_md(task_file)
        if plan is None:
            return "Error: No task plan found. Use 'create' first."

        steps = plan.get("steps", [])
        target = next((s for s in steps if s.get("id") == step_id), None)
        if target is None:
            return f"Error: Step {step_id} not found in the current plan."

        target["status"] = step_status
        if step_result is not None:
            target["result"] = step_result
        plan["updated_at"] = _utcnow()

        # Auto-advance: if all steps are done, mark plan as completed
        all_done = all(s.get("status") in ("completed", "failed") for s in steps)
        if all_done and plan.get("status") == "in_progress":
            any_failed = any(s.get("status") == "failed" for s in steps)
            plan["status"] = "failed" if any_failed else "completed"

        task_file.write_text(_render_task_md(plan), encoding="utf-8")
        return (
            f"Step {step_id} updated to '{step_status}'. "
            + (f"Result: {step_result}. " if step_result else "")
            + f"Plan status: {plan['status']}."
        )

    def _complete(self, task_file: Path, plan_status: str | None) -> str:
        if plan_status not in ("completed", "failed"):
            return "Error: 'plan_status' must be 'completed' or 'failed'."

        plan = _load_task_md(task_file)
        if plan is None:
            return "Error: No task plan found. Use 'create' first."

        plan["status"]     = plan_status
        plan["updated_at"] = _utcnow()
        task_file.write_text(_render_task_md(plan), encoding="utf-8")
        return f"Task plan marked as '{plan_status}'. TASK.md updated."

    def _get(self, task_file: Path) -> str:
        plan = _load_task_md(task_file)
        if plan is None:
            return "No active task plan found."

        steps = plan.get("steps", [])
        lines = [
            f"Goal: {plan.get('goal', '(unknown)')}",
            f"Status: {plan.get('status', '(unknown)')}",
            f"Updated: {plan.get('updated_at', '')}",
            "",
            "Steps:",
        ]
        for s in steps:
            icon = {"completed": "✅", "failed": "❌", "in_progress": "🔄", "pending": "⬜"}.get(
                s.get("status", "pending"), "⬜"
            )
            result_str = f" → {s['result']}" if s.get("result") else ""
            lines.append(f"  {icon} [{s['id']}] {s['description']}{result_str}")
        return "\n".join(lines)
