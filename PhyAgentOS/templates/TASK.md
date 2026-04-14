# Task Plan

Managed by PhyAgentOS `manage_task` tool. Do not edit manually while the system is running.

Use the `manage_task` tool to:
- `create`      — decompose a new goal into ordered steps
- `update_step` — record the result of a completed/failed step
- `complete`    — mark the whole plan as done
- `get`         — read the current plan

When this file contains an active plan, the agent will read it on every turn
and use it to track progress across multi-step embodied tasks.

```yaml
goal: "(no active plan)"
status: pending
created_at: null
updated_at: null
steps: []
```
