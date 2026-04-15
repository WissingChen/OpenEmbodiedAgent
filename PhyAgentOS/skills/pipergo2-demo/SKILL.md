---
name: pipergo2-demo
description: Deterministic demo mapping for open sim, go to desk, then pick red cube and return to spawn.
metadata: {"PhyAgentOS":{"always":true},"nanobot":{"emoji":"🧪"}}
---

# PiperGo2 Demo Skill

This skill is a strict demo router for three sequential intents:

1. `open simulation`
2. `go to desk`
3. `pick up the red cube and return to the starting position`

## Preconditions

- HAL watchdog must already be running with:
  - driver: `pipergo2_manipulation`
  - driver-config: `examples/pipergo2_manipulation_driver.json` (or equivalent)
- If simulation may be cold, dispatch `enter_simulation` first.

## Intent Mapping (MUST follow)

### A) Open Simulation

When user input semantically means opening simulation (examples: `open simulation`, `start simulation`):

- call `execute_robot_action` with:
  - `action_type`: `enter_simulation`
  - `parameters`: `{}`
  - `reasoning`: short reason

### B) Go To Desk

When user input semantically means "go to desk" (examples: `go to desk`, `go near table`, `move to desk`):

- call `execute_robot_action` with:
  - `action_type`: `navigate_to_named`
  - `parameters`: `{"waypoint_key":"desk"}`
  - `reasoning`: short reason

### C) Pick Up Red Cube And Return To Start

When user input semantically means picking the red cube then driving back to the robot spawn / home
(examples: `pick up the red cube and return to the starting position`, `pick up the red cube and go back to the start`,
`抓起红方块回到出发点`, `grab the red cube and return home`; legacy wording such as
`pick up the red cube and move next to the rear pedestal` MUST map here as well — it no longer goes to the pedestal):

- call `execute_robot_action` **once** with:
  - `action_type`: `run_pick_place`
  - `parameters`: `{"target_color_cn":"red","execute_place":false}`
  - `reasoning`: short reason

Post-pick navigation to **robot_home** is driven by driver-config `pick_place_defaults.navigate_after_pick_xy` (not a second tool call).

## Demo Safety Rules

- Never claim success without tool result confirmation.
- Treat HAL watchdog `Result:` semantics as source of truth.
- If tool returns `Error: API not started`, do **not** auto-start; explicitly ask user to run `open simulation` first.
- Keep responses short and operational for live demo.
