# Track 7 Implementation — File Change Log

> Track 7: Framework Core | Date: 2026-03-15

This document lists all files created or modified as part of the Track 7 (Framework Core) implementation.

## New Files

| File | Description |
|------|-------------|
| `hal/__init__.py` | Package marker for the HAL module |
| `hal/base_driver.py` | Abstract base class `BaseDriver` — the contract every robot driver must implement (4 abstract methods: `get_profile_path`, `load_scene`, `execute_action`, `get_scene`) |
| `hal/drivers/__init__.py` | Driver registry — maps short names (e.g. `"simulation"`) to class paths; provides `load_driver()` and `list_drivers()` functions |
| `hal/drivers/simulation_driver.py` | `SimulationDriver(BaseDriver)` — reference implementation wrapping the existing PyBullet simulator |
| `hal/profiles/simulation.md` | EMBODIED.md profile for the simulated arm — declares DOF, supported actions, physical constraints |
| `hal/simulation/__init__.py` | Public API for the simulation sub-package (re-exports `PyBulletSimulator`, `load_scene_from_md`, `save_scene_to_md`) |
| `hal/simulation/pybullet_sim.py` | PyBullet physics engine wrapper — spawns world, executes actions, returns scene state |
| `hal/simulation/scene_io.py` | ENVIRONMENT.md read/write utilities — parses and serialises JSON code blocks |
| `tests/test_hal_base_driver.py` | Contract test suite — parametrised over all registered drivers; verifies BaseDriver invariants |

## Modified Files

| File | Change | Description |
|------|--------|-------------|
| `hal/hal_watchdog.py` | **Rewritten** | Now uses `--driver` flag to dynamically load any registered driver via the registry; auto-copies the driver's profile to `workspace/EMBODIED.md` on startup |
| `OEA/agent/context.py` | **Modified** | Split `BOOTSTRAP_FILES` into core files (always loaded) + `EMBODIED_FILES` (loaded when present); `_load_bootstrap_files()` now iterates both lists, enabling new tracks to add protocol files without touching this code |

## Unchanged Files (referenced but not modified)

| File | Role |
|------|------|
| `OEA/agent/tools/embodied.py` | `EmbodiedActionTool` — Critic validation + ACTION.md writing (created in earlier iteration, unchanged in Track 7) |
| `OEA/agent/loop.py` | Agent loop — registers `EmbodiedActionTool` (modified in earlier iteration, unchanged in Track 7) |
| `OEA/templates/EMBODIED.md` | Default EMBODIED.md template (created in earlier iteration) |
| `OEA/templates/ENVIRONMENT.md` | Default ENVIRONMENT.md template (created in earlier iteration) |
| `OEA/templates/ACTION.md` | Empty action buffer template (created in earlier iteration) |
| `OEA/templates/LESSONS.md` | Lessons learned template (created in earlier iteration) |
| `OEA/templates/HEARTBEAT.md` | Heartbeat tasks with embodied checks (modified in earlier iteration) |
