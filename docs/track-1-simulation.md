# Track 1: Simulation Environment — Robotic Arm Grasping

> Owner: TBD | Status: Planning | Priority: High (validates full pipeline)

## 1. Objective

Build a PyBullet-based simulation environment where a virtual robotic arm can execute pick-and-place tasks. This track serves as the **reference implementation** of `BaseDriver` and the primary test bed for the entire Markdown-based control pipeline.

## 2. Scope

### In Scope
- Implement `SimulationDriver(BaseDriver)` in `hal/drivers/simulation_driver.py`
- Create a PyBullet scene with a table, objects, and a simulated robotic arm (e.g., Franka Panda URDF)
- Support actions: `move_to`, `pick_up`, `put_down`, `push`, `point_to`
- Inverse kinematics for end-effector positioning
- Collision detection and physics-based grasping
- Write `hal/profiles/simulation.md` describing the simulated arm's capabilities
- Unit tests for each action in headless (DIRECT) mode

### Out of Scope
- Real hardware communication
- ROS2 integration (Track 3/4)
- Multi-arm coordination (Track 3)

## 3. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `hal/drivers/simulation_driver.py` | Create | `SimulationDriver(BaseDriver)` implementation |
| `hal/simulation/pybullet_sim.py` | Modify | Refactor as internal engine used by SimulationDriver |
| `hal/simulation/scene_io.py` | Keep | Shared ENVIRONMENT.md I/O (already exists) |
| `hal/profiles/simulation.md` | Create | EMBODIED.md profile for the simulated arm |
| `tests/test_simulation_driver.py` | Create | Driver-specific tests |

## 4. EMBODIED.md Profile Requirements

The `hal/profiles/simulation.md` must declare:
- Robot model name (e.g., Franka Panda)
- Degrees of freedom (7-DOF arm + 2-DOF gripper)
- Workspace bounds (x, y, z ranges in cm)
- Max payload (kg)
- Supported action types with parameter schemas
- Physical constraints (max velocity, force limits)

## 5. Action Specification

| Action | Parameters | Expected Behavior |
|--------|-----------|-------------------|
| `move_to` | `x, y, z` (cm) | Move end-effector to target position via IK |
| `pick_up` | `target: str` | Approach object, close gripper, lift |
| `put_down` | `target: str, location: str` | Move to location, open gripper, release |
| `push` | `target: str, direction: str` | Apply lateral force to object |
| `point_to` | `target: str` | Orient end-effector toward object |

## 6. Milestones & Acceptance Criteria

### Milestone M1: Driver Scaffold
- [ ] `SimulationDriver` class exists and imports without error
- [ ] Driver is registered in `hal/drivers/__init__.py` as `"simulation"`
- [ ] `pytest tests/test_hal_base_driver.py -k simulation` — all 10 contract tests green

### Milestone M2: Basic Actions (headless)
- [ ] `move_to(x=10, y=0, z=0)` returns success string and EE position updates
- [ ] `pick_up(target="red_apple")` lifts the object; `get_scene()` shows `location: "held"`
- [ ] `pick_up(target="ghost")` returns error string without raising
- [ ] `put_down(target="red_apple", location="table")` places object; physics settles without falling through floor
- [ ] `push(target="red_apple", direction="forward")` — object moves; new position in `get_scene()`

### Milestone M3: Environment Closed-Loop
- [ ] `ENVIRONMENT.md` is correctly updated after every action execution
- [ ] Reloading the scene from `ENVIRONMENT.md` and re-running actions produces consistent results
- [ ] All tests pass in headless (DIRECT) mode — no GUI required for CI

### Milestone M4: Full Pipeline
- [ ] `hal_watchdog.py --driver simulation` starts, copies `hal/profiles/simulation.md` to workspace, loads scene
- [ ] User types "pick up the apple" in CLI → Planner writes `ACTION.md` → Watchdog executes → `ENVIRONMENT.md` shows `location: "held"` → Agent confirms
- [ ] GUI mode (`--gui`) renders the scene for visual debugging

## 7. Dependencies

- `pybullet` (`pip install pybullet`)
- `pybullet_data` (bundled with pybullet)
- Robot URDF files (Franka Panda available in pybullet_data)
