# Track 3b: AgileX Piper — ReKep Constraint-Based Single-Arm Manipulation

> Owner: TBD | Status: Planning | Priority: High | Hardware: AgileX Piper 6-DOF arm

## 1. Objective

Build a driver for the **AgileX Piper** 6-DOF collaborative robot arm that uses the **ReKep constraint-solving paradigm** to convert high-level geometric constraints from `ACTION.md` into joint trajectories. Piper communicates over **CAN bus** via the official Python SDK.

## 2. Scope

### In Scope
- Implement `PiperDriver(BaseDriver)` in `hal/drivers/piper_driver.py`
- ReKep constraint solver reuse (shared `hal/drivers/rekep_solver.py` with Track 3)
- Single-arm manipulation: pick-and-place, push, point-to
- AgileX Piper SDK integration for real-time joint / Cartesian control
- Gripper open/close control via the same SDK
- Write `hal/profiles/piper.md` describing the arm's capabilities and joint limits
- Mock mode for testing without physical hardware

### Out of Scope
- Dual-arm coordination (Piper is a single-arm robot)
- Autonomous navigation (fixed-base)
- Vision processing (handled upstream by MCP Vision Server, Track 8)

## 3. Files to Create

| File | Action | Description |
|------|--------|-------------|
| `hal/drivers/piper_driver.py` | Create | `PiperDriver(BaseDriver)` implementation |
| `hal/drivers/rekep_solver.py` | Create (shared with Track 3) | ReKep QP solver module |
| `hal/profiles/piper.md` | Create | EMBODIED.md profile for AgileX Piper |
| `tests/test_piper_driver.py` | Create | Driver tests with SDK mock |

## 4. EMBODIED.md Profile Requirements

```markdown
- Robot model: AgileX Piper
- DOF: 6 (joints J1–J6) + 1 (gripper)
- Communication: CAN bus (USB-CAN adapter or built-in CAN)
- SDK: piper_sdk (official Python package)
- Workspace: reachability sphere ~580 mm from base
- Max payload: 1.5 kg
- Joint limits: [refer to official datasheet]
- End-effector: parallel jaw gripper (max opening ~80 mm)
- Supported actions: move_to, pick_up, put_down, push, point_to, open_gripper, close_gripper
- Safety: max joint velocity, max Cartesian speed, force/torque limits
```

## 5. Action Specification

| Action | Parameters | Expected Behavior |
|--------|-----------|-------------------|
| `move_to` | `x, y, z, rx, ry, rz` (mm + degrees) | Move EE to target Cartesian pose |
| `pick_up` | `target: str` | Approach object, close gripper, lift 10 cm |
| `put_down` | `target: str, location: str` | Move to location, open gripper |
| `push` | `target: str, direction: str` | Apply lateral impulse to object |
| `point_to` | `target: str` | Orient EE toward named object |
| `open_gripper` | `width_mm: float` | Set gripper opening to specified width |
| `close_gripper` | `force_n: float` | Close gripper with specified force |

### ReKep Constraint Format (shared with Track 3)

```json
{
  "action_type": "rekep_constraint",
  "parameters": {
    "keypoints": [
      {"name": "ee", "constraint": "align_with", "target": "cup_rim", "axis": "z_up"},
      {"name": "ee", "constraint": "maintain_distance", "target": "cup_center", "distance_mm": 50}
    ],
    "objective": "minimize_joint_displacement"
  }
}
```

## 6. Milestones & Acceptance Criteria

### Milestone M1: Driver Scaffold (no hardware required)
- [ ] `PiperDriver` class exists in `hal/drivers/piper_driver.py` and imports without error
- [ ] `hal/profiles/piper.md` exists and is non-empty
- [ ] Driver is registered in `hal/drivers/__init__.py` as `"piper"`
- [ ] `pytest tests/test_hal_base_driver.py -k piper` passes with mock (all 10 contract tests green)

### Milestone M2: Mock Execution (no hardware required)
- [ ] `move_to(x=100, y=0, z=300)` returns a success string without raising
- [ ] `pick_up(target="apple")` returns success when "apple" is in the scene
- [ ] `pick_up(target="ghost")` returns an error string (object not found)
- [ ] `get_scene()` reflects object state as `"held"` after successful `pick_up`
- [ ] `close_gripper(force_n=10)` returns success string

### Milestone M3: Hardware Integration (physical Piper required)
- [ ] `hal_watchdog.py --driver piper` starts without error and prints profile info
- [ ] CAN bus connection established; SDK reports joint positions
- [ ] `move_to` moves the arm to the target within ±5 mm tolerance
- [ ] `pick_up` successfully grasps a 50 g foam cube placed at a known position
- [ ] `put_down` releases the cube at a specified location
- [ ] Emergency stop: `CTRL+C` halts the arm smoothly

### Milestone M4: ReKep Integration
- [ ] `rekep_constraint` action type is parsed and forwarded to `rekep_solver.py`
- [ ] Solver produces a valid joint trajectory (no joint-limit violation)
- [ ] Trajectory executes on the physical arm without jerking or stopping

### Milestone M5: Full Pipeline Closed-Loop
- [ ] User types "pick up the apple and put it on the shelf" in CLI
- [ ] Planner generates correct `ACTION.md` (action_type: pick_up, target: apple)
- [ ] Critic validates against `hal/profiles/piper.md` (workspace bounds check passes)
- [ ] Watchdog dispatches to `PiperDriver.execute_action()`
- [ ] Physical arm picks up the apple
- [ ] `ENVIRONMENT.md` is updated: apple.location = "shelf"
- [ ] Agent confirms completion to user

## 7. Dependencies

- `piper_sdk` — official AgileX Python SDK (install via `pip install piper-sdk` or clone from GitHub)
- `can` / `python-can` — Python CAN bus library
- `scipy` or `cvxpy` — quadratic programming for ReKep solver
- `numpy` — kinematics calculations
- USB-CAN adapter (e.g., CANable) or built-in CAN interface

## 8. SDK & Communication Architecture

```
ACTION.md
    │
    ▼
┌──────────────────┐
│ piper_driver.py   │  ← Parses action, calls piper_sdk
│                   │  ← Reads EMBODIED.md for joint limits
└──────┬───────────┘
       │ piper_sdk
       ▼
┌──────────────────┐
│ CAN bus           │  ← USB-CAN adapter
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ AgileX Piper arm  │  ← Physical hardware
└──────┬───────────┘
       │ joint feedback
       ▼
┌──────────────────┐
│ piper_driver.py   │  ← get_scene() returns updated positions
└──────────────────┘
```

## 9. Safety Notes

- **Joint velocity limit**: Do not exceed 50% of max speed during initial testing
- **Workspace guard**: `EMBODIED.md` defines allowed Cartesian box; Critic rejects out-of-bound targets
- **Force limit**: Gripper force < 30 N during pick to avoid crushing fragile objects
- **Emergency stop**: Implement hardware E-stop trigger via SDK's emergency stop API
