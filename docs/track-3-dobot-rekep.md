# Track 3: Dobot Dual-Arm — ReKep Constraint-Based Manipulation

> Owner: TBD | Status: Planning | Priority: High

## 1. Objective

Build a driver for the Dobot (e.g., Nova 5) dual-arm robot that uses the **ReKep constraint-solving paradigm** to convert high-level geometric constraints from `ACTION.md` into joint trajectories. This is the core showcase of OEA's "constraint-based manipulation" thesis.

## 2. Scope

### In Scope
- Implement `DobotDriver(BaseDriver)` in `hal/drivers/dobot_driver.py`
- ReKep constraint solver: translate keypoint constraints (e.g., "end-effector aligned with cup rim, Z-axis up") into a quadratic program
- Dual-arm coordination: both arms can work simultaneously on a shared task
- Dobot SDK integration for real-time joint control
- ROS2 bridge for sensor data (optional, if Dobot supports ROS2)
- Write `hal/profiles/dobot_nova5.md` describing the dual-arm capabilities

### Out of Scope
- Autonomous navigation (robot is fixed-base)
- Vision processing (handled upstream by MCP Vision Server)
- Desktop pet expressions (Track 2)

## 3. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `hal/drivers/dobot_driver.py` | Create | `DobotDriver(BaseDriver)` implementation |
| `hal/drivers/rekep_solver.py` | Create | ReKep constraint solver module |
| `hal/profiles/dobot_nova5.md` | Create | EMBODIED.md profile for Dobot Nova 5 |
| `tests/test_dobot_driver.py` | Create | Driver tests (with Dobot SDK mock) |

## 4. EMBODIED.md Profile Requirements

- Robot model: Dobot Nova 5 (or specific model)
- Configuration: dual-arm, 6-DOF per arm
- Workspace bounds per arm (reachability sphere)
- Max payload per arm
- Joint limits (angle ranges, velocity limits, torque limits)
- End-effector type (gripper, suction, etc.)
- Supported action types: `move_to`, `pick_up`, `put_down`, `push`, `dual_arm_handover`
- Safety constraints: max force, collision avoidance zones

## 5. Action Specification

| Action | Parameters | Expected Behavior |
|--------|-----------|-------------------|
| `move_to` | `arm: str, x, y, z, rx, ry, rz` | Move specified arm's EE to target pose |
| `pick_up` | `arm: str, target: str` | Approach + grasp with specified arm |
| `put_down` | `arm: str, target: str, location: str` | Place object at location |
| `push` | `arm: str, target: str, direction: str` | Apply lateral force |
| `dual_arm_handover` | `from_arm: str, to_arm: str, target: str` | Transfer object between arms |

### ReKep Constraint Format in ACTION.md

The Planner can also write constraints in ReKep format:

```json
{
  "action_type": "rekep_constraint",
  "parameters": {
    "keypoints": [
      {"name": "ee_left", "constraint": "align_with", "target": "cup_rim", "axis": "z_up"},
      {"name": "ee_right", "constraint": "maintain_distance", "target": "ee_left", "distance_cm": 10}
    ],
    "objective": "minimize_joint_displacement"
  }
}
```

## 6. Milestones & Acceptance Criteria

### Milestone M1: Driver Scaffold
- [ ] `DobotDriver` exists in `hal/drivers/dobot_driver.py` and imports without error
- [ ] Driver registered as `"dobot_nova5"` in registry
- [ ] `hal/profiles/dobot_nova5.md` exists with DOF, workspace bounds, and joint limits
- [ ] `pytest tests/test_hal_base_driver.py -k dobot_nova5` — all 10 contract tests green (mock mode)

### Milestone M2: ReKep Solver Unit Tests
- [ ] `rekep_solver.py` exists and imports without error
- [ ] `solve(keypoints=[...], joint_limits=[...])` returns a valid joint array (no NaN, no limit violation)
- [ ] Solver unit test: "EE aligned with cup rim, Z-axis up" produces a trajectory where last pose has Z pointing up (within 5°)

### Milestone M3: Hardware Integration (physical Dobot required)
- [ ] Dobot SDK connection established without errors
- [ ] Single-arm `pick_up` grasps a 50g foam cube at a known position
- [ ] `put_down` places the cube within 20 mm of the target location
- [ ] `dual_arm_handover` transfers the cube from left to right arm without dropping

### Milestone M4: Safety Validation
- [ ] Critic rejects `move_to` coordinates outside workspace bounds (EMBODIED.md check)
- [ ] Physical fence: arm stops on contact force > 150N (confirmed with force gauge)
- [ ] Failed action writes a clear entry to `LESSONS.md`

### Milestone M5: Full ReKep Pipeline
- [ ] User types "align the gripper with the cup rim and pick it up" in CLI
- [ ] Planner generates `rekep_constraint` action in `ACTION.md`
- [ ] Solver produces valid trajectory; arm executes; cup is grasped
- [ ] `ENVIRONMENT.md` updated: `cup.location = "held"`

## 7. Dependencies

- Dobot SDK (vendor-provided Python bindings)
- `scipy` or `cvxpy` for quadratic programming (ReKep solver)
- `numpy` for kinematics
- Optional: ROS2 Humble + Dobot ROS2 package

## 8. ReKep Solver Architecture

```
ACTION.md (constraints)
        │
        ▼
  ┌──────────────┐
  │ rekep_solver  │  ← Reads EMBODIED.md for joint limits
  │   .py         │
  └──────┬───────┘
         │ joint_trajectory
         ▼
  ┌──────────────┐
  │ dobot_driver  │  ← Sends joint commands via Dobot SDK
  │   .py         │
  └──────────────┘
```

The solver is a separate module (`rekep_solver.py`) so it can be unit-tested independently of the Dobot hardware.
