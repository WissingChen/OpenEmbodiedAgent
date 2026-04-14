# Robot Embodiment Declaration — Simulation Arm

> Profile: simulation | Driver: SimulationDriver

## Identity

- **Name**: PyBullet Simulated Arm
- **Type**: 7-DOF robotic arm with 2-DOF gripper (simulation)
- **Engine**: PyBullet physics simulator

## Degrees of Freedom

| Joint | Range | Description |
|-------|-------|-------------|
| Arm joints 1-7 | Varies per URDF | 7-DOF arm kinematics |
| Gripper finger L | 0 – 0.04 m | Left gripper finger |
| Gripper finger R | 0 – 0.04 m | Right gripper finger |

## Supported Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `move_to` | `x, y, z` (cm) | Move end-effector to 3D coordinate via IK |
| `pick_up` | `target: string` | Approach named object, close gripper, lift |
| `put_down` | `target: string, location: string` | Move to location, open gripper, release |
| `push` | `target: string, direction: string` | Apply lateral force to object |
| `point_to` | `target: string` | Orient end-effector toward named object |
| `nod_head` | — | Simulate head nod gesture |
| `shake_head` | — | Simulate head shake gesture |

## Physical Constraints

- **Workspace bounds**: x ∈ [-50, 50], y ∈ [-50, 50], z ∈ [0, 30] (centimetres)
- **Max payload**: 2 kg
- **Max reach**: 85 cm from base
- **Collision policy**: Physics engine handles collisions; objects obey gravity
- **Simulation timestep**: 1/240 s
