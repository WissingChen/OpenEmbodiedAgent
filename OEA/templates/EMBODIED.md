# Robot Embodiment Declaration

This file describes the physical capabilities and constraints of the connected robot.
The Critic Agent reads this file to validate whether proposed actions are safe and feasible.

## Identity

- **Name**: OEA Desktop Pet
- **Type**: Desktop-level virtual pet (simulation mode)

## Degrees of Freedom

| Joint | Range | Description |
|-------|-------|-------------|
| Head Pan | -90° to +90° | Horizontal head rotation |
| Head Tilt | -45° to +45° | Vertical head rotation |

## Supported Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `nod_head` | — | Nod head up and down |
| `shake_head` | — | Shake head left and right |
| `point_to` | `target: string` | Point toward a named object in the scene |
| `move_to` | `x, y, z: float` | Move to a 3D coordinate |
| `pick_up` | `target: string` | Pick up a named object |
| `put_down` | `target: string, location: string` | Put an object down at a location |
| `push` | `target: string, direction: string` | Push an object in a direction |

## Physical Constraints

- **Workspace bounds**: x ∈ [-50, 50], y ∈ [-50, 50], z ∈ [0, 30] (centimetres)
- **Max payload**: 50 g
- **Max reach**: 15 cm from base
- **Collision policy**: Stop immediately on contact force > 2 N
