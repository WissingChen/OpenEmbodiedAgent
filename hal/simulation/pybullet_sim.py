"""
hal/simulation/pybullet_sim.py

PyBullet-backed physics simulator for Physical Agent Operating System.

Responsibilities
────────────────
- Spawn a PyBullet world (GUI or DIRECT mode).
- Load objects described in the scene dict (from ENVIRONMENT.md).
- Execute high-level robot actions (move_to, pick_up, put_down, push, …).
- Return the post-execution scene state so ENVIRONMENT.md can be updated.

PyBullet is an *optional* dependency.  If it is not installed, the module
raises a clear ImportError with install instructions.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    import pybullet as pb
    import pybullet_data
except ImportError as exc:
    raise ImportError(
        "PyBullet is required for physics simulation.\n"
        "Install it with:  pip install pybullet"
    ) from exc

# Object half-extents (metres) used when spawning simple box colliders
_OBJECT_HALF_EXTENTS: dict[str, tuple[float, float, float]] = {
    "fruit":     (0.03, 0.03, 0.03),
    "container": (0.04, 0.04, 0.06),
    "default":   (0.025, 0.025, 0.025),
}

# Table surface height (metres)
_TABLE_Z = 0.50
# Height of the plane (ground)
_GROUND_Z = 0.0


class PyBulletSimulator:
    """Thin wrapper around PyBullet for PhyAgentOS simulation.

    Parameters
    ----------
    gui:
        If *True* open a 3-D viewer window; otherwise run headlessly
        (DIRECT mode).  Use ``gui=True`` for manual inspection and
        ``gui=False`` (default) for automated tests.
    gravity:
        Gravitational acceleration in m/s² (default 9.81).
    """

    def __init__(self, gui: bool = False, gravity: float = 9.81) -> None:
        self._gui = gui
        self._client = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self._client)
        pb.setGravity(0, 0, -gravity, physicsClientId=self._client)

        # Load ground plane and a simple table
        self._plane_id = pb.loadURDF(
            "plane.urdf", physicsClientId=self._client
        )
        self._table_id = self._spawn_table()

        # Maps object name → PyBullet body ID
        self._body_ids: dict[str, int] = {}

        # Robot "end-effector" position (simplified as a floating point)
        self._ee_pos: list[float] = [0.0, 0.0, _TABLE_Z + 0.20]

        # Whether the robot is currently holding an object
        self._held_object: str | None = None

    # ── Scene management ────────────────────────────────────────────────────

    def load_scene(self, scene: dict[str, dict]) -> None:
        """Spawn all objects described in *scene* into the PyBullet world.

        Existing objects are removed first so the scene can be reloaded.
        """
        self._clear_objects()
        for name, props in scene.items():
            pos = props.get("position", {})
            x = float(pos.get("x", 0)) / 100.0   # cm → m
            y = float(pos.get("y", 0)) / 100.0
            z = _TABLE_Z + float(pos.get("z", 0)) / 100.0
            obj_type = props.get("type", "default")
            half = _OBJECT_HALF_EXTENTS.get(obj_type, _OBJECT_HALF_EXTENTS["default"])
            body_id = self._spawn_box(name, (x, y, z), half)
            self._body_ids[name] = body_id

    def get_scene(self) -> dict[str, dict]:
        """Return current object positions / states as a plain dict.

        This is written back to ENVIRONMENT.md after each action.
        """
        scene: dict[str, dict] = {}
        for name, body_id in self._body_ids.items():
            pos, _ = pb.getBasePositionAndOrientation(
                body_id, physicsClientId=self._client
            )
            # Convert back to centimetres
            x_cm = round(pos[0] * 100, 1)
            y_cm = round(pos[1] * 100, 1)
            z_cm = round((pos[2] - _TABLE_Z) * 100, 1)

            location = "held" if name == self._held_object else (
                "table" if pos[2] >= _TABLE_Z - 0.05 else "floor"
            )
            scene[name] = {
                "position": {"x": x_cm, "y": y_cm, "z": z_cm},
                "location": location,
            }
        return scene

    # ── Actions ─────────────────────────────────────────────────────────────

    def execute_action(self, action_type: str, params: dict) -> str:
        """Dispatch a high-level action and step the simulation.

        Returns a human-readable result string.
        """
        handlers = {
            "move_to":    self._move_to,
            "pick_up":    self._pick_up,
            "put_down":   self._put_down,
            "push":       self._push,
            "point_to":   self._point_to,
            "nod_head":   self._nod_head,
            "shake_head": self._shake_head,
        }
        handler = handlers.get(action_type)
        if handler is None:
            return f"Unknown action type: {action_type!r}"
        return handler(params)

    # ── Low-level action implementations ────────────────────────────────────

    def _move_to(self, params: dict) -> str:
        x = float(params.get("x", 0)) / 100.0
        y = float(params.get("y", 0)) / 100.0
        z = float(params.get("z", 0)) / 100.0 + _TABLE_Z
        self._ee_pos = [x, y, z]
        self._step()
        return f"End-effector moved to ({x*100:.1f}, {y*100:.1f}, {z*100:.1f}) cm."

    def _pick_up(self, params: dict) -> str:
        target = params.get("target", "")
        if target not in self._body_ids:
            return f"Failed: object '{target}' not found in scene."
        if self._held_object is not None:
            return f"Failed: already holding '{self._held_object}'."
        body_id = self._body_ids[target]
        # Move EE to object
        pos, _ = pb.getBasePositionAndOrientation(
            body_id, physicsClientId=self._client
        )
        self._ee_pos = list(pos)
        # Lift it
        lift_pos = (pos[0], pos[1], pos[2] + 0.20)
        pb.resetBasePositionAndOrientation(
            body_id, lift_pos, (0, 0, 0, 1), physicsClientId=self._client
        )
        pb.changeDynamics(
            body_id, -1, mass=0, physicsClientId=self._client
        )  # make static so gravity doesn't drop it
        self._held_object = target
        self._step()
        return f"Picked up '{target}'."

    def _put_down(self, params: dict) -> str:
        target = params.get("target", "")
        location = params.get("location", "table")
        if self._held_object != target:
            return f"Failed: not holding '{target}'."
        body_id = self._body_ids[target]
        # Determine drop position
        if location == "floor":
            drop_z = _GROUND_Z + 0.03
        else:  # table
            drop_z = _TABLE_Z + 0.03
        drop_pos = (self._ee_pos[0], self._ee_pos[1], drop_z)
        pb.resetBasePositionAndOrientation(
            body_id, drop_pos, (0, 0, 0, 1), physicsClientId=self._client
        )
        pb.changeDynamics(
            body_id, -1, mass=0.1, physicsClientId=self._client
        )  # restore mass
        self._held_object = None
        self._step(steps=120)  # let physics settle
        return f"Put down '{target}' at '{location}'."

    def _push(self, params: dict) -> str:
        target = params.get("target", "")
        direction = params.get("direction", "forward")
        if target not in self._body_ids:
            return f"Failed: object '{target}' not found in scene."
        body_id = self._body_ids[target]
        impulse_map = {
            "forward":  (0,  0.5, 0),
            "backward": (0, -0.5, 0),
            "left":     (-0.5, 0, 0),
            "right":    (0.5, 0, 0),
        }
        impulse = impulse_map.get(direction, (0, 0.5, 0))
        pb.applyExternalForce(
            body_id, -1, impulse, (0, 0, 0), pb.WORLD_FRAME,
            physicsClientId=self._client,
        )
        self._step(steps=240)
        return f"Pushed '{target}' {direction}."

    def _point_to(self, params: dict) -> str:
        target = params.get("target", "")
        if target in self._body_ids:
            pos, _ = pb.getBasePositionAndOrientation(
                self._body_ids[target], physicsClientId=self._client
            )
            self._ee_pos = list(pos)
        return f"Pointed to '{target}'."

    def _nod_head(self, _params: dict) -> str:
        if self._gui:
            time.sleep(0.3)
        return "Nodded head."

    def _shake_head(self, _params: dict) -> str:
        if self._gui:
            time.sleep(0.3)
        return "Shook head."

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _spawn_table(self) -> int:
        """Spawn a flat static box to act as a table surface."""
        col = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[0.30, 0.30, 0.01],
            physicsClientId=self._client,
        )
        vis = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=[0.30, 0.30, 0.01],
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
            physicsClientId=self._client,
        )
        return pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, _TABLE_Z],
            physicsClientId=self._client,
        )

    def _spawn_box(
        self,
        name: str,
        position: tuple[float, float, float],
        half_extents: tuple[float, float, float],
    ) -> int:
        """Spawn a coloured box representing an object."""
        import hashlib
        # Deterministic colour from object name
        h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
        r = ((h >> 16) & 0xFF) / 255.0
        g = ((h >> 8)  & 0xFF) / 255.0
        b = (h & 0xFF) / 255.0

        col = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=list(half_extents),
            physicsClientId=self._client,
        )
        vis = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=list(half_extents),
            rgbaColor=[r, g, b, 1.0],
            physicsClientId=self._client,
        )
        return pb.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
            physicsClientId=self._client,
        )

    def _clear_objects(self) -> None:
        """Remove all previously spawned objects from the world."""
        for body_id in self._body_ids.values():
            pb.removeBody(body_id, physicsClientId=self._client)
        self._body_ids.clear()
        self._held_object = None

    def _step(self, steps: int = 60) -> None:
        """Advance the simulation by *steps* timesteps."""
        for _ in range(steps):
            pb.stepSimulation(physicsClientId=self._client)
            if self._gui:
                time.sleep(1.0 / 240.0)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Disconnect from PyBullet and free resources."""
        try:
            pb.disconnect(physicsClientId=self._client)
        except pb.error:
            pass

    def __enter__(self) -> "PyBulletSimulator":
        return self

    def __exit__(self, *_) -> None:
        self.close()
