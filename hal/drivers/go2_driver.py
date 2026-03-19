"""Mock-friendly Go2 navigation driver."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from hal.base_driver import BaseDriver
from hal.ros2 import ROS2Bridge

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


class Go2Driver(BaseDriver):
    """Navigation-oriented driver with a dependency-free mock runtime."""

    def __init__(self, gui: bool = False, bridge: ROS2Bridge | None = None, **_kwargs: Any):
        self._gui = gui
        self._bridge = bridge or ROS2Bridge(enabled=False)
        self._objects: dict[str, dict] = {}
        self._runtime_state = {
            "robots": {
                "go2_edu_001": {
                    "robot_pose": {
                        "frame": "map",
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "yaw": 0.0,
                        "stamp": self._stamp(),
                    },
                    "nav_state": {"mode": "idle", "status": "idle", "recovery_count": 0},
                }
            }
        }

    def get_profile_path(self) -> Path:
        return _PROFILES_DIR / "go2_edu.md"

    def load_scene(self, scene: dict[str, dict]) -> None:
        self._objects = dict(scene)

    def execute_action(self, action_type: str, params: dict) -> str:
        if action_type == "semantic_navigate":
            return self._semantic_navigate(params)
        if action_type == "localize":
            return self._localize(params)
        if action_type == "stop":
            state = self._robot_state(params.get("robot_id"))
            state["nav_state"] = {"mode": "idle", "status": "stopped", "recovery_count": 0}
            return "Navigation stopped."
        return f"Unknown action: {action_type}"

    def get_scene(self) -> dict[str, dict]:
        return dict(self._objects)

    def get_runtime_state(self) -> dict[str, Any]:
        return self._runtime_state

    def _semantic_navigate(self, params: dict[str, Any]) -> str:
        robot_id = params.get("robot_id", "go2_edu_001")
        goal_pose = params.get("goal_pose") or {}
        target_ref = params.get("target_ref") or {}
        if "x" not in goal_pose or "y" not in goal_pose:
            state = self._robot_state(robot_id)
            state["nav_state"] = {
                "mode": "navigating",
                "status": "failed",
                "last_error": "target_not_found",
                "recovery_count": 0,
            }
            return "Navigation failed: goal pose missing."

        state = self._robot_state(robot_id)
        dx = float(goal_pose["x"]) - float(state["robot_pose"]["x"])
        dy = float(goal_pose["y"]) - float(state["robot_pose"]["y"])
        yaw = float(goal_pose.get("yaw", math.atan2(dy, dx) if dx or dy else 0.0))
        state["robot_pose"] = {
            "frame": goal_pose.get("frame", "map"),
            "x": float(goal_pose["x"]),
            "y": float(goal_pose["y"]),
            "z": float(goal_pose.get("z", 0.0)),
            "yaw": yaw,
            "stamp": self._stamp(),
        }
        state["nav_state"] = {
            "mode": "navigating",
            "status": "arrived",
            "goal_id": target_ref.get("id", "semantic_goal"),
            "target_ref": target_ref,
            "goal": {
                "x": float(goal_pose["x"]),
                "y": float(goal_pose["y"]),
                "yaw": yaw,
            },
            "path_progress": 1.0,
            "recovery_count": 0,
            "last_error": None,
        }
        self._bridge.publish("/navigate_to_pose", goal_pose)
        return f"Navigation success: arrived near {target_ref.get('label', 'target')}."

    def _localize(self, params: dict[str, Any]) -> str:
        robot_id = params.get("robot_id", "go2_edu_001")
        state = self._robot_state(robot_id)
        nav_state = dict(state.get("nav_state", {}))
        nav_state.update(
            {
                "mode": "localizing",
                "status": "localized",
                "last_error": None,
                "relocalization_confidence": 0.82,
                "recovery_count": nav_state.get("recovery_count", 0),
            }
        )
        state["nav_state"] = nav_state
        state["robot_pose"]["stamp"] = self._stamp()
        return f"Localization success for {robot_id}."

    def _robot_state(self, robot_id: str | None) -> dict[str, Any]:
        robot_id = robot_id or "go2_edu_001"
        robots = self._runtime_state.setdefault("robots", {})
        if robot_id not in robots:
            robots[robot_id] = {
                "robot_pose": {
                    "frame": "map",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "yaw": 0.0,
                    "stamp": self._stamp(),
                },
                "nav_state": {"mode": "idle", "status": "idle", "recovery_count": 0},
            }
        return robots[robot_id]

    @staticmethod
    def _stamp() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
