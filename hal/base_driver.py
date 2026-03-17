"""
hal/base_driver.py

Abstract base class for all robot body drivers.

Every hardware or simulation embodiment MUST subclass `BaseDriver` and
implement the four abstract methods.  The HAL Watchdog loads a driver by
name and interacts with it exclusively through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDriver(ABC):
    """Contract that every robot body driver must fulfil.

    Methods
    -------
    get_profile_path()
        Return the path to this driver's ``EMBODIED.md`` profile.
    load_scene(scene)
        Initialise the physical / simulated world from a scene dict.
    execute_action(action_type, params)
        Execute one atomic action; return a human-readable result.
    get_scene()
        Return the current world state as a scene dict.
    close()
        Release hardware resources (optional override).
    """

    # ── Required ────────────────────────────────────────────────────────

    @abstractmethod
    def get_profile_path(self) -> Path:
        """Return the filesystem path to this driver's EMBODIED.md profile.

        The watchdog copies this file into the workspace on startup so the
        Planner and Critic agents know what the robot body can do.
        """

    @abstractmethod
    def load_scene(self, scene: dict[str, dict]) -> None:
        """Initialise the world from a scene dict (parsed from ENVIRONMENT.md).

        Called once at startup.  Implementations should set up objects in the
        simulator or calibrate sensors on real hardware.
        """

    @abstractmethod
    def execute_action(self, action_type: str, params: dict) -> str:
        """Execute an atomic action and return a human-readable result string.

        Must **never** raise for unknown action types — return an error
        message string instead so the Planner can adapt.
        """

    @abstractmethod
    def get_scene(self) -> dict[str, dict]:
        """Return the current world state.

        The watchdog writes this dict back to ENVIRONMENT.md after every
        action so the LLM agent sees the updated reality.
        """

    # ── Optional ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release hardware resources.  Override if needed."""

    # ── Context-manager support ─────────────────────────────────────────

    def __enter__(self) -> "BaseDriver":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
