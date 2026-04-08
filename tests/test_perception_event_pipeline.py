"""Tests for the perception event pipeline:
  EnvironmentWriter.write(events=...) → MessageBus.perception → AgentLoop._inject_perception_events
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hal.perception.environment_writer import EnvironmentWriter, PerceptionEventSpec
from hal.simulation.scene_io import save_environment_doc
from PhyAgentOS.bus.events import PerceptionEvent
from PhyAgentOS.bus.queue import MessageBus


# ── helpers ───────────────────────────────────────────────────────────────────

def _seed_env(path: Path) -> None:
    save_environment_doc(path, {
        "schema_version": "PhyAgentOS.environment.v1",
        "scene_graph": {"nodes": [], "edges": []},
        "robots": {},
        "objects": {},
    })


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── EnvironmentWriter with bus ────────────────────────────────────────────────

def test_environment_writer_publishes_perception_event(tmp_path: Path) -> None:
    """EnvironmentWriter.write() with events= should put a PerceptionEvent on the bus."""
    env_file = tmp_path / "ENVIRONMENT.md"
    _seed_env(env_file)

    bus = MessageBus()
    writer = EnvironmentWriter(tmp_path, bus=bus)

    async def _go():
        writer.write(
            robot_id="go2_edu_001",
            events=[
                PerceptionEventSpec(
                    event_type="target_detected",
                    description="Apple detected at (0.5, 0.3, 0.8)",
                    session_key="cli:direct",
                    data={"confidence": 0.95},
                )
            ],
        )
        # Give the event loop a tick to process the create_task
        await asyncio.sleep(0)

    _run(_go())

    assert bus.perception_size == 1
    event = bus.try_consume_perception()
    assert event is not None
    assert event.event_type == "target_detected"
    assert event.robot_id == "go2_edu_001"
    assert "Apple" in event.description
    assert event.session_key == "cli:direct"
    assert event.data["confidence"] == 0.95


def test_environment_writer_without_bus_does_not_raise(tmp_path: Path) -> None:
    """EnvironmentWriter without a bus should silently ignore events."""
    env_file = tmp_path / "ENVIRONMENT.md"
    _seed_env(env_file)

    writer = EnvironmentWriter(tmp_path, bus=None)
    # Should not raise even when events are provided
    writer.write(
        robot_id="go2_edu_001",
        events=[PerceptionEventSpec("target_detected", "Apple found")],
    )


def test_environment_writer_no_events_no_publish(tmp_path: Path) -> None:
    """When events=None, nothing is published to the bus."""
    env_file = tmp_path / "ENVIRONMENT.md"
    _seed_env(env_file)

    bus = MessageBus()
    writer = EnvironmentWriter(tmp_path, bus=bus)
    writer.write(robot_id="go2_edu_001")

    assert bus.perception_size == 0


# ── MessageBus perception queue ───────────────────────────────────────────────

def test_message_bus_drain_perception_returns_all_events() -> None:
    bus = MessageBus()

    async def _fill():
        await bus.publish_perception(PerceptionEvent("a", "r1", "desc1"))
        await bus.publish_perception(PerceptionEvent("b", "r1", "desc2"))
        await bus.publish_perception(PerceptionEvent("c", "r2", "desc3"))

    _run(_fill())

    events = bus.drain_perception()
    assert len(events) == 3
    assert [e.event_type for e in events] == ["a", "b", "c"]
    assert bus.perception_size == 0


def test_message_bus_drain_perception_empty_returns_empty_list() -> None:
    bus = MessageBus()
    assert bus.drain_perception() == []


def test_message_bus_try_consume_perception_returns_none_when_empty() -> None:
    bus = MessageBus()
    assert bus.try_consume_perception() is None


# ── AgentLoop._inject_perception_events ──────────────────────────────────────

def test_agent_loop_inject_perception_events_publishes_inbound() -> None:
    """_inject_perception_events should convert PerceptionEvents to InboundMessages."""
    bus = MessageBus()

    async def _go():
        # Pre-load two perception events
        await bus.publish_perception(PerceptionEvent(
            event_type="target_detected",
            robot_id="go2_edu_001",
            description="Apple detected",
            session_key="telegram:12345",
        ))
        await bus.publish_perception(PerceptionEvent(
            event_type="obstacle_detected",
            robot_id="go2_edu_001",
            description="Obstacle ahead",
            session_key=None,  # broadcast → cli:direct
        ))

        # Build a minimal AgentLoop-like object with just the method we need
        from PhyAgentOS.agent.loop import AgentLoop
        loop_obj = object.__new__(AgentLoop)
        loop_obj.bus = bus

        await loop_obj._inject_perception_events()

    _run(_go())

    # Both events should now be in the inbound queue as system messages
    assert bus.inbound_size == 2

    msg1 = bus.inbound.get_nowait()
    assert msg1.channel == "system"
    assert msg1.chat_id == "telegram:12345"
    assert "target_detected" in msg1.content
    assert "Apple detected" in msg1.content

    msg2 = bus.inbound.get_nowait()
    assert msg2.channel == "system"
    assert msg2.chat_id == "cli:direct"
    assert "obstacle_detected" in msg2.content
