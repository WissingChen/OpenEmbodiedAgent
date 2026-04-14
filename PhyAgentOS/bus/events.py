"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    channel: str  # telegram, discord, slack, whatsapp, system
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    session_key_override: str | None = None  # Optional override for thread-scoped sessions

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionEvent:
    """Fired by a perception service when a noteworthy environment change occurs.

    The AgentLoop can subscribe to these events and inject them as system
    messages into the active session so the LLM can react proactively.

    Attributes
    ----------
    event_type:
        Short machine-readable label, e.g. ``"target_detected"``,
        ``"target_lost"``, ``"obstacle_detected"``, ``"nav_completed"``.
    robot_id:
        The robot that generated the event.
    description:
        Human-readable description forwarded to the LLM as a system message.
    session_key:
        Optional ``"channel:chat_id"`` string.  When set the event is routed
        to that specific session; otherwise it is broadcast to all active
        sessions.
    data:
        Arbitrary structured payload (e.g. bounding box, confidence score).
    """

    event_type: str
    robot_id: str
    description: str
    session_key: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
