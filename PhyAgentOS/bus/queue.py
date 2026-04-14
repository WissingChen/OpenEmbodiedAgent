"""Async message queue for decoupled channel-agent communication."""

import asyncio

from PhyAgentOS.bus.events import InboundMessage, OutboundMessage, PerceptionEvent


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.

    Perception services can publish ``PerceptionEvent`` objects to the
    ``perception`` queue.  The AgentLoop drains this queue and converts
    events into system ``InboundMessage`` objects so the LLM can react
    proactively to environment changes.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self.perception: asyncio.Queue[PerceptionEvent] = asyncio.Queue()

    # ── inbound (user → agent) ────────────────────────────────────────────────

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    # ── outbound (agent → user) ───────────────────────────────────────────────

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    # ── perception events (sensor → agent) ───────────────────────────────────

    async def publish_perception(self, event: PerceptionEvent) -> None:
        """Publish a perception event from a sensor/perception service."""
        await self.perception.put(event)

    def try_consume_perception(self) -> PerceptionEvent | None:
        """Non-blocking drain of one perception event; returns None if empty."""
        try:
            return self.perception.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def drain_perception(self) -> list[PerceptionEvent]:
        """Drain all pending perception events without blocking."""
        events: list[PerceptionEvent] = []
        while True:
            event = self.try_consume_perception()
            if event is None:
                break
            events.append(event)
        return events

    # ── size helpers ──────────────────────────────────────────────────────────

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()

    @property
    def perception_size(self) -> int:
        """Number of pending perception events."""
        return self.perception.qsize()
