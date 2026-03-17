"""Message bus module for decoupled channel-agent communication."""

from OEA.bus.events import InboundMessage, OutboundMessage
from OEA.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
