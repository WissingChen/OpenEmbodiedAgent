"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from PhyAgentOS.agent.context import ContextBuilder
from PhyAgentOS.embodiment_registry import EmbodimentRegistry
from PhyAgentOS.agent.memory import MemoryConsolidator
from PhyAgentOS.agent.subagent import SubagentManager
from PhyAgentOS.agent.tools.cron import CronTool
from PhyAgentOS.agent.tools.agent import AgentModeTool
from PhyAgentOS.agent.tools.image import ImageTool
from PhyAgentOS.agent.tools.embodied import EmbodiedActionTool
from PhyAgentOS.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from PhyAgentOS.agent.tools.message import MessageTool
from PhyAgentOS.agent.tools.registry import ToolRegistry
from PhyAgentOS.agent.tools.scene_graph import SceneGraphQueryTool
from PhyAgentOS.agent.tools.shell import ExecTool
from PhyAgentOS.agent.tools.semantic_navigation import SemanticNavigationTool
from PhyAgentOS.agent.tools.spawn import SpawnTool
from PhyAgentOS.agent.tools.web import WebFetchTool, WebSearchTool
from PhyAgentOS.agent.tools.target_navigation import TargetNavigationTool
from PhyAgentOS.agent.tools.task import TaskPlanningTool
from PhyAgentOS.bus.events import InboundMessage, OutboundMessage, PerceptionEvent
from PhyAgentOS.bus.queue import MessageBus
from PhyAgentOS.triggers.buffer import BufferedMessage, TriggerBufferManager
from PhyAgentOS.providers.base import LLMProvider
from PhyAgentOS.providers.providers_manager import ProvidersManager
from PhyAgentOS.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from PhyAgentOS.config.schema import ChannelsConfig, ExecToolConfig, TriggersConfig
    from PhyAgentOS.cron.service import CronService
    from PhyAgentOS.triggers.registry import TriggerRegistry


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        embodiment_registry: EmbodimentRegistry | None = None,
        triggers_config: TriggersConfig | None = None,
        trigger_registry: TriggerRegistry | None = None,
    ):
        from PhyAgentOS.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        # Trigger registry (optional — no-op when None)
        self.trigger_registry = trigger_registry

        # Trigger message buffer manager
        _tc = triggers_config
        self.trigger_buffer_manager = TriggerBufferManager(
            capacity=_tc.buffer_capacity if _tc else 256,
            soft_watermark=_tc.buffer_soft_watermark if _tc else 0.80,
        )

        # Per-session wakeup events
        self._trigger_wakeup: dict[str, asyncio.Event] = {}

        self.context = ContextBuilder(workspace, trigger_registry=trigger_registry)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.embodiment_registry = embodiment_registry
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()
        # Load env variables
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=self.workspace / ".env")
        except Exception:
            logger.warning("Failed to load .env file, ignore using env variables")

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        if isinstance(self.provider, ProvidersManager):
            self.tools.register(AgentModeTool(self.provider))
            self.tools.register(ImageTool(self.provider, send_callback=self.bus.publish_outbound))

        action_tool = EmbodiedActionTool(
            workspace=self.workspace,
            provider=self.provider,
            model=self.model,
            registry=self.embodiment_registry,
        )
        self.tools.register(action_tool)
        self.tools.register(SceneGraphQueryTool(workspace=self.workspace))
        self.tools.register(SemanticNavigationTool(
            workspace=self.workspace,
            action_tool=action_tool,
            registry=self.embodiment_registry,
        ))
        self.tools.register(TargetNavigationTool(
            workspace=self.workspace,
            action_tool=action_tool,
            registry=self.embodiment_registry,
        ))
        self.tools.register(TaskPlanningTool(workspace=self.workspace))

        # 触发器管理工具
        from PhyAgentOS.agent.tools.trigger_mgmt import TriggerManagementTool
        self.tools.register(TriggerManagementTool(registry=self.trigger_registry))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from PhyAgentOS.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron", "trigger"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "message":
                        tool.set_context(channel, chat_id, message_id)
                    elif name == "trigger":
                        tool.set_context(channel, chat_id)
                    else:
                        tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session: Session | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop.

        If *session* is provided, each new message is persisted to the
        JSONL file immediately as it is generated, not batched at the end.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                # 实时持久化：智能体工具调用消息
                if session:
                    self._persist_message(session, messages[-1])

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                            # 检查工具是否产生了静音结果。
                    # 静音工具结果在当前 LLM 轮次中仍然可见
                    # （API 要求工具结果与调用匹配），但会在 _save_turn 中
                    # 被标记为 muted=True，从而在后续轮次中通过
                    # get_history() 的过滤被排除出上下文。
                    _tool = self.tools.get(tool_call.name)
                    _is_muted = getattr(_tool, '_last_muted', False) if _tool else False
                    if _tool and hasattr(_tool, '_last_muted'):
                        _tool._last_muted = False  # 为下次调用重置状态
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    if _is_muted:
                        messages[-1]["_muted"] = True
                    # 实时持久化：工具执行结果
                    if session:
                        self._persist_message(session, messages[-1])
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                # 实时持久化：智能体最终回复
                if session:
                    self._persist_message(session, messages[-1])
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                # On each idle cycle:
                # 1. Drain and inject any pending perception events.
                # 2. Check Trigger buffer wakeup events and inject synthetic
                #    inbound messages so the agent processes buffered trigger
                #    messages exactly like user messages.
                await self._inject_perception_events()
                await self._inject_trigger_wakeups()
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _inject_trigger_wakeups(self) -> None:
        """Check all Trigger buffer wakeup events and inject synthetic inbound
        messages for sessions whose buffers have pending visible messages.

        This allows trigger messages to wake the agent in the same way that
        user messages do, without requiring a separate asyncio task per session.

        The synthetic message uses channel="trigger" so _process_message can
        identify it and flush the buffer into the session before building the
        LLM context.
        """
        for session_key, event in list(self._trigger_wakeup.items()):
            if not event.is_set():
                continue
            # Only inject if no task is already running for this session.
            active = self._active_tasks.get(session_key, [])
            if any(not t.done() for t in active):
                continue
            if":" in session_key:
                channel, chat_id = session_key.split(":", 1)
            else:
                channel, chat_id = "trigger", session_key
            wakeup_msg = InboundMessage(
                channel="trigger",
                sender_id="trigger_buffer",
                chat_id=chat_id,
                content="[Trigger message(s) pending]",
                session_key_override=session_key,
            )
            logger.debug("Trigger wakeup injected for session '{}'", session_key)
            task = asyncio.create_task(self._dispatch(wakeup_msg))
            self._active_tasks.setdefault(session_key, []).append(task)
            task.add_done_callback(
                lambda t, k=session_key: (
                    self._active_tasks.get(k, []) and
                    self._active_tasks[k].remove(t)
                    if t in self._active_tasks.get(k, []) else None
                )
            )

    async def _inject_perception_events(self) -> None:
        """Drain perception events from the bus and inject them as system messages.

        Each :class:`PerceptionEvent` is converted into a ``system``-channel
        :class:`InboundMessage` so that ``_process_message`` can handle it
        with the normal LLM reasoning path.

        The ``chat_id`` of the system message is set to the event's
        ``session_key`` when provided, otherwise it defaults to
        ``"cli:direct"`` so there is always a valid routing target.
        """
        events = self.bus.drain_perception()
        for event in events:
            session_key = event.session_key or "cli:direct"
            # session_key format is "channel:chat_id"
            if ":" in session_key:
                channel, chat_id = session_key.split(":", 1)
            else:
                channel, chat_id = "cli", session_key

            content = (
                f"[Perception Alert — robot_id={event.robot_id}, "
                f"event_type={event.event_type}]\n{event.description}"
            )
            logger.info(
                "Injecting perception event: robot={} type={} session={}",
                event.robot_id, event.event_type, session_key,
            )
            sys_msg = InboundMessage(
                channel="system",
                sender_id=f"perception:{event.robot_id}",
                chat_id=f"{channel}:{chat_id}",
                content=content,
            )
            await self.bus.publish_inbound(sys_msg)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    # 仅当 MessageTool 未在本轮发送过消息时，才发送空响应
                    # 否则空响应会在交互模式中导致异常渲染
                    mt = self.tools.get("message")
                    if not (mt and isinstance(mt, MessageTool) and mt._sent_in_turn):
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
session_key=key,
            )
            # 实时持久化系统消息（用户/系统消息在循环开始前就写入）
            self._persist_message(session, messages[-1])
            final_content, _, all_msgs = await self._run_agent_loop(
                messages, session=session,
            )
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        # Trigger wakeup messages: flush the buffer into the session, then
        # build the LLM context so the agent sees the trigger messages.
        if msg.channel == "trigger":
            key = session_key or msg.session_key
            session = self.sessions.get_or_create(key)
            await self._flush_trigger_buffer(session)
            if":" in key:
                channel, chat_id = key.split(":", 1)
            else:
                channel, chat_id = "trigger", key
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, None)
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message="[Trigger messages received — please review and respond as appropriate.]",
                channel=channel, chat_id=chat_id,
                session_key=key,
            )
            # 实时持久化触发器唤醒消息
            self._persist_message(session, messages[-1])
            final_content, _, all_msgs = await self._run_agent_loop(
                messages, session=session,
            )
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            if final_content:
                return OutboundMessage(channel=channel, chat_id=chat_id,
                                      content=final_content)
            return None

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Flush any pending Trigger buffer messages into the session before
        # processing the user message, so the agent sees them in context.
        await self._flush_trigger_buffer(session)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed, session not cleared. Please try again.",
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/mute":
            # 切换勿扰模式：用户的工具调用结果不激活智能体回复
            muted = session.metadata.get("_user_muted", False)
            session.metadata["_user_muted"] = not muted
            self.sessions.save(session)
            status = "🔇 勿扰模式已开启" if not muted else "🔔 勿扰模式已关闭"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=status)
        if cmd.startswith("/tool "):
            # 用户直接调用工具：/tool <name> {"param": "value"}
            return await self._handle_user_tool_call(msg, session)
        if cmd.startswith("/triggers"):
            return self._handle_triggers_command(msg, key)
        if cmd == "/help":
            lines = [
                "🐈 PhyAgentOS commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/tool <name> <params> — 直接调用工具",
                "/mute — 切换勿扰模式",
                "/triggers — Manage trigger environments",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            session_key=key,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # 实时持久化用户消息
        self._persist_message(session, initial_messages[-1])
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
            session=session,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _persist_message(self, session: Session, msg: dict) -> None:
        """将单条消息实时持久化到 session（内存 + JSONL 文件）。

        在 _run_agent_loop 内部每产生一条新消息后立即调用，
        确保即使中途崩溃也不会丢失已生成的消息。
        """
        from datetime import datetime
        entry = dict(msg)
        role, content = entry.get("role"), entry.get("content")
        # 跳过空 assistant 消息
        if role == "assistant" and not content and not entry.get("tool_calls"):
            return
        # 工具结果：截断 + muted 标记
        if role == "tool":
            if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            if entry.pop("_muted", False):
                entry["muted"] = True
        # 用户消息：剥离 Runtime Context
        elif role == "user":
            if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                ps = content.split("\n\n", 1)
                if len(ps) > 1 and ps[1].strip():
                    entry["content"] = ps[1]
                else:
                    return
            if isinstance(content, list):
                filtered = [c for c in content
                            if not (c.get("type") == "text" and isinstance(c.get("text"), str)
                                    and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG))]
                filtered = [{"type": "text", "text": "[image]"}
                            if c.get("type") == "image_url" and c.get("image_url", {}).get("url", "").startswith("data:image/")
                            else c for c in filtered]
                if not filtered:
                    return
                entry["content"] = filtered
        entry.setdefault("timestamp", datetime.now().isoformat())
        session.messages.append(entry)
        self.sessions.append_message(session, entry)

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                if entry.pop("_muted", False):
                    entry["muted"] = True
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
            # 实时写入：每条消息立即追加到 JSONL 文件
            # 以行为最小写入单元，使用文件锁保证并发安全
            self.sessions.append_message(session, entry)
        session.updated_at = datetime.now()

    async def _flush_trigger_buffer(self, session: Session) -> None:
        """Flush all pending Trigger buffer messages into the session.

        Each buffered message is:
        - Written to the session via ``session.add_message()`` (in-memory).
        - Appended to the session file via ``sessions.append_message()``
          (single-line append, file-locked).

        Dropped messages are included in the file (for user transparency)
        but are excluded from the LLM context by ``session.get_history()``.
        """
        buf = self.trigger_buffer_manager.get(session.key)
        if buf is None:
            return
        pending = await buf.flush()
        if not pending:
            return
        for bm in pending:
            entry = session.add_message(
                role=bm.role,
                content=bm.content,
                priority=bm.priority,
                muted=bm.muted,
                dropped=bm.dropped,
                relates_to=bm.relates_to,
                sender_id=bm.sender_id,
                **(bm.extra or {}),
            )
            self.sessions.append_message(session, entry)
        logger.debug(
            "Flushed {} trigger buffer message(s) into session '{}'",
            len(pending), session.key,
        )

    def get_or_create_trigger_buffer_for_session(self, session_key: str):
        """Return (creating if needed) the TriggerBuffer for *session_key*.

        Also registers a wakeup event so the run() loop can detect pending
        trigger messages.  Call this from trigger code orEnvironmentSession
        instances that need to enqueue messages for a session.
        """
        event = self._trigger_wakeup.setdefault(session_key, asyncio.Event())
        return self.trigger_buffer_manager.get_or_create(session_key, wakeup_event=event)

    async def _handle_user_tool_call(
        self, msg: InboundMessage, session: Session,
    ) -> OutboundMessage | None:
        """处理用户直接调用工具：/tool <name> <json_params>

        用户的工具调用与智能体的工具调用使用相同的 role name 写入 session，
        实现"用户可以做智能体能做的所有事"。
        如果处于勿扰模式（/mute），结果标记为 muted 不激活智能体回复。
        """
        parts = msg.content.strip().split(None, 2)  # /tool name {params}
        if len(parts) < 2:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="用法: /tool <name> [json_params]\n"
                        f"可用工具: {', '.join(self.tools.tool_names)}",
            )
        tool_name = parts[1]
        params_str = parts[2] if len(parts) > 2 else "{}"

        # 解析参数
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError:
            # 尝试简单的 key=value 格式
            params = {"command": params_str} if tool_name == "exec" else {}

        # 先持久化用户的工具调用命令
        self._persist_message(session, {
            "role": "user",
            "content": msg.content,
        })

        # 执行工具
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        result = await self.tools.execute(tool_name, params)

        # 检查勿扰模式
        is_muted = session.metadata.get("_user_muted", False)

        # 以 tool role 写入结果（与智能体调用一致）
        tool_entry = {
            "role": "tool",
            "name": tool_name,
            "content": result,
        }
        if is_muted:
            tool_entry["muted"] = True
        self._persist_message(session, tool_entry)

        # 返回结果给用户
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=result,
            metadata={"_tool_result": True, "_muted": is_muted},
        )

    def _handle_triggers_command(
        self, msg: InboundMessage, session_key: str,
    ) -> OutboundMessage:
        """Handle /triggers slash commands.

        Supported commands:
            /triggers              — list all triggers (brief)
            /triggers list         — same
            /triggers list detail— with descriptions and permissions
            /triggers list active  — filter by state (active/muted/inactive)
            /triggers status <n>   — single trigger details
            /triggers set <n> <s>  — change trigger state
            /triggers envs         — list registered environments
        """
        from PhyAgentOS.triggers.trigger import TriggerState
        # 解析命令: /triggers [子命令] [参数...]
        # 例如 "/triggers list detail active" 或 "/triggers set temp_alert muted"
        parts = msg.content.strip().split()
        sub = parts[1] if len(parts) > 1 else "list"  # 默认子命令

        # /triggers envs — list registered environments
        if sub == "envs":
            if not self.trigger_registry:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="No trigger registry configured.",
                )
            envs = self.trigger_registry.list_available()
            active = self.trigger_registry.list_active_sessions()
            lines = ["**Registered Trigger Environments:**"]
            for name in envs:
                status = "🟢 online" if any(
                    self.trigger_registry.get_instance(k)
                    and self.trigger_registry.get_instance(k).name == name
                    for k in active
                ) else "⭕ offline"
                lines.append(f"- {name} ({status})")
            if not envs:
                lines.append("(none registered)")
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="\n".join(lines),
            )

        # /triggers status <name> — single trigger details
        if sub == "status" and len(parts) > 2:
            trigger_name = parts[2]
            if not self.trigger_registry:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="No trigger registry configured.",
                )
            instance = self.trigger_registry.get_instance(session_key)
            if not instance:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="No active environment for this session.",
                )
            trigger = instance.get_trigger(trigger_name)
            if not trigger:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Trigger '{trigger_name}' not found.",
                )
            lines = [
                f"**Trigger: {trigger.name}** (id={trigger.trigger_id})",
                f"State: {trigger.state.value}",
                f"Description: {trigger.description or '(none)'}",
                f"Watched: {trigger.watched_observations or '(all)'}",
                f"Actions: {trigger.allowed_actions or '(all)'}",
                f"Agent visible: {trigger.is_agent_visible}",
                f"Agent modifiable: {trigger.is_agent_modifiable}",
                f"Agent startable: {trigger.is_agent_startable}",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="\n".join(lines),
            )

        # /triggers set <name> <state> — change trigger state
        if sub == "set" and len(parts) > 3:
            trigger_name = parts[2]
            state_str = parts[3].lower()
            state_map = {
                "active": TriggerState.ACTIVE,
                "muted": TriggerState.MUTED,
                "inactive": TriggerState.INACTIVE,
            }
            if state_str not in state_map:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Invalid state '{state_str}'. Use: active, muted, inactive",
                )
            if not self.trigger_registry:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="No trigger registry configured.",
                )
            instance = self.trigger_registry.get_instance(session_key)
            if not instance:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="No active environment for this session.",
                )
            if instance.set_trigger_state(trigger_name, state_map[state_str]):
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Trigger '{trigger_name}' → {state_str}",
                )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Trigger '{trigger_name}' not found.",
            )

        # /triggers [list] [detail] [active|muted|inactive]
        detail = "detail" in parts
        state_filter = None
        for p in parts:
            if p in ("active", "muted", "inactive"):
                state_filter = TriggerState(p)
                break

        if not self.trigger_registry:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="No trigger registry configured.",
            )
        # 查找当前会话的活跃环境实例以列出触发器
        instance = self.trigger_registry.get_instance(session_key)
        if not instance:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="No active environment for this session.\n"
                        f"Available environments: {', '.join(self.trigger_registry.list_available()) or '(none)'}",
            )
        triggers = instance.list_triggers(state_filter=state_filter)
        if not triggers:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="No triggers match the filter.",
            )
        state_icons = {"active": "🟢", "muted": "🔇", "inactive": "⭕"}
        lines = [f"**Triggers in '{instance.name}'** ({len(triggers)} total):"]
        for t in triggers:
            icon = state_icons.get(t["state"], "❓")
            line = f"{icon} **{t['name']}** [{t['state']}]"
            if detail:
                line += f" — {t.get('description') or '(no description)'}"
                perms = []
                if not t.get("is_agent_visible"):
                    perms.append("hidden")
                if not t.get("is_agent_modifiable"):
                    perms.append("fixed")
                if not t.get("is_agent_startable"):
                    perms.append("disabled")
                if perms:
                    line += f" ({', '.join(perms)})"
            lines.append(f"- {line}")
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="\n".join(lines),
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()

        # 解析 session_key，确保工具上下文绑定到正确的会话
        if session_key != "cli:direct":
            if ":" in session_key:
                channel, chat_id = session_key.split(":", 1)
            else:
                chat_id = session_key

        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)

        # 在 direct 模式下，CLI 没有运行总线消费者任务，
        # 我们需要手动排空总线中由工具（如 MessageTool）发送的消息。
        outbound_texts = []
        choices_presented = None

        while not self.bus.outbound.empty():
            out_msg = self.bus.outbound.get_nowait()
            if out_msg.content:
                outbound_texts.append(out_msg.content)
            if getattr(out_msg, "choices", None):
                choices_presented = out_msg.choices

        if response and response.content:
            outbound_texts.append(response.content)

        combined = "\n\n".join(outbound_texts)

        # 如果智能体发起了选择题，在 direct 模式下我们阻塞等待用户输入
        if choices_presented:
            print(f"\n{combined}\n")
            for c in choices_presented:
                print(f"- {c.get('id')}: {c.get('label')}")
            try:
                import asyncio
                user_choice = await asyncio.to_thread(input, "\n[Choice] 请输入选项: ")
                # 递归处理用户的选择
                return await self.process_direct(
                    user_choice, session_key, channel, chat_id, on_progress
                )
            except (KeyboardInterrupt, EOFError):
                pass

        return combined
