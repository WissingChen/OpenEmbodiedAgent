"""极简 Trigger 集成测试。

测试流程：
1. 创建一个计数器环境（每2秒tick一次，count自增）
2. 注册一个阈值触发器（count >= 3时发消息）
3. 启动环境，验证触发器正确触发并写入session JSONL

运行方式：
    cd PhyAgentOS && python -m pytest tests/test_trigger_simple.py -v
    或直接运行：
    cd PhyAgentOS && python tests/test_trigger_simple.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 确保能 import PhyAgentOS
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PhyAgentOS.triggers.base import TriggerEnvironment
from PhyAgentOS.triggers.trigger import BaseTrigger, TriggerContext, TriggerState
from PhyAgentOS.triggers.buffer import TriggerBuffer, TriggerBufferManager
from PhyAgentOS.triggers.registry import TriggerRegistry, EnvironmentSession


# ---------------------------------------------------------------------------
# 极简计数器环境
# ---------------------------------------------------------------------------

class CounterEnv(TriggerEnvironment):
    """极简测试环境：一个自增计数器。

    观测空间: count（整数）
    动作空间: reset（重置为0）
    时钟周期: 2秒
    """
    name = "counter"
    description = "极简计数器测试环境"
    tick_interval = 2.0

    def __init__(self, **kwargs) -> None:
        self._count = 0
        self._running = False

    def get_observation_space(self):
        return {"count": {"type": "int", "description": "自增计数器"}}

    def get_action_space(self):
        return {"reset": {"params": {}, "description": "重置计数器为0"}}

    def get_global_observation(self):
        return {"env_name": "counter", "version": "1.0"}

    def get_current_observation(self):
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "stale": False,
            "source": "counter",
            "payload": {"count": self._count},
        }

    def start(self) -> None:
        self._running = True
        print(f"[CounterEnv] 环境启动，tick_interval={self.tick_interval}s")

    def stop(self) -> None:
        self._running = False
        print("[CounterEnv] 环境停止")

    def execute_action(self, action_type, params):
        if action_type == "reset":
            old = self._count
            self._count = 0
            return {"status": "succeeded", "result_text": f"计数器从 {old} 重置为 0"}
        return {"status": "failed", "error_code": "INVALID_PARAMS"}

    def tick(self) -> None:
        """手动推进一个 tick（测试用）。"""
        self._count += 1


# ---------------------------------------------------------------------------
# 极简阈值触发器
# ---------------------------------------------------------------------------

class ThresholdTrigger(BaseTrigger):
    """当 count >= threshold 时发送告警消息。"""
    name = "threshold_alert"
    description = "计数器达到阈值时告警"
    watched_observations = ["count"]
    allowed_actions = ["reset"]

    def __init__(self, threshold: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self._threshold = threshold
        self._alerted = False

    async def on_tick(self, ctx: TriggerContext) -> None:
        obs = ctx.get_current_observation()
        count = obs.get("payload", {}).get("count", 0)
        if count >= self._threshold and not self._alerted:
            await ctx.emit_message(
                f"⚠️ 计数器达到阈值！count={count} >= {self._threshold}",
                priority="high",
            )
            self._alerted = True
            print(f"[ThresholdTrigger] 触发告警: count={count}")
        elif count < self._threshold:
            self._alerted = False


# ---------------------------------------------------------------------------
# 测试主函数
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("Trigger 集成测试")
    print("=" * 60)

    # 1. 创建注册表和缓冲区
    registry = TriggerRegistry()
    buffer_mgr = TriggerBufferManager(capacity=64)

    # 2. 注册环境
    registry.register(CounterEnv)
    print(f"\n可用环境: {registry.list_available()}")

    # 3. 实例化环境（绑定到测试session）
    session_key = "test:trigger_test"
    env_session = registry.instantiate("counter", session_key=session_key)

    # 4. 创建缓冲区和触发器
    wakeup = asyncio.Event()
    buf = buffer_mgr.get_or_create(session_key, wakeup_event=wakeup)

    trigger = ThresholdTrigger(threshold=3)
    env_session.add_trigger(trigger, buffer=buf)
    print(f"触发器已添加: {env_session.list_triggers()}")

    # 5. 启动环境
    env_session.start()

    # 6. 设置触发器状态为 active
    env_session.set_trigger_state("threshold_alert", TriggerState.ACTIVE)

    # 7. 手动模拟 tick 循环（实际中由 EnvironmentSession 的后台任务负责）
    env = env_session.environment
    print("\n开始模拟 tick 循环...")

    for i in range(5):
        env.tick()  # 推进计数器
        obs = env.get_current_observation()
        print(f"\nTick {i+1}: count={obs['payload']['count']}")

        # 调用所有 active 触发器的 on_tick
        for tid, (trig, ctx) in env_session._triggers.items():
            if trig.state == TriggerState.ACTIVE and ctx:
                await trig.on_tick(ctx)

        # 检查缓冲区
        if wakeup.is_set():
            messages = await buf.flush()
            for m in messages:
                print(f"  📨 缓冲区消息: [{m.role}] {m.content[:80]} (priority={m.priority}, muted={m.muted})")
            print(f"  缓冲区已清空，共 {len(messages)} 条消息")

        await asyncio.sleep(0.1)  # 短暂等待（测试中不需要等整个 tick_interval）

    # 8. 测试上下文块生成
    print("\n上下文块:")
    print(env_session.get_context_block()[:500])

    # 9. 测试动作执行（通过触发器的 enqueue_action）
    print("\n--- 测试动作执行 ---")
    env.tick()  # count = 6
    # 手动通过 TriggerContext 执行动作
    for tid, (trig, ctx) in env_session._triggers.items():
        if trig.name == "threshold_alert" and ctx:
            action_id = ctx.enqueue_action("reset")
            print(f"  执行动作 'reset', action_id={action_id}")
            obs = env.get_current_observation()
            print(f"  重置后 count={obs['payload']['count']}")
            assert obs["payload"]["count"] == 0, "reset 动作应将计数器归零"
            print("  ✅ 动作执行成功")
            break

    # 10. 测试 TriggerManagementTool（智能体的CRUD接口）
    print("\n--- 测试 TriggerManagementTool ---")
    from PhyAgentOS.agent.tools.trigger_mgmt import TriggerManagementTool
    mgmt = TriggerManagementTool(registry=registry)
    mgmt.set_context("test", "trigger_test")  # 设置 session context

    # 10a. list
    result = await mgmt.execute(action="list")
    print(f"  list: {result}")
    assert "threshold_alert" in result, "list 应包含已注册的触发器"

    # 10b. status
    result = await mgmt.execute(action="status", name="threshold_alert")
    print(f"  status: {result[:100]}...")
    assert "threshold_alert" in result, "status 应返回触发器信息"

    # 10c. create（创建新触发器）
    result = await mgmt.execute(
        action="create",
        name="test_trigger",
        description="测试用触发器",
        watched_observations=["count"],
        allowed_actions=["reset"],
    )
    print(f"  create: {result}")
    assert "成功" in result, "create 应返回成功"

    # 10d. list（验证新触发器出现）
    result = await mgmt.execute(action="list", detail=True)
    print(f"  list detail: {result}")
    assert "test_trigger" in result, "新创建的触发器应出现在列表中"

    # 10e. modify（修改描述）
    result = await mgmt.execute(
        action="modify",
        name="test_trigger",
        description="修改后的描述",
    )
    print(f"  modify: {result}")
    assert "已修改" in result, "modify 应返回已修改"

    # 10f. rename（重命名）
    result = await mgmt.execute(
        action="rename",
        name="test_trigger",
        new_name="renamed_trigger",
    )
    print(f"  rename: {result}")
    assert "已重命名" in result, "rename 应返回已重命名"

    # 10g. set_state（修改状态）
    result = await mgmt.execute(
        action="set_state",
        name="renamed_trigger",
        state="muted",
    )
    print(f"  set_state: {result}")
    assert "muted" in result, "set_state 应返回新状态"

    # 10h. delete（删除触发器）
    result = await mgmt.execute(
        action="delete",
        name="renamed_trigger",
    )
    print(f"  delete: {result}")
    assert "已删除" in result, "delete 应返回已删除"

    # 10i. 验证删除后 list 不再包含该触发器
    result = await mgmt.execute(action="list")
    assert "renamed_trigger" not in result, "删除后触发器不应出现在列表中"
    print("  ✅ CRUD 全部测试通过")

    # 11. 测试权限限制（创建越界的触发器）
    print("\n--- 测试安全校验 ---")
    result = await mgmt.execute(
        action="create",
        name="bad_trigger",
        watched_observations=["nonexistent_key"],
        allowed_actions=["nonexistent_action"],
    )
    print(f"  越界创建: {result}")
    assert "Error" in result or "失败" in result, "越界的触发器应被拒绝"
    print("  ✅ 安全校验通过")

    # 12. 停止环境
    env_session.stop()
    registry.remove_instance(session_key)

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
