# PROJ.md vs 当前框架设计 — 差距分析

> 2026-03-15 逐条对照 `docs/PROJ.md` 白皮书与已产出的 Track 文档

## 一、愿景与目标 (PROJ §一)

| PROJ.md 要求 | 当前框架覆盖 | 差距 |
|-------------|------------|------|
| 基于约束求解范式 | ✅ Track 3 (ReKep solver) | — |
| 消费级，不具备专业知识的用户 | ✅ Track 2 (桌面宠物 + SOUL.md) | — |
| 自然语言与机器人交互 | ✅ OEA CLI + chat channels | — |
| Anti-Shitstorm (防失控) | ✅ Critic Agent in embodied.py | — |
| Multi-Agent 校验层 | ✅ Track 5 (Orchestrator) + Critic | — |
| 视觉下沉为文本化 Scene-Graph | ⚠️ **未覆盖**: 没有 Vision MCP Server 的 Track 文档 | **GAP-1** |

### GAP-1: 缺少 Vision MCP Server Track
PROJ.md §四.2.1 明确要求"感知提取 MCP: 定时拍摄画面，通过小规模 VLM 提取 Scene-Graph 覆写 ENVIRONMENT.md"。当前没有专门的 Track 来规划这个视觉感知服务。

**建议**: 新增 `docs/track-8-vision-mcp.md`，负责：
- MCP Vision Server 的开发
- VLM 模型选型 (如 LLaVA, Qwen-VL)
- 摄像头接入与定时拍摄
- Scene-Graph 文本格式标准化
- 覆写 `ENVIRONMENT.md` 的协议

---

## 二、用户画像与场景 (PROJ §二)

| PROJ.md 要求 | 当前框架覆盖 | 差距 |
|-------------|------------|------|
| Tier 1: 极客/开发者 (Franka/Xlerobot/Go2) | ⚠️ Franka 和 Xlerobot 无对应 Driver Track | **GAP-2** |
| Tier 2: 科技爱好者 (教育级机械臂) | ✅ Track 1 (仿真) + Track 3 (Dobot) | — |
| Tier 3: 普通消费者 (桌面宠物) | ✅ Track 2 (Desktop Pet) | — |
| 场景A: 主动服务 (Heartbeat) | ✅ HEARTBEAT.md + OEA heartbeat service | — |
| 场景B: 多机协作 | ✅ Track 5 (Orchestrator + TASK.md) | — |
| 场景C: 技能自主学习与自省 | ⚠️ **部分覆盖**: LESSONS.md 存在，但缺少 Refactor Agent 和 SKILL.md 自主优化机制 | **GAP-3** |

### GAP-2: 缺少 Franka 和 Xlerobot (轮式双臂) Driver
PROJ.md §三.2 列出了 5 种本体: OEA, Go2, Xlerobot, Franka, Elfin。当前 Track 文档只覆盖了 OEA (Track 2), Go2 (Track 4), 仿真 (Track 1), 和 Dobot (Track 3)。

**缺失**: Franka Research 3, Xlerobot (轮式双臂), Elfin E05L-Pro

**建议**: 
- Franka 和 Elfin 可合并为一个 Track（都是高精度工业臂，接口相似）
- Xlerobot（轮式双臂 + 全向轮）需要单独 Track（涉及移动底盘 + 双臂协同）
- 或者在 Track 3 中扩展为通用"多臂操作"Track

### GAP-3: 缺少 Refactor Agent 和 SKILL.md 自主优化
PROJ.md §四.2.4 提到"记忆重构者 (Refactor Agent): 夜间触发，清理 SKILL.md 里的冗余步骤"。当前框架没有此功能。

**建议**: 在 Track 6 (时空记忆) 中增加 Refactor Agent 的需求，或作为 Track 5 (Orchestrator) 的扩展功能。

---

## 三、研发组织与硬件基座 (PROJ §三)

| PROJ.md 要求 | 当前框架覆盖 | 差距 |
|-------------|------------|------|
| Track A/B 双轨分工 | ✅ 完全对齐 | — |
| OEA 桌面宠 | ✅ Track 2 | — |
| Unitree Go2 EDU | ✅ Track 4 | — |
| Xlerobot (轮式双臂) | ❌ **未覆盖** | **GAP-2** |
| Franka Research 3 | ❌ **未覆盖** | **GAP-2** |
| Elfin E05L-Pro | ❌ **未覆盖** | **GAP-2** |

---

## 四、核心系统架构 (PROJ §四)

| PROJ.md 要求 | 当前框架覆盖 | 差距 |
|-------------|------------|------|
| EMBODIED.md (本体声明) | ✅ Profile 机制 | — |
| ENVIRONMENT.md (环境感知) | ✅ scene_io.py + 模板 | — |
| LESSONS.md (经验避坑库) | ✅ Critic 写入 | — |
| SKILL.md (工作流记忆) | ⚠️ 模板存在于 OEA/skills/，但没有本体无关的 SOP 格式定义 | **GAP-4** |
| ACTION.md (软硬耦合层) | ✅ EmbodiedActionTool + watchdog | — |
| SOUL.md & HEARTBEAT.md | ✅ OEA 自带 | — |
| 感知提取 MCP | ❌ **未覆盖** | **GAP-1** |
| Planner Agent | ✅ OEA 主循环 | — |
| Critic Agent | ✅ embodied.py | — |
| Refactor Agent | ❌ **未覆盖** | **GAP-3** |
| ROS2 统一数据总线 | ⚠️ Track 4 (Go2) 提到 ROS2，但没有统一的 ROS2 桥接层设计 | **GAP-5** |
| ReKep 约束求解引擎 | ✅ Track 3 (rekep_solver.py) | — |
| 动态电子围栏 — 逻辑围栏 | ✅ Critic 对照 EMBODIED.md 校验 | — |
| 动态电子围栏 — 物理围栏 | ⚠️ 在 Track 4/3 中提到 safety，但没有统一的 Geofence 模块设计 | **GAP-6** |

### GAP-4: SKILL.md 本体无关技能格式
PROJ.md 的核心价值之一是"本体无关技能共享"——技能以约束参数形式存在，可跨品牌迁移。当前 `OEA/skills/` 是 OEA 自带的技能系统，但没有定义具身技能的 SOP 格式（让一个在 Franka 上验证的抓取技能能迁移到 Dobot 上）。

**建议**: 在 DEVELOPER_GUIDE.md 中增加 SKILL.md 的具身技能格式规范。

### GAP-5: ROS2 统一桥接层
PROJ.md §四.3.1 要求"ROS2 统一数据总线"，但当前每个硬件 Track 各自处理 ROS2 通信。缺少一个统一的 `hal/ros2_bridge.py` 或通信中间件设计。

**建议**: 在 Track 7 (Framework Core) 中增加 ROS2 桥接层的需求。

### GAP-6: 统一电子围栏模块
PROJ.md 要求逻辑围栏 + 物理围栏。逻辑围栏由 Critic 实现（已有），但物理围栏（力矩传感器熔断 → 回写 LESSONS.md）需要在 BaseDriver 接口中预留标准化的 safety callback。

**建议**: 在 `BaseDriver` 中增加 `on_collision(force, position) -> None` 回调和 `safety_limits` 属性。

---

## 五、敏捷落地路线图 (PROJ §五)

| PROJ.md Phase | 当前覆盖 | 差距 |
|--------------|---------|------|
| Phase 1: 桌面闭环 | ✅ 已实现 (EmbodiedActionTool + hal_watchdog + 仿真) | — |
| Phase 2: 视觉解耦 + Go2 | ⚠️ Go2 Track 有，但 Vision MCP 缺失 | **GAP-1** |
| Phase 3: ReKep + 多机协同 + SKILL.md 市场 | ⚠️ ReKep Track 有，多机协同 Track 有，但 SKILL.md 市场缺失 | **GAP-4** |

---

## 差距汇总

| GAP ID | 描述 | 严重程度 | 建议 |
|--------|------|---------|------|
| **GAP-1** | 缺少 Vision MCP Server Track | 高 | 新增 track-8-vision-mcp.md |
| **GAP-2** | 缺少 Franka, Xlerobot, Elfin Driver Track | 中 | 按需新增，或在现有 Track 中扩展 |
| **GAP-3** | 缺少 Refactor Agent (SKILL.md 自主优化) | 低 | 合并到 Track 5 或 Track 6 |
| **GAP-4** | 缺少本体无关技能 SOP 格式 | 中 | 在 DEVELOPER_GUIDE.md 中新增规范 |
| **GAP-5** | 缺少 ROS2 统一桥接层设计 | 中 | 在 Track 7 中增加 |
| **GAP-6** | 缺少统一电子围栏 / Safety 模块 | 中 | 在 BaseDriver 接口中增加 safety 回调 |
