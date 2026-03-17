# OpenEmbodiedAgent — Implementation Plan (v0.0.2)

> Last Updated: 2026-03-15

## Overview

This document is the top-level index for the OEA development plan. The project is organized into **10 parallel development tracks**, each with its own specification document. This version incorporates findings from the [GAP_ANALYSIS.md](../GAP_ANALYSIS.md) cross-referencing with [PROJ.md](../PROJ.md).

For the overall architecture and contribution rules, see [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md).

## Development Tracks

| Track | Title | Spec | Priority | Dependency |
|-------|-------|------|----------|------------|
| 1 | Simulation Arm (PyBullet) | [track-1-simulation.md](../track-1-simulation.md) | High | Track 7 |
| 2 | Desktop Pet Robot | [track-2-desktop-pet.md](../track-2-desktop-pet.md) | Medium | Track 7 |
| 3 | Dobot Dual-Arm (ReKep) | [track-3-dobot-rekep.md](../track-3-dobot-rekep.md) | High | Track 7 |
| 3b | AgileX Piper (ReKep, single-arm) | [track-3b-piper-rekep.md](../track-3b-piper-rekep.md) | High | Track 7, shares rekep_solver with Track 3 |
| 4 | Unitree Go2 Navigation | [track-4-go2-navigation.md](../track-4-go2-navigation.md) | High | Track 7 |
| 5 | Multi-Agent Orchestrator | [track-5-multi-agent.md](../track-5-multi-agent.md) | High | None |
| 6 | Spatiotemporal Memory | [track-6-spatiotemporal-memory.md](../track-6-spatiotemporal-memory.md) | Medium | None |
| 7 | Framework Core | [track-7-framework-core.md](../track-7-framework-core.md) | **Critical** | None |
| 8 | Vision MCP Server | TBD | High | None |
| 9 | Franka / Xlerobot / Elfin Drivers | TBD | Medium | Track 7 |
| 10 | Skill SOP & Refactor Agent | TBD | Low | Track 5, Track 6 |

### Tracks added in v0.0.2 (from Gap Analysis)

- **Track 8 — Vision MCP Server** (GAP-1): Develops the MCP-based visual perception service that runs a lightweight VLM (e.g., LLaVA, Qwen-VL) to capture camera frames and convert them into a text Scene-Graph, overwriting `ENVIRONMENT.md`. This is required by PROJ.md §四.2.1 and is the Phase 2 critical-path item.

- **Track 9 — Franka / Xlerobot / Elfin Drivers** (GAP-2): Covers the remaining robot bodies listed in PROJ.md §三.2 that are not yet assigned a track. Each body gets its own `hal/drivers/` file and `hal/profiles/` EMBODIED.md. Can be split into sub-tracks as team members are assigned.

- **Track 10 — Skill SOP & Refactor Agent** (GAP-3 + GAP-4): Defines the body-agnostic skill format in `SKILL.md` (so a grasping skill validated on Franka can migrate to Dobot) and implements the Refactor Agent that autonomously cleans redundant steps from SKILL.md during idle time.

### Cross-cutting concerns folded into Track 7

- **ROS2 unified bridge layer** (GAP-5): A shared `hal/ros2_bridge.py` abstraction that hardware Tracks 3, 4, 9 can import instead of each writing their own ROS2 boilerplate.
- **Safety / Geofence module** (GAP-6): `BaseDriver` gains `safety_limits` property and `on_collision()` callback. Physical fence violation → auto-write to `LESSONS.md`.

## Execution Order

```
Phase 0 (now):  Track 7 — BaseDriver + watchdog + context.py
                Track 5 — Orchestrator protocol (TASK.md)
                Track 6 — Memory protocol (MEMORY_SPATIAL.md)
                    │
Phase 1:        Track 1 — Simulation driver (reference impl)
                Track 8 — Vision MCP Server prototype
                    │
Phase 2:        Tracks 2,3,4,9 — All hardware drivers (parallel)
                Track 10 — Skill SOP format + Refactor Agent
                    │
Phase 3:        Integration — Multi-device coordination tests
                             SKILL.md community marketplace
```

## Document Index

| Document | Purpose |
|----------|---------|
| [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) | Architecture overview, file ownership rules, getting started |
| [PROJ.md](../PROJ.md) | Original project whitepaper and vision |
| [GAP_ANALYSIS.md](../GAP_ANALYSIS.md) | PROJ.md vs current framework gap analysis |
| [track-1-simulation.md](../track-1-simulation.md) | PyBullet simulation arm specification |
| [track-2-desktop-pet.md](../track-2-desktop-pet.md) | Desktop pet robot specification |
| [track-3-dobot-rekep.md](../track-3-dobot-rekep.md) | Dobot dual-arm ReKep specification |
| [track-3b-piper-rekep.md](../track-3b-piper-rekep.md) | AgileX Piper single-arm ReKep specification |
| [track-4-go2-navigation.md](../track-4-go2-navigation.md) | Unitree Go2 navigation specification |
| [track-5-multi-agent.md](../track-5-multi-agent.md) | Multi-agent orchestrator specification |
| [track-6-spatiotemporal-memory.md](../track-6-spatiotemporal-memory.md) | Spatiotemporal memory specification |
| [track-7-framework-core.md](../track-7-framework-core.md) | Framework core specification |
| Track 8-10 specs | TBD — to be written when owners are assigned |

---

# 中文版：OpenEmbodiedAgent 实施计划 (v0.0.2)

> 最后更新：2026-03-15

## 概述

本文档是 OEA 开发计划的总索引。项目被组织为 **10 个并行开发轨道 (Track)**，每个轨道有独立的规格文档。本版本整合了 [GAP_ANALYSIS.md](../GAP_ANALYSIS.md) 中与 [PROJ.md](../PROJ.md) 的逐条对照结果。

## 开发轨道

| 轨道 | 标题 | 规格文档 | 优先级 | 依赖 |
|------|------|---------|--------|------|
| 1 | 仿真机械臂 (PyBullet) | [track-1-simulation.md](../track-1-simulation.md) | 高 | Track 7 |
| 2 | 桌面机器人 (宠物) | [track-2-desktop-pet.md](../track-2-desktop-pet.md) | 中 | Track 7 |
| 3 | 越疆双臂 (ReKep 约束求解) | [track-3-dobot-rekep.md](../track-3-dobot-rekep.md) | 高 | Track 7 |
| 3b | 松灵Piper | [track-3b-piper-rekep.md](../track-3b-piper-rekep.md) | 高 | Track 7, shares rekep_solver with Track 3 |
| 4 | 宇树 Go2 四足导航 | [track-4-go2-navigation.md](../track-4-go2-navigation.md) | 高 | Track 7 |
| 5 | 多智能体中控 (Orchestrator) | [track-5-multi-agent.md](../track-5-multi-agent.md) | 高 | 无 |
| 6 | 时空记忆系统 | [track-6-spatiotemporal-memory.md](../track-6-spatiotemporal-memory.md) | 中 | 无 |
| 7 | 框架核心 (BaseDriver + 看门狗) | [track-7-framework-core.md](../track-7-framework-core.md) | **关键路径** | 无 |
| 8 | 视觉感知 MCP 服务 | 待编写 | 高 | 无 |
| 9 | Franka / Xlerobot / Elfin 驱动 | 待编写 | 中 | Track 7 |
| 10 | 技能 SOP 格式与重构 Agent | 待编写 | 低 | Track 5, 6 |

### v0.0.2 新增轨道 (来自差距分析)

- **Track 8 — 视觉感知 MCP 服务** (GAP-1)：开发基于 MCP 协议的视觉感知服务。运行轻量级 VLM (如 LLaVA、Qwen-VL) 对摄像头画面进行分析，将其转化为纯文本 Scene-Graph 并覆写 `ENVIRONMENT.md`。这是 PROJ.md 第四章第2节第1条的核心要求，也是 Phase 2 的关键阻塞项。

- **Track 9 — Franka / Xlerobot / Elfin 驱动** (GAP-2)：覆盖 PROJ.md 第三章第2节列出但尚未分配 Track 的机器人本体。每种本体在 `hal/drivers/` 中有独立的 Driver 文件，在 `hal/profiles/` 中有独立的 EMBODIED.md。当负责人确定后可拆分为子轨道。

- **Track 10 — 技能 SOP 格式与重构 Agent** (GAP-3 + GAP-4)：定义"本体无关"的技能格式（让在 Franka 上验证过的抓取技能可以迁移到 Dobot 上），并实现 PROJ.md 提出的"记忆重构者 (Refactor Agent)"——在空闲时段自动清理 `SKILL.md` 中的冗余步骤。

### 合并到 Track 7 的横切关注点

- **ROS2 统一桥接层** (GAP-5)：提供共享的 `hal/ros2_bridge.py` 抽象层，供 Track 3、4、9 的硬件驱动共用，避免各自重复编写 ROS2 通信代码。
- **安全模块 / 电子围栏** (GAP-6)：在 `BaseDriver` 接口中增加 `safety_limits` 属性和 `on_collision()` 回调。当物理围栏被触发时（如力矩传感器检测到碰撞），自动将事件写入 `LESSONS.md`。

## 执行顺序

```
Phase 0（当前）: Track 7 — BaseDriver 抽象类 + 看门狗重构 + context.py 动态扫描
                 Track 5 — 中控协议设计 (TASK.md)
                 Track 6 — 记忆协议设计 (MEMORY_SPATIAL.md)
                     │
Phase 1:         Track 1 — 仿真驱动（参考实现）
                 Track 8 — 视觉 MCP 服务原型
                     │
Phase 2:         Tracks 2,3,4,9 — 所有硬件驱动（并行开发）
                 Track 10 — 技能 SOP 格式 + 重构 Agent
                     │
Phase 3:         集成测试 — 多设备协同测试
                            SKILL.md 社区技能市场上线
```

---

## 术语表 (Glossary)

| 术语 | 英文全称 | 含义 |
|------|---------|------|
| **HAL** | Hardware Abstraction Layer | 硬件抽象层。一层软件接口，屏蔽不同硬件的底层差异，让上层代码不需要关心具体是哪种机器人。类比：显卡驱动就是一种 HAL，让游戏不需要为每张显卡写不同的代码。 |
| **BaseDriver** | — | OEA 定义的抽象基类。每种机器人本体（仿真臂、桌面宠物、越疆双臂、Go2 四足等）必须实现这个接口的 4 个方法：`get_profile_path()`、`load_scene()`、`execute_action()`、`get_scene()`。 |
| **MCP** | Model Context Protocol | 模型上下文协议。OEA 支持的外部工具服务器协议，允许 LLM 调用外部工具（如视觉识别、网页搜索）。Vision MCP Server 就是一个通过 MCP 提供视觉感知能力的服务。 |
| **ReKep** | Relational Keypoint Constraints | 关系关键点约束。一种将机器人操作任务描述为几何约束的范式，例如"末端执行器与杯口对齐，且 Z 轴向上"。这些约束被转化为二次规划 (QP) 问题并求解为关节轨迹。 |
| **Scene-Graph** | — | 场景图。用结构化文本描述环境中有哪些物体、它们的位置和空间关系。例如：`红苹果 在 桌子上，坐标(5,5,0)，蓝杯子 在 红苹果 右侧`。这是大模型"看到"世界的唯一方式。 |
| **Workspace** | — | 工作区。位于 `~/.OEA/workspace/` 的一个本地文件夹，所有 `.md` 文件都存放在这里。软件大脑和硬件小脑都通过读写这个文件夹中的文件来通信，不直接调用对方的代码。 |
| **Planner Agent** | — | 规划者。使用最强基座大模型的核心角色。读取用户指令和环境信息，生成动作计划并写入 `ACTION.md` 草稿。 |
| **Critic Agent** | — | 校验官。独立的校验角色。在 Planner 写出动作草稿后，对照 `EMBODIED.md` 中的物理极限进行安全校验。只有通过校验的动作才会被正式写入 `ACTION.md`。 |
| **Orchestrator Agent** | — | 中控调度者。当任务涉及多个机器人协同时，负责将任务分解为子任务、分配给不同设备、监控执行进度。通过 `TASK.md` 和 `ORCHESTRATOR.md` 管理状态。 |
| **Refactor Agent** | — | 记忆重构者。在空闲时段（如深夜）自动触发，清理 `SKILL.md` 中的冗余步骤，优化工作流记忆。 |
| **HAL Watchdog** | hal_watchdog.py | 硬件看门狗守护进程。一个独立运行的 Python 脚本，持续轮询 `ACTION.md` 文件。当检测到新的动作指令时，调用当前加载的 Driver 执行动作，然后将执行后的环境状态回写到 `ENVIRONMENT.md`，并清空 `ACTION.md` 表示完成。通过 `--driver` 参数选择不同的硬件本体。它是 Track A（软件大脑）和 Track B（硬件小脑）之间的唯一桥梁。 |
| **Heartbeat** | — | 心跳机制。OEA 内置的定时唤醒功能，每 30 分钟检查 `HEARTBEAT.md` 中的任务。在具身场景中用于主动检测环境变化并通知用户。 |
| **Profile** | — | 本体档案。每种机器人硬件在 `hal/profiles/` 下有一个 `.md` 文件，描述该机器人的能力、关节极限、支持的动作等。启动看门狗时通过 `--driver` 参数选择档案，自动复制到工作区的 `EMBODIED.md`。 |
| **Geofence** | Electronic Geofence | 电子围栏。分为逻辑围栏（Critic 在下发动作前根据坐标预判是否越界）和物理围栏（硬件层的力矩传感器在检测到碰撞时立即熔断停止）。 |
| **SOP** | Standard Operating Procedure | 标准操作流程。在 OEA 中特指存储在 `SKILL.md` 中的、经过验证的可复用工作流。 |
| **QP** | Quadratic Programming | 二次规划。ReKep 求解器将关键点约束转化为 QP 问题，并在 < 50ms 内求解出关节轨迹。 |
| **IK** | Inverse Kinematics | 逆运动学。给定末端执行器的目标位姿，计算出各关节需要的角度。 |
| **DDS** | Data Distribution Service | 数据分发服务。ROS2 底层使用的通信中间件，支持发布/订阅模式。Go2 通过 DDS 与外部通信。 |
| **VLM** | Vision-Language Model | 视觉语言模型。能同时理解图像和文本的 AI 模型（如 LLaVA、Qwen-VL）。在 OEA 中用于将摄像头画面转化为文本 Scene-Graph。 |
| **EE** | End-Effector | 末端执行器。机械臂最末端的工具，如夹爪、吸盘。 |
| **URDF** | Unified Robot Description Format | 统一机器人描述格式。描述机器人物理结构（关节、连杆、惯性等）的 XML 格式文件，被 PyBullet 等仿真器使用。 |
