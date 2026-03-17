<div align="center">
  <img src="oea-logo-v3-full_mmmxy7a5.png" alt="OpenEmbodiedAgent" width="500">
  <h1>OpenEmbodiedAgent (OEA)</h1>
  <p><b>基于约束求解与多智能体协同的消费级具身智能框架</b></p>
  <p>
    <a href="./README.md">English</a> | <a href="./README_zh.md">中文</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/version-0.0.1-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

🐈 **OpenEmbodiedAgent (OEA)** 是一个致力于降低机器人使用门槛的开源具身智能框架。它摒弃了传统“大模型直接控制硬件”的危险黑盒模式，首创了**“万物皆 Markdown (State-as-a-File)”**的协议矩阵，并通过**双轨多体系统**（软件大脑 Track A + 硬件小脑 Track B）实现了安全、可解释、可进化的机器人控制。

⚡️ 当前版本 **v0.0.1 (OEA 先行版)** 基于超轻量级的 `nanobot` 架构构建，旨在通过桌面级虚拟宠物和仿真环境，快速验证 OEA 的核心协议与工作流。

## 📢 News

- **2026-03-13** 🚀 Released **v0.0.1** — OEA 先行版发布，确立了“万物皆 Markdown”的核心协议，并跑通了基于仿真环境的软硬解耦与多 Agent 校验流。

## Key Features of OEA:

🪶 **万物皆 Markdown**: 软硬件通过读写本地 Markdown 文件（如 `ENVIRONMENT.md`, `ACTION.md`）进行通信，彻底解耦，极度透明。

🧠 **双轨多体系统**:
- **Track A (大脑)**: 包含 Planner (规划) 与 Critic (校验) 机制。大模型不直接下发指令，必须经过 Critic 对照物理极限（`EMBODIED.md`）校验后才落盘。
- **Track B (小脑)**: 独立的硬件看门狗 (`hal_watchdog.py`) 监听指令并执行。

🛡️ **Anti-Shitstorm 机制**: 严格的动作校验与 `LESSONS.md` 经验避坑库，防止 Agent 工作流失控。

🎮 **仿真环境闭环**: 内置轻量级仿真支持，无需真实硬件即可验证从自然语言指令到物理状态改变的全链路。

## 🏗️ Architecture

OEA 的核心是一个本地工作区（Workspace），软硬件作为独立的守护进程对文件进行读写：

```mermaid
graph TD
    subgraph Track A: 软件大脑 (Software Brain)
        Planner[Planner Agent]
        Critic[Critic Agent]
        Vision[Vision MCP Server]
    end

    subgraph Workspace API: 状态即文件 (State-as-a-File)
        ENV[ENVIRONMENT.md<br/>环境感知]
        EMB[EMBODIED.md<br/>本体声明]
        ACT[ACTION.md<br/>动作指令]
        LES[LESSONS.md<br/>经验避坑]
    end

    subgraph Track B: 硬件小脑 (Hardware HAL)
        Watchdog[HAL Watchdog]
        Sim[Simulation Env / Real Robot]
    end

    Vision -->|写入 Scene-Graph| ENV
    Planner -->|读取| ENV
    Planner -->|读取| LES
    Planner -->|生成草稿| Critic
    Critic -->|读取物理极限| EMB
    Critic -->|校验通过写入| ACT
    Watchdog -->|监听并解析| ACT
    Watchdog -->|驱动| Sim
    Sim -->|状态回传更新| ENV
```

## Table of Contents

- [News](#-news)
- [Key Features](#key-features-of-oea)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Contribute & Roadmap](#-contribute--roadmap)

## 🚀 Quick Start

### 1. 安装依赖

```bash
git clone https://github.com/your-repo/OpenEmbodiedAgent.git
cd OpenEmbodiedAgent
pip install -e .
# 安装仿真环境依赖 (如 watchdog)
pip install watchdog
```

### 2. 初始化工作区

```bash
OEA onboard
```
这将在 `~/.OEA/workspace/` 下生成核心的 Markdown 协议文件（`EMBODIED.md`, `ENVIRONMENT.md` 等）。

### 3. 启动系统

需要开启两个终端：

**终端 1: 启动硬件看门狗与仿真环境 (Track B)**
```bash
python hal/hal_watchdog.py
```

**终端 2: 启动大脑 Agent (Track A)**
```bash
OEA agent
```

### 4. 交互示例

在 `OEA agent` 的 CLI 中输入：
> "看看桌子上有什么，然后把那个苹果推到地上。"

你将在终端 1 的仿真日志中看到动作的执行，并在终端 2 收到 Agent 的完成确认。

## 📁 Project Structure

```text
OpenEmbodiedAgent/
├── OEA/                # Track A: 软件大脑核心 (基于 OEA 扩展)
│   ├── agent/              # Agent 逻辑 (Planner, Critic)
│   ├── templates/          # Workspace Markdown 模板
│   └── ...
├── hal/                    # Track B: 硬件小脑与仿真 (新增)
│   ├── hal_watchdog.py     # 硬件看门狗守护进程
│   └── simulation/         # 仿真环境相关代码
├── workspace/              # 运行时生成的工作区 (Workspace API)
│   ├── EMBODIED.md         # 机器人本体能力声明
│   ├── ENVIRONMENT.md      # 当前环境 Scene-Graph
│   ├── ACTION.md           # 待执行的动作指令
│   ├── LESSONS.md          # 失败经验记录
│   └── SKILL.md            # 成功工作流 SOP
├── docs/                   # 项目文档
│   ├── PLAN.md             # 详细实施方案
│   └── PROJ.md             # 项目白皮书与架构设计
├── README.md               # 英文说明
└── README_zh.md            # 中文说明
```

## 🤝 Contribute & Roadmap

欢迎提交 PR 或 Issue！请参考 `docs/PROJ.md` 了解详细的架构设计与团队分工。

**Roadmap** — Pick an item and open a PR!

- [x] **Phase 1 (当前 v0.0.1): 桌面闭环与 Markdown 协议确立**
  - [x] 扩展 Workspace 模板，确保包含 `EMBODIED.md`, `ENVIRONMENT.md`, `ACTION.md`, `LESSONS.md`, `SKILL.md`
  - [x] 修改 `OEA/agent/context.py` 强制注入 `EMBODIED.md` 和 `ENVIRONMENT.md`
  - [x] 开发 `EmbodiedActionTool` 实现 Critic 校验机制并落盘 `ACTION.md`
  - [x] 配置 Heartbeat 主动唤醒机制
  - [x] 开发 `hal_watchdog.py` 监听 `ACTION.md` 并接入仿真环境执行
  - [x] 联调与测试：运行 `OEA agent` 和 `hal_watchdog.py`（带仿真界面），下发指令验证闭环
- [ ] **Phase 2: 视觉解耦与工具链合并**
  - [ ] 开发真实的视觉感知 Server (MCP Vision Server)
  - [ ] 将相机的多模态信息稳定降维成文本 Scene-Graph 写入 `ENVIRONMENT.md`
  - [ ] 激活 `LESSONS.md` 机制，让模型学会“吃一堑长一智”
  - [ ] 在 ROS2 环境下，跑通 Go2 EDU 四足底盘的指令下发与状态回传
- [ ] **Phase 3: 约束求解与高阶异构协同**
  - [ ] 基于 Franka 和 Xlerobot，完成 C++ 版本的高性能 ReKep 约束求解器
  - [ ] 接入 ROSClaw Bridge
  - [ ] 升级调度逻辑，在 `ACTION.md` 中实现多设备并发指令的时间锁/空间锁
  - [ ] 实现从“桌面 OEA 情感交互”到“一车一臂协同整理客厅”的跨越
  - [ ] 正式上线基于 `SKILL.md` 的社区生态市场