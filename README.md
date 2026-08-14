# vllm-ascend-workspace

**中文** | **[English](README.en.md)**

一个可组合的本地开发脚手架，让你在同一个工作区里同时开发 [vLLM](https://github.com/vllm-project/vllm) 和 [vLLM Ascend 插件](https://github.com/vllm-project/vllm-ascend)，并通过内置的 AI Agent 技能和 `remote` CLI 完成环境初始化、远程 NPU 机器操作、代码同步和编译验证。

## 这个项目解决什么问题

vLLM Ascend 的开发通常需要在本地编辑代码、在远程昇腾 NPU 服务器上运行测试，同时还要跟踪上游 vLLM 的变化。手动维护这套工作流涉及大量重复的 Git、SSH 和环境配置操作。

`vllm-ascend-workspace` 把这些操作封装成一组 AI Agent 技能，你可以用自然语言让 Agent 代劳，也可以完全忽略这些技能、只把它当作一个普通的多仓库工作区。

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/maoxx241/vllm-ascend-workspace.git
cd vllm-ascend-workspace

# 初始化子模块
git submodule update --init --recursive
```

如果你使用支持 Agent 的 IDE（Cursor、Windsurf 等）或终端工具（Claude Code、Codex CLI 等），可以直接用自然语言完成后续配置：

> "初始化这个工作区，帮我配好 vLLM Ascend 的开发环境。"

Agent 会自动检测你的环境、安装所需工具、配置 Git 远程仓库和 Fork。

## 内置技能


| 技能                       | 用途                                             | 何时使用               |
| ------------------------ | ---------------------------------------------- | ------------------ |
| **remote-plugin**        | 经 `remote` CLI 在远程机器/容器上查占用、同步代码、编译与跑任务      | 需要操作远程 NPU 机器完成开发、编译验证、冒烟测试时 |
| **repo-init**            | 安装 GitHub CLI、登录 GitHub、初始化子模块、配置 Fork 和远程仓库拓扑 | 首次 clone 后初始化工作区   |
| **modelscope**           | 下载、续传、查看进度并 SHA256 校验 ModelScope 模型权重                  | 需要把模型权重下载到明确目录时 |
| **handoff-context**      | 读取/构建 CONTEXT.md，完成项目上下文交接                  | 新 agent 接手或需要更新上下文时 |
| **concise-code-explanation** | 用中文"总—分—总"结构解释代码机制与调用链                | 需要代码阅读、模块设计分析时 |
| **grill**                | 对计划/决策做高压质询，绘制决策树并批量提出高影响问题             | 需要严格挑战一个方案时 |


所有技能都是**可选的**。你可以只用其中的一部分，也可以完全不用。

## 使用示例

与 Agent 对话时，可以这样说：

```
# 初始化
"帮我初始化一下这个仓库"
"帮我配置一下这个仓库"

# 远程操作（经 remote-plugin 的 remote CLI）
"看下哪台机器空闲"
"帮我同步代码到服务器上并重新编译"
"在 x.x.x.x 上跑一下冒烟测试"

# 模型权重下载
"帮我把 Qwen/Qwen3-32B 从 ModelScope 下载到 /root/Qwen/Qwen3-32B，并查看进度"
"校验一下 /root/Qwen/Qwen3-32B 的 ModelScope 权重"
```

## 仓库结构

```
.
├── vllm/                  # vLLM 上游（Git 子模块）
├── vllm-ascend/           # vLLM Ascend 插件（Git 子模块）
├── .agents/
│   └── skills/
│       ├── remote-plugin/         # remote CLI 远程开发插件
│       ├── repo-init/             # 工作区初始化技能
│       ├── modelscope/            # ModelScope 权重下载与校验技能
│       ├── handoff-context/       # 项目上下文交接技能
│       ├── concise-code-explanation/ # 代码机制解释技能
│       └── grill/                 # 方案质询技能
├── .cursor/rules/         # Cursor IDE 专用规则
├── .trae/                 # TRAE IDE 专用规则与技能
├── AGENTS.md              # 跨工具 Agent 指令（AI Agent 读这个）
├── CLAUDE.md              # Claude Code 指令入口
└── README.md              # 你正在看的这个文件
```

## 设计原则

- **不强制任何流程** — 所有技能都可选，开发者自由选择使用哪些部分。
- **本地状态不入库** — 用户特定的远程仓库、认证信息、机器配置等只存在于本地未跟踪的目录中（机器注册信息手写维护在 remote-plugin 的 `.remote/machines.json`）。
- **远端操作走 remote CLI** — 一切远程操作（查占用、同步、编译、跑任务）统一经 remote-plugin 的 `remote` CLI，不裸用 ssh。
- **子模块指向社区** — `.gitmodules` 始终指向 `vllm-project` 的官方仓库，个人 Fork 是本地运行时配置。
- **Agent 驱动，但不依赖 Agent** — 所有操作都可以手动完成，Agent 只是让流程更方便。

## 推荐的远程仓库拓扑

技能会推荐以下拓扑结构，但不强制要求：


| 仓库            | `origin`    | `upstream`                       |
| ------------- | ----------- | -------------------------------- |
| workspace     | 你的 Fork（可选） | `maoxx241/vllm-ascend-workspace` |
| `vllm`        | 你的 Fork（可选） | `vllm-project/vllm`              |
| `vllm-ascend` | 你的 Fork     | `vllm-project/vllm-ascend`       |


## 多工具支持

本仓库支持主流 AI 编程工具：


| 文件               | 覆盖工具                                    |
| ---------------- | ---------------------------------------- |
| `AGENTS.md`      | Codex CLI、GitHub Copilot、Cursor、TRAE、OpenCode |
| `CLAUDE.md`      | Claude Code                              |
| `.cursor/rules/` | Cursor                                   |
| `.trae/`         | TRAE                                     |


## Roadmap

### 已完成

- [x] **repo-init** — 工作区初始化：GitHub CLI 安装、认证、子模块、Fork 与远程仓库拓扑配置
- [x] **remote-plugin** — 远程开发：`remote` CLI 统一完成机器占用查询、代码同步、远程编译与任务执行

### 计划中

- [ ] **精度测试与 aisbench 集成** — 基于 aisbench 的自动化评测，支持 HTML 报告自动分析、系统调度评估及 DP 均衡度分析
- [ ] **性能 Profiling 分析** — 自动分析模型主要算子耗时，热点算子 AIC/AIV/MTE2 ratio 分析，AICPU 算子识别，host bound 识别与诊断
- [ ] **同步打断优化** — 针对具体 case 提供异步拷贝掩盖方案，减少同步等待开销
- [ ] **计算图分析** — 构建模型计算图，提供基于计算图的理论性能评估报告及优化方案
- [ ] **外置知识库接入** — 接入外部知识库，扩展 Agent 的能力边界

## 许可证

本脚手架仓库的许可证独立于子模块。`vllm/` 和 `vllm-ascend/` 各自遵循其上游项目的许可证。
