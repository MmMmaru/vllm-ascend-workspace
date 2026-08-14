# vllm-ascend-workspace

**[中文](README.md)** | **English**

A composable local development scaffold for working on [vLLM](https://github.com/vllm-project/vllm) and [vLLM Ascend Plugin](https://github.com/vllm-project/vllm-ascend) in a single workspace, with built-in AI Agent skills and a `remote` CLI for environment setup, remote NPU machine operations, and code synchronization.

## What problem does this solve

Developing vLLM Ascend typically involves editing code locally, running tests on remote Ascend NPU servers, and tracking upstream vLLM changes — all of which require repetitive Git, SSH, and environment configuration.

`vllm-ascend-workspace` wraps these operations into a set of AI Agent skills. You can ask an Agent to handle them in natural language, or ignore the skills entirely and use it as a plain multi-repo workspace.

## Quick start

```bash
# Clone the repository
git clone https://github.com/maoxx241/vllm-ascend-workspace.git
cd vllm-ascend-workspace

# Initialize submodules
git submodule update --init --recursive
```

If you use an Agent-capable IDE (Cursor, Windsurf, etc.) or terminal tool (Claude Code, Codex CLI, etc.), you can complete the rest of the setup in natural language:

> "Initialize this workspace and set me up for vLLM Ascend development."

The Agent will detect your environment, install required tools, and configure Git remotes and forks.

## Built-in skills


| Skill                  | Purpose                                                                                      | When to use                                                |
| ---------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **remote-plugin**      | Operate remote machines/containers via the `remote` CLI: check occupancy, sync code, build, and run jobs | When a task needs a remote NPU machine for development, build verification, or smoke tests |
| **repo-init**          | Install GitHub CLI, authenticate, initialize submodules, configure forks and remote topology | After first clone                                          |
| **modelscope**       | Download, resume, status-check, and SHA256-verify ModelScope model weights                  | When model weights need to be downloaded into an explicit local directory |
| **handoff-context**  | Read/build CONTEXT.md for project context handoff                                          | When a new agent takes over or context needs refreshing    |
| **concise-code-explanation** | Explain code mechanisms and call chains in Chinese, structured overview-to-detail    | For code reading and module design analysis                |
| **grill**            | Stress-test a plan or decision with high-impact questions                                  | When a plan needs rigorous challenge                       |


All skills are **optional**. Use any subset, or none at all.

## Usage examples

When talking to an Agent:

```
# Initialization
"Help me initialize this repo"
"Help me set up this repo"

# Remote operations (via remote-plugin's remote CLI)
"Show me which machines are idle"
"Sync my code to the server and rebuild"
"Run a smoke test on x.x.x.x"

# Model weight download
"Download Qwen/Qwen3-32B from ModelScope to /root/Qwen/Qwen3-32B and show progress"
"Verify the ModelScope weights under /root/Qwen/Qwen3-32B"
```

## Repository layout

```
.
├── vllm/                  # Upstream vLLM (Git submodule)
├── vllm-ascend/           # vLLM Ascend Plugin (Git submodule)
├── .agents/
│   └── skills/
│       ├── remote-plugin/         # remote CLI remote-development plugin
│       ├── repo-init/             # Workspace initialization skill
│       ├── modelscope/            # ModelScope weight download and verification skill
│       ├── handoff-context/       # Project context handoff skill
│       ├── concise-code-explanation/ # Code explanation skill
│       └── grill/                 # Plan stress-testing skill
├── .cursor/rules/         # Cursor IDE specific rules
├── .trae/                 # TRAE IDE specific rules and skills
├── AGENTS.md              # Cross-tool Agent instructions (Agents read this)
├── CLAUDE.md              # Claude Code instruction entry point
└── README.md              # Chinese README (default)
```

## Design principles

- **Nothing is mandatory** — All skills are optional. Developers choose what to use.
- **Local state stays untracked** — User-specific remotes, auth, and machine config live only in untracked local files (machine registry is hand-maintained in remote-plugin's `.remote/machines.json`).
- **Remote operations go through the remote CLI** — All remote work (occupancy checks, sync, builds, jobs) goes through remote-plugin's `remote` CLI, never raw ssh.
- **Submodules point to community** — `.gitmodules` always targets `vllm-project` official repos. Personal forks are a local runtime concern.
- **Agent-driven, not Agent-dependent** — Everything can be done manually. Agent skills just make it more convenient.

## Recommended remote topology

Skills recommend the following topology, but never enforce it:


| Repository    | `origin`             | `upstream`                       |
| ------------- | -------------------- | -------------------------------- |
| workspace     | Your fork (optional) | `maoxx241/vllm-ascend-workspace` |
| `vllm`        | Your fork (optional) | `vllm-project/vllm`              |
| `vllm-ascend` | Your fork            | `vllm-project/vllm-ascend`       |


## Multi-tool support

This repository supports mainstream AI coding tools:


| File             | Tools covered                                     |
| ---------------- | ------------------------------------------------- |
| `AGENTS.md`      | Codex CLI, GitHub Copilot, Cursor, TRAE, OpenCode |
| `CLAUDE.md`      | Claude Code                                       |
| `.cursor/rules/` | Cursor                                            |
| `.trae/`         | TRAE                                              |


## Roadmap

### Done

- **repo-init** — Workspace initialization: GitHub CLI install, auth, submodules, fork & remote topology
- **remote-plugin** — Remote development: the `remote` CLI unifies machine occupancy checks, code sync, remote builds, and job execution

### Planned

- **Accuracy testing & aisbench integration** — Automated evaluation based on aisbench, with HTML report analysis, system scheduling assessment, and DP balance analysis
- **Performance profiling** — Automatic operator latency breakdown, hot operator AIC/AIV/MTE2 ratio analysis, AICPU operator identification, host bound detection and diagnosis
- **Sync-break optimization** — Provide async copy overlap strategies for specific cases to reduce synchronization overhead
- **Compute graph analysis** — Build model compute graphs, generate theoretical performance evaluation reports and optimization recommendations
- **External knowledge base** — Integrate external knowledge sources to extend Agent capabilities

## License

This scaffold repository is licensed independently from its submodules. `vllm/` and `vllm-ascend/` each follow their respective upstream licenses.
