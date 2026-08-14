
# Repository instructions

## 开发前必读
@.agents/AGENTS.md
@Contributing.md
@CONTEXT.md

在commit前运行pre-commit只需要运行ruff和markdownlint就可以
在commit的时候需要sing off
[`vllm/vllm/config/parallel.py`](../vllm/vllm/config/parallel.py#L126) 使用这种格式描述代码。
测试中的模型路径使用环境变量读取，保证本地和上游都有模型用：
os.environ.get("SP_TEST_MODEL", "Qwen/Qwen3-30B-A3B")
上传代码仅使用纯代码上传，不要用仓库自带install编译。
编译请严格按照docs/instructions的步骤。

## 开发环境
目前在远程开发

## 验证条件
全量完成测试验证
额外需要增加e2e功能测试，测试相关功能。

Local `vllm` + `vllm-ascend` development scaffold. `vllm/` and `vllm-ascend/` are Git submodules.

## Remote development model

远程开发统一使用 `.agents/skills/remote-plugin` 的 `remote` CLI（仓库根目录的可执行脚本 `./remote`，经 Bash 调用）。一切远程操作（查占用、同步代码、编译、跑任务、看日志）都走 `remote` CLI，不裸用 ssh。用法与命令速查见 `.agents/skills/remote-plugin/skills/remote-plugin/SKILL.md`。

机器注册的唯一来源是 remote-plugin 仓库内的 `.remote/machines.json`（手写维护）。

## Skills

Repo-local skills live under `.agents/skills/`. Each has its own `SKILL.md` with usage, entry points, and routing rules — read that before invoking.

| Skill | Purpose |
|-------|---------|
| `remote-plugin` | 经 `remote` CLI 在远程机器/容器上查占用、同步代码、编译与跑任务 |
| `repo-init` | Initialize workspace: `gh`, GitHub auth, submodules, fork topology |
| `modelscope` | Download / resume / status-check / SHA256-verify ModelScope model weights under explicit local directories |
| `handoff-context` | 项目上下文交接：读取/构建 CONTEXT.md |
| `concise-code-explanation` | 中文"总—分—总"代码机制解释 |
| `grill` | Stress-test a plan or decision with high-impact questions |

None of these are gates for normal local coding, docs work, or unrelated Git tasks.

## Repo-wide rules

- Never write secrets, passwords, or tokens into tracked files.
- Keep `.gitmodules` on community upstream URLs.
- 远程操作一律走 remote-plugin 的 `remote` CLI，不裸用 ssh / 手写远程命令。
- Skill wrappers: progress on `stderr`, final JSON on `stdout`.
- This repo targets Huawei Ascend NPU. Local machines (Mac/PC) cannot run `torch`/`torch_npu`-dependent code. Do not attempt local test execution — go straight to the remote container.

## Maintenance

When changing a skill, update the whole package together: `SKILL.md`, `scripts/`, `references/`, `agents/`, and other supporting files as applicable.
