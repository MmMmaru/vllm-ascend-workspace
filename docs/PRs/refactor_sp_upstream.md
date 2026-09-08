# 人类编写
## 目的
这个PR只看上游相关代码，忽略下游。
1、下沉allgather、reduce scatter到linear里
2、添加SP开关（moe 满足之前的条件默认开，可以手动关闭），dense token数量大于一定数量走SP
3、需要讨论pad逻辑是否保留（先不管，先保留）
4、改模型文件（先做一版qwen3.5的）

## 测试
使用nvidia的卡H20
前后benchmark

# SP 重构方案：09-07 grill 对齐版

## 现状基线

- 分支：`vllm/` 子模块 `sp_refactor_new`（上游 main `504bb8b0c3` + 已有 6 个 commit，
  HEAD `52afa9ad07`），worktree 在 workspace 根目录 `sp-refactor-new/`。
- 已有实现：linear 四个类已加 `sequence_parallel` 构造参数（Column 入口 AG / Row 出口 RS 替代
  all-reduce）；qwen3_next / qwen3_5 已迁移；显式开关 `enable_sequence_parallel_moe` 已加但
  校验仍要求 DP>1；RS 内自带 pad（`linear.py:1808-1813`）；`sequence_parallel_unpadded_size`
  穿线仍保留（linear / LoRA / GDN 4 个平台 forward）。

## 已确认决策（09-07 与用户 grill 对齐，以此为准，覆盖旧表）

| # | 决策点 | 结论 |
|---|--------|------|
| 1 | 命名 | 全套改名叫 SP：字段 `enable_sequence_parallel`、CLI `--enable-sequence-parallel`、property `use_sequence_parallel_moe` → `use_sequence_parallel`（约 38 处读取点一起改）；模型侧 `use_attn_reduce_scatter_for_moe` 别名同步清理（qwen3_next_mtp / qwen3_5_mtp 跟着改） |
| 2 | 显式开关语义 | `None` 走现有派生启发式（含 DP>1 要求）；`False` 强制关；`True` 强制开并校验：MoE 模型要求 EP + TP>1 + all2all backend 白名单（**不再要求 DP>1**）；dense 模型不受 EP/白名单限制（TP>1 即可） |
| 3 | DP=1 MoE | `use_all2all` 联动保持不动（SP on 会置真），DP=1 下靠 H20 实测暴露问题再修 |
| 4 | dense SP 机制 | 运行时动态切换：每步 token 数 > 1000（硬编码阈值）时 runner pad + embedding 后 chunk，linear 经 forward context 标志做 AG/RS；≤1000 走原 TP 路径。dense MLP 分支接 `sequence_parallel=`（推翻旧决策 9） |
| 5 | pad | 先保留 RS 内 pad 版本跑通；之后做 "RS 不 pad + runner pad" A/B 前后对照实验再定去留 |
| 6 | commit 历史 | 不动现有 6 个 commit，最后统一 squash/合并 |
| 7 | 验证环境 | 全程 H20（90.90.81.179 / 90.90.81.182 / 90.90.81.187，各 8 卡），完全不碰 NPU |
| 8 | 模型范围 | 仅 qwen3.5 家族（`qwen3_5.py` + `qwen3_next.py` + GDN 构造器）；qwen3_moe / deepseek / kimi_k3 等不动 |
| 9 | 下游协同 | 代码完全不管 vllm-ascend；breaking 点（改名等）在 PR 描述声明 |

## 已知问题清单（按处理顺序）

1. **疑似 bug（DP=1 精度首疑）**：`qwen3_next.py` `Qwen3NextSparseMoeBlock.forward` 里
   `replicated_shared_output` 计算后未加回 `final_hidden_states`（旧 `+=` 被删未补），
   `replicate_shared_expert=True` 时共享专家输出整个丢失。先查证 `FusedMoEFactory` 是否
   兜底，不兜底则补回。
2. 校验放开：当前显式 True 要求 DP>1，DP=1 启动即 `ValueError`（`config/parallel.py:888-895`），
   按决策 2 改。
3. `sequence_parallel_unpadded_size` 穿线暂保留，随 dense SP 的 forward context 方案一起评估
   是否删除。
4. MoE 内 chunk 已删（只靠 embedding 后 chunk），维持现状。

## 执行顺序

1. 改名（config / CLI / property / 读取点 / MTP）+ 校验规则放开（决策 2）。
2. 查证并修共享专家输出丢失；H20 上复现 DP=1 精度问题（对比 SP-off 基线）。
3. 修到 DP=1 + 显式 SP 精度与不开 SP 相当。
4. dense SP + 阈值 1000 运行时动态切换。
5. pad A/B 对照实验（保留版 vs RS 不 pad + runner pad）。
6. 收尾：squash commit、pre-commit（ruff + markdownlint）、H20 benchmark 前后对比、发 PR。

## 验收

- DP=1 + 显式开 SP：精度与 SP-off 基线相当（H20，Qwen3.5 家族模型，logprobs / 输出对照）。
- 默认 fallback（不显式配置）行为与 main 一致。
- dense SP：token > 1000 走 SP、≤1000 走 TP，两路径输出正确。
- vllm 单测 `test_linear_sequence_parallel.py` 通过；pre-commit 干净。
- H20 前后 benchmark 对比附在 PR 描述；PR 声明 AI 参与。

## 遗留（本轮不做）

- PP + SP（pp>1 时 SP 整体关闭）。
- 其他模型家族迁移（deepseek 系、qwen3_moe、kimi_k3 等）。
- vllm-ascend 侧适配（改名 follow-up 记在 PR 描述）。

## 09-08 代码实现（未验证）

- 已统一 config / CLI / property 命名；其他模型仅同步 property 读取名称，不迁移通信逻辑。
- 显式 MoE SP 支持 DP=1；模型类型确定后再次校验 EP / backend；PP>1 关闭 SP。
- 复制共享专家未传入 FusedMoEFactory，已补回其输出。
- dense 使用未 padding 的步 token 数判断 `>1000`，runner pad、模型 chunk/gather、
  linear AG/RS 由同一个 forward context 标志驱动；dense MLP 保持 TP 权重切分。
- 复用的 Qwen2MoeMLP 只增加独立的 `sequence_parallel` 参数，原共享专家行为保留。
- 首版动态 dense SP 暂关闭 CUDA graph，避免 capture bucket 跨阈值复用错误布局；
  后续需实现区分 SP/TP 布局的 graph dispatch，再评估性能。
- 保留 RS 内 pad 和 unpadded_size 穿线。补充配置、linear 动态切换和 Qwen3.5 E2E 用例；
  E2E 分别通过 `SP_DENSE_MODEL` / `SP_MOE_MODEL` 指定本地权重。
- 按用户最新要求，本轮不运行测试、lint、H20 精度验证或 benchmark，不做 pad A/B、squash 或发布。
- Breaking change：旧 `enable_sequence_parallel_moe` / `use_sequence_parallel_moe` / CLI 名称删除，
  vllm-ascend 需后续适配；本次代码由 AI 辅助编写，尚未通过验证。

## PR 草稿

### Title

`[Core] Move sequence parallel collectives into linear layers for Qwen3.5`

### Purpose and Changes

This change moves sequence-parallel (SP) communication for the Qwen3.5 family into the
linear layers, unifies the SP switch, and adds a token-count-based SP/TP execution path
for dense models. In SP mode, ColumnParallelLinear, MergedColumnParallelLinear, and
QKVParallelLinear perform token all-gather on their inputs. RowParallelLinear performs
reduce-scatter on its outputs to combine TP partial sums and shard the tokens. The model
still retains the chunk after the embedding layer and the gather at the output.

The migration covers the shared Qwen3.5/Qwen3-Next implementation, Gated DeltaNet,
and the corresponding MTP paths. Other models only update the configuration property
name; their communication implementation is not migrated. LoRA linear paths reuse the
linear-layer input preparation and output reduction logic.

This adds the `enable_sequence_parallel` configuration and the
`--enable-sequence-parallel` CLI option with the following semantics:

| Configuration | MoE | Dense |
| --- | --- | --- |
| `None` (default) | Preserve the existing automatic enablement conditions: DP > 1, TP > 1, EP, and a supported backend | Not enabled automatically |
| `False` | Disabled | Disabled |
| `True` | Requires TP > 1, EP, and a supported backend; DP = 1 is allowed | Requires TP > 1; use SP when the number of tokens in a step is > 1000, otherwise use TP |

When PP > 1, SP controlled by this configuration is disabled. This switch is separate
from the compilation pass's `enable_sp` setting; the two mechanisms are not merged in
this PR.

The dense path generates a forward-context flag from the unpadded number of tokens in
each step. This flag consistently controls the model's chunk/gather operations and the
linear-layer AG/RS operations. The runner pads the token count of SP steps to a multiple
of the TP size. Dense MLPs retain TP-sharded weights and enable communication through
linear-layer parameters.

The output of replicated shared experts is also added back. In replicated mode, the
shared expert is not passed to `FusedMoEFactory`, so its separately computed result must
be added to the routed-expert output on the model side.

### Current Limitations and Compatibility

- The internal reduce-scatter padding and the `sequence_parallel_unpadded_size`
  parameter are retained. This version of the PR does not remove the padding or include
  an A/B experiment for the padding strategy.
- Dynamic dense SP currently disables CUDA graphs to prevent an incorrect layout from
  being reused when a graph bucket crosses the token threshold. Restoring CUDA graph
  support requires separate capture and dispatch paths for SP and TP layouts; no
  performance benefit is claimed at this stage.
- `use_sequence_parallel_moe` has been renamed to `use_sequence_parallel`, with no
  alias for the old name. The earlier branch-specific
  `enable_sequence_parallel_moe` / `--enable-sequence-parallel-moe` names have also
  been unified; the Qwen3.5 family no longer uses the
  `use_attn_reduce_scatter_for_moe` alias.
- Downstream plugins that reference the old names must be updated. This PR does not
  include changes on the vllm-ascend side.

### Testing and Model Evaluation

This is currently an unvalidated draft. The following test cases have been added to the
code but have not been run. There are no test, lint, accuracy-evaluation, or benchmark
results yet, so these changes must not be considered evidence that the DP = 1 accuracy
issue has been resolved.

| Coverage | Test case | Status |
| --- | --- | --- |
| Dense switch, 1000/1001 threshold, validation after model-type resolution, and PP disablement | `tests/config/test_sequence_parallel.py` | Not run |
| Linear AG/RS, padding, LoRA, and restoration of TP for short steps | `tests/model_executor/layers/test_linear_sequence_parallel.py` | Not run |
| MoE default heuristic, explicit DP = 1, and unsupported topologies | Configuration cases in `tests/models/kimi_k3/test_sequence_parallel.py` | Not run |
| Qwen3.5 dense/MoE with DP = 1 and TP = 2: SP on/off output and logprob comparison | `tests/compile/correctness_e2e/test_sequence_parallel.py::test_qwen35_explicit_sp_dp1` | Not run |

The new E2E test uses `SP_DENSE_MODEL` / `SP_MOE_MODEL` to specify local Qwen3.5
weights and covers prefill with 1001, 1000, and 17 tokens, followed by decode. The test
currently uses eager mode and does not cover compiled execution, CUDA graphs, or MTP
accuracy.

Before submitting the PR for review, add the test commands and results from H20, the
model paths and versions, and an SP on/off accuracy comparison. Also record the
benchmark configurations and TTFT, TPOT, and throughput before and after the refactor.
The performance comparison must explicitly show the impact of disabling CUDA graphs for
dynamic dense SP. Ruff and markdownlint have not been run yet.

### Related Work and AI Assistance

The search for duplicate or related work has not yet been performed. Before publication,
add related issues/PRs, the search results, and the differences from existing work. This
draft does not claim that duplicate implementations have been ruled out.

AI assistance was used to implement the code, write regression tests, and draft this
description. The submitter must still review the changes line by line, complete the
relevant tests and model evaluations, and be able to explain and maintain the
implementation.
