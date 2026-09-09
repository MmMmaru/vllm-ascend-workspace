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

# PR 草稿

## Title

`[Core] Refactor sequence parallel collectives into linear layers`

## Purpose

This PR proposes a sequence-parallel (SP) refactor.

Today, the all-gather before attention and the reduce-scatter after attention are owned by model-specific decoder code. This couples the SP boundary to individual model implementations, duplicates collective-selection logic, and makes it easy for related paths—such as LoRA, linear attention, and other model runners—to diverge from the main attention path. Meanwhile, SP is automatically enabled under certain conditions, including DP > 1, TP > 1, EP, and supported all-to-all backends. We believe this option should also be user-configurable.

This PR moves these communication boundaries into the tensor-parallel linear layers that naturally own them, since SP is coupled to the TP group:

- `ColumnParallelLinear` gathers a sequence shard before an attention input projection.
- `RowParallelLinear` performs a reduce-scatter on the attention output when the caller requests unreduced row-parallel results.

The refactor preserves the existing MoE SP behavior while making the communication contract reusable by attention implementations that use the same parallel linear layers.

This PR also considers SP for dense models. When explicitly enabled, SP is used for dense models when the number of tokens in a step exceeds a threshold.

## Implementation
This PR primarily targets the Qwen3.5 model family. Other models can be migrated similarly.

- Add sequence-parallel collective wrappers in `communication_op.py`. These wrappers use custom device-communicator collectives when available and otherwise fall back to the TP all-gather and reduce-scatter implementations.
- Add the `enable_sequence_parallel` configuration and the `--enable-sequence-parallel` CLI option. When unset, the existing automatic MoE heuristic is preserved; `False` disables SP, while `True` requires TP > 1, allows DP = 1, and additionally requires EP and a supported all-to-all backend for MoE models. Pipeline parallelism with PP > 1 disables SP.
- Enable the linear-layer SP path for Qwen3-Next and Qwen3.5 when `use_sequence_parallel` is enabled. Other models retain their current behavior.
- Extend the dense Qwen3.5 attention and MLP paths to pass `sequence_parallel` to their parallel linear layers, so manually enabled dense SP uses the same token all-gather and reduce-scatter boundaries as the MoE path.
- Select dense SP dynamically based on the unpadded number of tokens in each step: steps with more than 1000 tokens use SP, while shorter steps retain the regular TP path. A forward-context flag derived from this unpadded token count keeps model-level chunk/gather operations and linear-layer collectives on the same path.
- Remove the duplicated attention all-gather and reduce-scatter logic from the Qwen3-Next decoder layer. The decoder remains responsible for sharding the residual when transitioning to the MoE path.
- Route the Qwen Gated DeltaNet linear-attention input through the same `ColumnParallelLinear` preparation path.
- Reuse the base linear-layer input-preparation and output-reduction logic in the LoRA wrappers so that LoRA follows the same SP boundary.
- Make the V1 GPU model runner pad SP-MoE and dense-SP batches to a token count that is divisible by the TP world size and align CUDA-graph capture sizes with this padded token count. Model-level SP helpers can therefore require padding rather than introducing it independently.

## Scope
| Configuration | MoE | Dense |
| --- | --- | --- |
| `None` (default) | Preserve the existing automatic enablement conditions: DP > 1, TP > 1, EP is enabled, and a supported all-to-all backend is selected | Not enabled automatically |
| `False` | Disabled | Disabled |
| `True` | Requires TP > 1, EP, and a supported all-to-all backend; DP = 1 is allowed | Requires TP > 1; use SP when the number of tokens in a step is > 1000, otherwise disable it |

## Discussion points

- Are `ColumnParallelLinear` and `RowParallelLinear` the preferred shared boundary for model SP communication, or should this be represented by a more explicit attention-level abstraction?
- Is limiting the initial rollout to specific model types acceptable, with follow-up migrations for other SP-capable models?
- Does the runner-owned padding contract provide the right separation between scheduling and model execution?
- Is it reasonable to add `--enable-sequence-parallel` as a user-facing CLI option?

## Test Plan
The final PR will include targeted unit tests and TP > 1 serving-correctness results for Qwen3-Next and Qwen3.5-MoE with `--enable-sequence-parallel`.

Tested on four H20 GPUs using:
```bash
vllm serve \
  --model Qwen/Qwen3.5-35B-A3B \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --enable-sequence-parallel \
  --enable-expert-parallel \
  --enforce-eager \
  --all2all-backend allgather_reducescatter
```
Vllm bench test is conducted under `concurrency=1` and different input sequence length
## Test Results

| Stage  | Sequence length | Mean TTFT (ms) |
| --- | ---: | ---: |
| SP off | 8k | 267.40 |
| SP on | 8k | 254.02 |

## 09-08 SP-on 对话冒烟补充

DP1/TP4/EP、V1 eager 下完成 10 次 chat 请求：9 次关闭 thinking 的请求正常 stop，
覆盖中英文、算术、多轮记忆、5049-token 文本提取及 3 路并发，未观察到乱码或异常重复。
多轮活动推荐未严格满足避开爬山的约束；默认模板输出推理文本，在 512 token 截断，
尚未得到最终回答。未做 SP-off 同提示精度对照。详见性能记录的对话冒烟章节。

## Dense SP CUDA graph 支持（更新）

已删除 dense SP 强制将 cudagraph_mode 设为 NONE、清空 capture sizes 的代码。
此前的风险是 1000-token TP 步被补到 1024-token SP 捕获桶，及无 guards 编译缓存
复用首次追踪的分支。现在捕获尺寸在 1000 处分界，SP 捕获尺寸必须被 TP size 整除；
模型为 SP / TP 分别建立编译实例及 AOT 缓存，共享原有参数，重置编译时清理两份实例。

H20 TP4、Qwen3.5-4B BF16 验证：

- 3 项阈值 / 编译分支回归通过，覆盖两种首次编译顺序。
- 交替输入 1000、1001、17、1001、1000 tokens，每次生成 4 tokens。
- FULL 配置经 Qwen3.5 后端调整为 decode-only FULL，实际捕获 decode 图。
- PIECEWISE 实际捕获 1、1000、1004、1024 四个桶，SP/TP 分别使用 `.sp_1` / `.sp_0` 缓存。
- FULL 和 PIECEWISE 的生成 token 均与各自无图基线一致。PIECEWISE 与开启编译的
  无图基线相比，所选 token 的 logprob 最大差为 0.0148；top-5 集合并非完全一致。
- 现有根 conftest 缺少 tblib，聚焦单测通过 `--confcutdir=tests/compile` 运行；
  模型 E2E 使用独立驱动直接调用 LLM，新增仓库 E2E 用例尚未通过原 fixture 执行。
- 原始输出及捕获日志：`.log/spbench-results/dense-graph/`。

此前“动态 dense SP 暂关闭 CUDA graph”的实现状态由本节取代。
本轮未验证 MTP、LoRA 和其他模型家族的图执行，也未重新测量性能。
