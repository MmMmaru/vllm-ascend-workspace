# 重构/精简 maybe 算子 & 清理 attention need_gather_q_kv

来源：`AGENT_TODO.md`。本文档是与用户对齐后的实施计划（grill 十问结论 + 改动清单 + 验证方案）。

## 前提（已验证）

当前固定的 vllm 0.26 模型层在 decoder 内自行处理 MoE SP 的
gather/scatter，但首层 embedding 仍必须保持全量 token；SP 从首层 attention
输出之后才开始：

- `vllm/vllm/model_executor/models/deepseek_v2.py:1315-1323`：attention 后 `tensor_model_parallel_reduce_scatter` + `sequence_parallel_chunk(residual)`
- `vllm/vllm/model_executor/models/qwen3_next.py:500-548`：首层 attention 输入不是
  SP shard；attention 输出 reduce-scatter 后，residual 才做 `sequence_parallel_chunk`
- `vllm/vllm/model_executor/models/qwen3_next.py:573` `_all_gather_hidden_and_residual`：后续层 attention 前 gather
- vllm-ascend 的 VL 模型均走上游/ patched 上游模型代码（用户确认）

因此 attention 链路里的 `need_gather_q_kv` 整条参数链是冗余的。

## 决策记录（用户已拍板）

1. **完全删除 `need_gather_q_kv`**：从 `ops/mla.py` → `mla_forward` custom op → `mla_v1.py`/`sfa_v1.py`/`dsa_v1.py`/`dsa_cp.py` 等 attention impl 整条链删除；`_mla_preprocess` 里的 gather 调用一并删除。
2. **VL first-layer 特判整套删除**：`ops/mla.py` 的 `is_vl_first_layer` init 检测（`is_vl_model`+`parse_layer_idx`）与 forward 的 output 缩容分支全部移除，output 恒为 `hidden_states.shape[0]`。
3. **设置直接从上游读取**：内联 `get_current_vllm_config().parallel_config.use_sequence_parallel_moe`，不带 try/except；若编译期/fake 阶段出现无 config 上下文的 AssertionError，再补 fallback False（见风险 R1）。
4. **`maybe_all_gather_and_maybe_unpad` 做成 EP-only**：删 `label`、删纯 SP 分支（`register_custom_ops.py:87-88`）、删 `is_ep_comm`；非 EP 调用处迁移为显式 gather（见改动清单 8）。
5. **pad 逻辑保留**：`all_gather` 要求各 rank 输入等长，`_pad_to_ep_local_size` 必要；把 `# TODO1：pad逻辑为什么在这里？` 改成说明性注释。
6. **`maybe_pad_and_reduce` 做成 EP-only**：删 `is_ep_comm`；非 EP 调用处按各自
   数据布局显式选择 collective，不能把 `use_sequence_parallel_moe` 当成通用的
   reduce-scatter 开关。EP 路径上的 VL-draft 特判（`:111-112`）保留在算子内。
   两个算子都加注释标注"仅用于 EP 通信场景"。
7. **fake impl 同口径精简**：只保留 EP 形状推算（`local_sizes`→`sum`/`local_sizes[rank]`，否则按 `ep_world_size` 乘/除）；按决策 3 去掉 fake 里的 try/except。
8. **代码一次改完，测试文件同步改签名对齐**，标注"未运行"（环境不 ok）。
9. **删除 `maybe_all_reduce_tensor_model_parallel`**：op 注册 + impl 全删，`fused_moe.py:165,176` 两处调用点内联等价逻辑（MoECommType 判断 + 上游 SP 配置直读）。`deepseek_v4.py:509` 调的是上游 FusedMoE 同名方法，不动。
10. **验证按 `benchmark-delete-flashcomm.md`** 执行（见下）。

## 改动清单

### 1. `vllm_ascend/ops/register_custom_ops.py`（核心）

- 删 `_sequence_parallel_enabled()`（:21-25）。
- `_maybe_all_gather_and_maybe_unpad_impl`（:54）：
  - 签名改为 `(x: torch.Tensor) -> torch.Tensor`。
  - 只保留 EP 分支：`local_sizes` 存在时先 `_pad_to_ep_local_size` 再 `ep_group.all_gather` + 按 `local_sizes` 截回；否则按 `num_tokens_across_dp_cpu`/`_EXTRA_CTX.padded_length` unpad。
  - 删 `label` 判断、纯 SP `elif` 分支、`is_ep_comm` 参数。
  - `:69` 的 TODO1 疑问改为说明："all_gather 要求各 rank 输入等长，先 pad 到 max_local_size，gather 后按 local_sizes 截回"。
  - docstring 标注：仅用于 EP 通信场景（EP all_gather + DP unpad）。
- `_maybe_pad_and_reduce_impl`（:93）：
  - 签名改为 `(x: torch.Tensor) -> torch.Tensor`。
  - 删 `try/except AssertionError`（决策 3）、删非 EP 分支（:102-110）。
  - 保留 VL-draft 特判（`:111-112` → all_reduce）与 EP+DP 两条 unpad/pad 路径。
  - docstring 标注：仅用于 EP 通信场景。
- 两个 fake impl（:145, :167）：同口径精简（决策 7）。
- 删 `_maybe_all_reduce_tensor_model_parallel_impl`（:191-199）及其注册（约 :257）。
- 更新两个 maybe 算子的 `direct_register_custom_op` 注册（schema 随签名变化）。
- 清理因此不再使用的 import（注意 `is_vl_model` 仍被 VL-draft 特判使用，保留）。

### 2. `vllm_ascend/ops/mla.py`

- `forward`（:152-173）：删 `need_gather_q_kv` 计算与 VL first-layer 分支，output 恒为全量形状；调用改为 `torch.ops.vllm.mla_forward(hidden_states, output, self.prefix)`。
- 删 init 里的 VL 检测（:138-145，`is_vl_first_layer`），清理不再使用的 `is_vl_model`/`parse_layer_idx` import。
- `mla_forward`/`mla_forward_fake`：删 `need_gather_q_kv` 参数。

### 3. `vllm_ascend/attention/mla_v1.py`

- `_mla_preprocess`（:1626）：删 `need_gather_q_kv` 参数；删 gather 调用（:1651-1652）及 step-2 注释。
- 各 forward 变体（:1676-1740）：删 `need_gather_q_kv` 参数与 `:1727` 的 gather 调用。

### 4-7. attention 其余文件（同一规则）

- `vllm_ascend/attention/sfa_v1.py`（:1867, :1902 区域）
- `vllm_ascend/attention/dsa_v1.py`（:1786）
- `vllm_ascend/attention/context_parallel/dsa_cp.py`（:1423）
- `vllm_ascend/ops/dsa.py`、`vllm_ascend/attention/sfa_kv_offload.py` 中的 need_gather 引用

规则：删 `need_gather_q_kv` 参数链与所有 `maybe_all_gather_and_maybe_unpad(..., need_gather_q_kv)` 调用。

### 8. 非 EP gather 调用处迁移

统一规则：**编译图/可能被 trace 的代码用上游 custom op** `torch.ops.vllm.all_gather(x, 0, get_tp_group().world_size, get_tp_group().unique_name)`；确定不在编译图内的可用 `tensor_model_parallel_all_gather(x, 0)`。保留各调用点原有守卫条件。

- `vllm_ascend/ops/rotary_embedding.py:247`（positions gather）
- `vllm_ascend/spec_decode/llm_base_proposer.py:2048,2053`（mtp+shared_expert_dp 分支内）
- `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py` 约 12 处 pattern：`maybe_all_gather_and_maybe_unpad(t, True)` → `torch.ops.vllm.all_gather(t, 0, tp_world_size, tp_group_name)`（tp 组信息在 pattern 定义时获取，见风险 R2）

### 9. 非 EP reduce 调用处迁移（决策 6）

- `vllm_ascend/ops/vocab_parallel_embedding.py:234-262`：普通 TP embedding 始终
  all-reduce，保持首层 attention 的完整 token 序列；专用 embedding-TP 路径继续
  维护自己的 gather/scatter 协议。
- `vllm_ascend/models/minimax_m3/minimax_m3_vl.py:86`：同上（保留外层 `tp_world_size > 1` 守卫）。
- 迁移时带上非 EP 侧的 VL-draft 判断（原 `:102-103` 语义）。

### 10. EP 调用点去实参

- `vllm_ascend/ops/fused_moe/prepare_finalize.py:394-400`：去掉 `True, True`。
- `vllm_ascend/ops/fused_moe/prepare_finalize.py:529`：去掉 `True`。

### 11. `vllm_ascend/ops/fused_moe/fused_moe.py`（决策 9）

- `:165`（0.26.0 分支）：现有 `if not self.moe_config.is_sequence_parallel` 守卫保留，内部改为内联 comm-type 判断 + `tensor_model_parallel_all_reduce`。
- `:176`（else 分支）：内联完整条件：`_EXTRA_CTX.moe_comm_type not in {ALLTOALL, MC2, FUSED_MC2} and not get_current_vllm_config().parallel_config.use_sequence_parallel_moe` 时才 `tensor_model_parallel_all_reduce(states)`。
- 补齐 import。

### 12. 测试同步（只改签名对齐，标注未运行）

- `tests/ut/ops/test_register_custom_ops.py`：适配 EP-only 签名；删 `maybe_all_reduce_tensor_model_parallel` 相关用例。
- `tests/ut/attention/a2/test_mla_v1.py`：删 need_gather 相关传参/断言。
- `tests/ut/ops/test_fused_moe.py`：适配。
- `tests/ut/spec_decode/a2/test_eagle_proposer.py`：确认 mock 点不受影响，受影响则适配。
- `tests/e2e/pull_request/one_card/compile/test_norm_quant_fusion.py`、`test_graphex_norm_quant_fusion.py`：op 引用改为 `all_gather` 等价形式。

## 验证方案（按 `benchmark-delete-flashcomm.md`）

### 对话功能验证（2026-08-12 已在 80.5.17.111 执行，DP2 TP2 EP + enforce-eager）

> 更正：Qwen3.5 的所谓 main 对照运行已证伪。运行时实际 checkout 的仍是
> `delete-flashcomm`，因此该次结果不能作为 main 基线；需要在确认 commit 和运行时
> 源码路径后重新执行 main 对照。

| setup | 模型 | PR（本重构） | main 对照 | 结论 |
|---|---|---|---|---|
| 1 | Qwen3-30B-A3B | ✅ 对话正常 | — | 通过 |
| 2 | Qwen3.5-35B-A3B | ✅ 默认 SP + MTP3 输出恢复正常 | ⚠️ 未有效执行（误用了 `delete-flashcomm`） | `delete-flashcomm` 的两处 SP 协议断点已修复；不依赖关闭平台默认 SP |
| 3 | DeepSeek-V4-Flash-w4a8 | ❌ 启动报 `input_ids.numel()=32 != rows=16`（hash router + SP + DP2 的 token 对齐问题，路径 `fused_topk_router.py:106-112` × `deepseek_v4.py:469-470`，均非本重构触碰文件） | ❌ 同样报错 | 存量问题，无回归 |

过程中实锤并修复 R1（见风险节）；另发现容器内 editable install 指向 `/home/x50063850/vllm-ascend-workspace`，服务必须带上历史 serving 状态的 PYTHONPATH（指到 `/vllm-workspace`）+ ATB 环境块才能跑在同步代码上。

### Qwen3.5 默认 SP 根因与修复验证（2026-08-12）

目标配置保持上游默认 `allgather_reducescatter`，没有恢复平台层的 SP-off
override。确认的两个下游协议断点为：

1. 普通 TP vocab embedding 根据 `use_sequence_parallel_moe` 提前执行
   reduce-scatter，使首层 attention 收到半段 token；修复为普通路径保持 TP
   all-reduce，SP 在首层 attention 输出之后开始。
2. shared expert 在 ALLTOALL/MC2/FUSED_MC2 下无条件执行 TP all-reduce；SP 时各
   TP rank 使用完整复制权重处理不同 token shard，该 all-reduce 会把无关 token
   按位置相加。修复为 SP 跳过该归约，非 SP 保持原行为。

直接在 `delete-flashcomm` 分支同步纯代码验证（无 worktree、无重新编译）：

- 定向 UT：`test_vocab_parallel_embedding.py` + `test_fused_moe.py`，42 passed。
- Ruff：4 个本次修改文件全部通过。
- E2E：Qwen3.5-35B-A3B，DP2/TP2/EP，默认
  `allgather_reducescatter`；分别验证 `enforce-eager + MTP3`，以及仓库四卡
  E2E 对齐的 `FULL_DECODE_ONLY + MTP3`（capture sizes 4/8/12/16）。
- 四组短请求均恢复语义输出（`2 + 2 = 4`、France → Paris 等），不再出现
  `!!!` 或损坏的重复 Thinking。
- 406-token 长 prefill 输出 `4`，覆盖长 prefill 的 ALLTOALL 与后续
  MC2/speculative decode。

### Benchmark（环境具备后执行）

每个 setup 做：对话正常性检查 + 下列 benchmark（PR vs main 对比）：

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8010 \
  --endpoint /v1/completions \
  --served-model-name qwen \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 2048 \
  --num-prompts 100 \
  --num-warmups 10 \
  --max-concurrency 1 \
  --metric-percentiles 50,90,99 \
  --seed 0
```

| setup | 模型 | 并行 |
|---|---|---|
| 1 | Qwen3-30B-A3B | DP2 TP2 EP |
| 2 | Qwen3.5-35B-A3B | DP2 TP2 EP |
| 3 | DeepSeek-V4-Flash-w4a8 | DP2 TP2 EP |

提交前按 AGENTS.md 跑 pre-commit；commit 需要 sign off。

## 风险

- **R1（已实锤并修复）**：dummy run（`_initialize_kv_caches` → profile forward）阶段 worker 无 config 上下文。普通 vocab embedding 不再读取该 config，并恢复上游 TP all-reduce 语义；其余仍需读取 config 的内联点保留 try/except fallback False。
- **R2**（已静态排除）：`norm_quant_fusion_pass` 的 6 个 SP pattern 类在 `__init__` 捕获 `get_tp_group().world_size/unique_name`；已核实 pass 注册在分布式初始化之后（上游 `vllm/vllm/compilation/backends.py:953` 编译期调用；e2e 测试先 `ensure_model_parallel_initialized`）。
- **R3**：本次根因相关的两个定向 UT 文件已在远端运行（42 passed）；其余本计划涉及的 attention/fusion 全量测试仍待执行。
- **R4**：proposer/rotary/vocab embedding 迁移点位于编译图内，已统一用 `torch.ops.vllm.all_gather/reduce_scatter/all_reduce`（custom op，无 graph break）；minimax_m3 fallback 不在图内，用 python 函数。
- **R5（重点验证）**：`llm_base_proposer.py` 迁移后，守卫 `enable_shared_expert_dp`（= `ascend_config.enable_shared_expert_dp or use_sequence_parallel_moe`）比旧 op 内部的 `_sequence_parallel_enabled()` 更宽——`enable_shared_expert_dp=True 且 use_sequence_parallel_moe=False` 时会从 identity 变为真正执行 TP all_gather。环境恢复后需重点验证该配置组合。
- **R6**：fusion pass pattern 的搜索/替换两侧都从 `maybe_all_gather_and_maybe_unpad` 换成 `torch.ops.vllm.all_gather`，命中率依赖图中 gather 的实际发射形式（上游模型层 SP gather 应下沉为该 op），需环境恢复后确认 pass 命中率无回退。
- **R7**：`vocab_parallel_embedding.py` 新增 `tp_group.world_size == 1` 提前返回（旧路径经 GroupCoordinator 自带 bypass，上游 custom op 无 bypass）；rotary/proposer 两处未加该守卫，TP=1+EP 时会多一次 1-rank all_gather（语义等价，仅多一次通信）。
