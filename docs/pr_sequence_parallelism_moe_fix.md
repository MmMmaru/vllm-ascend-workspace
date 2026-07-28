# [BugFix][Core] Fix Sequence Parallelism for MoE and Dense Models

## What this PR does / why we need it?

本 PR 修复 Ascend Sequence Parallelism（SP）编译 pass 与 MoE/EP 通信路径之间的形状和分支不一致问题，覆盖 dense 模型和 MoE 模型。

主要问题有三类：

1. MoE 输出经过 `maybe_all_reduce_tensor_model_parallel` 和 `aten.alias` 后，实际 FX 图不再匹配原有的 `all_reduce -> RMSNorm` 搜索模板，导致 SP pass 无法改写该路径。
2. `enable_sp_by_pass` 只表示启用了编译期 SP pass，并不等价于 `moe_config.is_sequence_parallel`。原实现把两者直接合并判断，可能使 MoE 错误进入 EP sequence-parallel 路径，在路由前重复 all-gather token。
3. SP RMSNorm 之后 residual 可能仍是 TP local sequence，而 MoE 通信路径可能已经恢复了完整 sequence。原 `maybe_chunk_residual` 只支持“完整 residual -> local residual”的方向，不能处理反向的形状恢复，最终可能在后续 residual add 处产生 shape mismatch。

本 PR 通过补充实际图形态的 pattern、区分 pass-level SP 与 MoE-level SP、完善 residual shape 对齐逻辑，保证编译后的通信顺序与运行时 tensor shape 一致。

## 详细修改说明

### 1. 扩展 Sequence Parallelism pass 的 MoE pattern 匹配

文件：`vllm_ascend/compilation/passes/sequence_parallelism.py`

#### 1.1 支持显式配置 SP 最小 token 数

`get_sp_min_token_num()` 现在优先读取已有的 `config.compilation_config.pass_config.sp_min_token_num`：

```python
configured = config.compilation_config.pass_config.sp_min_token_num
if configured is not None:
    return configured
```

未显式配置时保持原有默认策略：

- MoE 模型：默认 `1`，允许较小 token 数触发 pass。
- Dense 模型：默认 `1000`，避免小 batch/短 sequence 下引入 SP 通信开销。

影响位置：`sequence_parallelism.py:25-33`。

#### 1.2 增加 `maybe_all_reduce` 搜索模板

新增 `_maybe_all_reduce()` 和 `_maybe_all_reduce_search_pattern()`，用于匹配 MoE 输出后的真实图结构：

```text
maybe_all_reduce_tensor_model_parallel
        -> aten.alias
        -> maybe_chunk_residual
        -> npu_add_rms_norm_bias
```

搜索模板显式保留 `aten.alias`，并通过 `_users=2` 约束 alias 同时连接 residual 路径和 RMSNorm 输入路径。这样可以匹配 Dynamo/Inductor 生成的 alias 节点，而不是只匹配没有 wrapper/alias 的普通 all-reduce 图。

影响位置：`sequence_parallelism.py:59-98`。

模板同时支持：

- middle layer 返回 `(output, residual)`；
- last layer 只返回 `output`；
- Qwen3-VL 的 `alias + deepstack_input_embeds` 输入；
- `eps=1e-5` 和 `eps=1e-6` 两种 `npu_add_rms_norm_bias` 参数形式；
- RMSNorm 的 `getitem(..., 0)` 主输出和 `getitem(..., 2)` residual 输出。

#### 1.3 统一 RMSNorm 调用参数个数

新增 `_add_rms_norm_bias()`，根据 epsilon 选择 Dynamo 实际生成的调用形式：

- `eps == 1e-6`：调用三参数形式；
- 其他 epsilon：补充 `None` 和 epsilon 参数。

这避免 pattern example 使用固定五参数形式，而实际图在默认 epsilon 下使用三参数形式时匹配失败。

影响位置：`sequence_parallelism.py:106-115`。

#### 1.4 在所有相关 pattern 中补充 residual 对齐

原 pattern 直接把 `x/residual` 传给 RMSNorm。现在在 RMSNorm 前统一调用：

```python
residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
```

该逻辑应用于：

- `MiddleAllReduceRMSNormPattern`；
- `LastAllReduceRMSNormPattern`；
- `Qwen3VLMiddleAllReduceRMSNormPattern`；
- 新增的 `maybe_all_reduce` 搜索/替换 pattern。

影响位置：`sequence_parallelism.py:140-322`。

#### 1.5 注册两类搜索 pattern

每个 all-reduce/RMSNorm pattern 现在保留两套搜索入口：

```text
普通路径：all_reduce -> RMSNorm
MoE 路径：maybe_all_reduce -> alias -> maybe_chunk_residual -> RMSNorm
```

两套 pattern 使用相同的 replacement：

```text
all_reduce
    -> reduce_scatter
    -> local RMSNorm
    -> all_gather(output)
```

对于中间层，residual 继续作为第二个输出返回；对于最后一层，只返回主 output。

影响位置：`sequence_parallelism.py:165-186`、`227-248`、`298-322`。

#### 1.6 调整编译日志

将 pass 执行前日志明确为 `before apply replacement`，与执行后的 graph 日志形成对应，便于区分 noop cleanup 前后以及 pattern replacement 前后的图形态。

影响位置：`sequence_parallelism.py:347-353`。

### 2. 复用统一 RMSNorm helper 并增强 MoE pass 日志

文件：`vllm_ascend/compilation/passes/sequence_parallelism_moe.py`

#### 2.1 统一调用 `_add_rms_norm_bias`

原有 all-gather epilogue pattern 直接调用固定参数形式的 `npu_add_rms_norm_bias`。现在 middle、last 和 Qwen3-VL 三类 pattern 都复用基类 helper，从而统一处理 epsilon 和算子参数个数。

影响位置：`sequence_parallelism_moe.py:32-142`。

#### 2.2 增强调试日志可读性

pass 执行前后使用带换行的 graph 日志，并输出替换数量，便于直接对比：

```text
SequenceParallelismMoePass before apply replacement
<graph>
SequenceParallelismMoePass after apply replacement
<graph>
SequenceParallelismMoePass replaced <N> patterns
```

影响位置：`sequence_parallelism_moe.py:187-199`。

### 3. 区分编译期 SP 与 MoE sequence-parallel 通信路径

文件：`vllm_ascend/ops/fused_moe/prepare_finalize.py`

新增 `_use_ep_sequence_parallel()`：

```python
return enable_sp() or (enable_sp_by_pass() and self.moe_config.is_sequence_parallel)
```

这里区分两个概念：

- `enable_sp()`：运行时 FlashComm/sequence-parallel 通信已经启用，MoE 使用 EP sequence-parallel 路径。
- `enable_sp_by_pass()`：编译 pass 已启用，只说明 RMSNorm 周围可能被改写为 TP reduce-scatter/all-gather；它不代表 MoE 配置已经把 token ownership 设置为 sequence parallel。
- `self.moe_config.is_sequence_parallel`：MoE 本身的 `sp_size > 1` 状态，决定 MoE 是否拥有 sequence-parallel token 分片。

prepare 和 finalize 都改为使用这个统一判断：

```text
_use_ep_sequence_parallel() == True
    prepare  -> _prepare_with_ep_group()
    finalize -> _finalize_with_ep_group()

_use_ep_sequence_parallel() == False
    prepare  -> _prepare_with_dp_group()
    finalize -> _finalize_with_dp_group()
```

影响位置：`prepare_finalize.py:335-364`、`prepare_finalize.py:489-505`。

EP sequence-parallel 路径仍然执行：

```text
TP AG -> Attention -> TP RS -> EP AG -> MoE -> EP RS
```

本次修改的重点是避免仅因为启用了 SP pass，就把 `moe_config.is_sequence_parallel == False` 的 MoE 错误切换到 EP 路径，从而避免路由前重复收集 token。

### 4. 修复 residual 的双向 shape 对齐

文件：`vllm_ascend/ops/register_custom_ops.py`

`_maybe_chunk_residual_impl()` 现在根据 residual 和当前输入 `x` 的 sequence 维大小选择方向：

```text
residual.size(0) == x.size(0)
    保持不变

residual.size(0) < x.size(0)
    TP all-gather residual，恢复完整 sequence

residual.size(0) > x.size(0)
    按 pad_size 补齐后，按 TP rank chunk 成 local sequence
```

其中新增的第一条分支解决以下场景：

```text
前一个 SP RMSNorm
    residual 是 TP local sequence
        -> MoE communication path 恢复 full sequence
            -> 下一个 residual add 前需要 all-gather residual
```

原实现只处理 full residual 到 local residual 的切分，无法处理 local residual 到 full residual 的恢复。

影响位置：`register_custom_ops.py:22-42`。

### 5. 修正 `maybe_pad_and_reduce` fake 实现的 shape 推导

`_maybe_pad_and_reduce_fake()` 现在只在以下场景推导为 TP 分片 shape：

```python
_EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
```

也就是说：

- FlashComm 或 EP communication + SP pass：fake shape 按 reduce-scatter 后的 token 数推导；
- 普通 TP communication：保持 all-reduce 语义，不错误地把 shape 除以 TP size。

这使 fake tensor 的编译期 shape 推导与 `_maybe_pad_and_reduce_impl()` 的运行时分支保持一致，避免编译图和实际执行路径出现 shape 不一致。

影响位置：`register_custom_ops.py:122-131`。

## 变更前后数据流

### 普通 SP replacement

```text
Before:
input -> TP AllReduce -> RMSNorm -> output

After:
input -> TP ReduceScatter -> local RMSNorm -> TP AllGather -> output
                       └-> residual shape alignment
```

### MoE 输出后的 SP replacement

```text
Before graph:
MoE output -> maybe_all_reduce -> alias -> RMSNorm

After graph:
MoE output -> ReduceScatter -> local RMSNorm -> AllGather
                         └-> maybe_chunk_residual
```

### EP sequence-parallel 分支选择

```text
enable_sp()
    └─ true ------------------------------------┐
                                               ├─ EP prepare/finalize path
enable_sp_by_pass() and moe_config.is_sequence_parallel
    └─ true ------------------------------------┘

仅 enable_sp_by_pass()，但 moe_config.is_sequence_parallel == false
    └─ DP prepare/finalize path，避免 MoE 路由前重复 EP all-gather
```

## 影响范围

### 正向影响

- MoE 模型的 `maybe_all_reduce + alias + RMSNorm` 图可以被 SP pass 正确匹配和替换。
- Dense 模型继续使用原有 all-reduce pattern，并获得可配置的 SP token threshold。
- Qwen3-VL deepstack 输入和 middle/last layer 的不同输出形式继续被覆盖。
- residual 在 local sequence 和 full sequence 之间切换时保持 shape 一致，降低后续 residual add 出错的风险。
- fake shape propagation 与真实 EP/TP communication 路径一致，减少编译期 shape 错误。
- 仅开启 SP pass 而没有启用 MoE sequence parallel 时，不再强制走 EP sequence-parallel prepare/finalize，避免多余 token all-gather。

### 潜在影响和兼容性

- 普通非 SP 路径仍使用原有 DP prepare/finalize 和 TP all-reduce 逻辑。
- `enable_sp()` 为 true 的运行时 EP/SP 路径保持不变，只复用了统一判断函数。
- `pass_config.sp_min_token_num` 现在会被 Ascend SP pass 直接采用；这是已有配置字段行为生效，不新增配置项。
- SP/EP 场景的通信拓扑和 tensor shape 发生了预期修正，可能改变通信数量和性能表现，需在真实 NPU 上进行吞吐、TTFT 和通信计数验证。
- 本 PR 未修改 `vllm` 子仓库，也未新增公共 Python API。

## 文件变更清单

| 文件 | 变更内容 |
| --- | --- |
| `vllm_ascend/compilation/passes/sequence_parallelism.py` | 增加 MoE wrapper/alias 搜索 pattern、统一 RMSNorm 调用、补充 residual 对齐、支持 token threshold 配置 |
| `vllm_ascend/compilation/passes/sequence_parallelism_moe.py` | 复用 RMSNorm helper，增强 pass 前后 graph 日志 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | 区分 pass-level SP 和 MoE-level SP，统一 EP path 判断 |
| `vllm_ascend/ops/register_custom_ops.py` | 修复 residual 双向 shape 对齐，修正 fake reduce shape 推导 |

## Does this PR introduce any user-facing change?

有间接行为变化，但没有新增命令行参数或公共 API：

- 已有的 `pass_config.sp_min_token_num` 配置现在会影响 Ascend SP pass 的适用范围。
- 启用 SP pass、但 MoE 自身未启用 sequence parallel 时，MoE 不再错误进入 EP sequence-parallel 通信路径。
- 相关模型的编译图和通信路径会发生修正，目标是消除 shape mismatch、错误路由通信和冗余 all-gather。

## How was this patch tested?

本次文档生成时未在当前会话执行 NPU 测试，因此不能宣称 E2E 或性能验证已经完成。合入前建议在匹配的 Ascend/CANN/torch_npu 容器中至少执行以下验证：

### 1. 静态检查

```bash
cd vllm-ascend
bash format.sh
```

检查四个变更文件的格式、import 和 pre-commit 规则。

### 2. 编译 pass 图验证

验证以下日志和结果：

- `npu_sequence_parallelism_pass` 的 `before apply replacement` 与 `after apply replacement` 图中，MoE wrapper 路径被替换为 `reduce_scatter -> RMSNorm -> all_gather`。
- `npu_sequence_parallelism_allgather_ep_pass` 的 replacement 数量和 graph 结构符合预期。
- 同时检查 `enable_sp_by_pass=True`、`moe_config.is_sequence_parallel=False` 时没有进入 `_prepare_with_ep_group()`。
- 检查 residual 的 local/full sequence shape 在下一次 RMSNorm 或 residual add 前一致。

### 3. 功能和 E2E 验证

建议覆盖：

- dense 模型：SP pass 开启和关闭；
- MoE 模型：SP pass 开启、MoE sequence parallel 开启/关闭；
- Qwen3-VL：带 `deepstack_input_embeds` 的 middle layer；
- middle layer 和 last layer；
- `eps=1e-5` 与 `eps=1e-6`；
- TP2/EP 场景下的 prefill 和 decode；
- 非 SP 配置回归测试。

每个 case 至少验证模型能够完成 warmup 和一轮实际生成，并检查输出 shape、无 rank hang、无 residual add shape mismatch。

### 4. 性能和通信验证

在同一硬件、同一模型和同一请求集下对比 SP pass 开启前后：

- TTFT、ITL、吞吐；
- TP all-reduce、reduce-scatter、all-gather 和 EP communication 次数；
- 是否仍存在重复的 EP all-gather；
- 编译日志中的 replacement 数量和实际 profiler 中的通信数量是否一致。

## Review focus

审阅时重点关注以下边界：

1. `enable_sp()` 与 `enable_sp_by_pass()` 的语义是否保持独立。
2. `moe_config.is_sequence_parallel` 是否准确代表 MoE token ownership，而不是仅代表 pass 已启用。
3. `_maybe_chunk_residual_impl()` 的 all-gather 是否只在 `residual.size(0) < x.size(0)` 时触发，避免对正常 local shape 产生额外通信。
4. fake shape 与真实 `_maybe_pad_and_reduce_impl()` 的 EP/TP 分支是否始终一致。
5. 各 epsilon、Qwen3-VL deepstack、middle/last layer pattern 是否都能匹配实际 Dynamo graph。
