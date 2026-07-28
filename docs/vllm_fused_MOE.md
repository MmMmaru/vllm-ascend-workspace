从当前工作区上游 vLLM（`vllm@7b3d595eb`）源码看，MoE 的核心数据流是：

**输入 hidden states → Router 计算 Top-K expert → 按并行模式准备/通信 → 本地 expert GEMM → 加权合并/通信 → 输出 hidden states。**

其中，TP、DP、EP 决定“权重和 token 如何分布”，SP 主要决定“token 是否沿 TP 维切分”。`enable_expert_parallel=True` 后，MoE 内部会把原本的 TP/DP/PCP 设备折叠为 EP group：MoE 专家按 rank 完整持有，MoE 内部 `tp_size` 变为 1；但 Transformer 的 Attention、Dense Linear 仍然可以使用 TP。

**目的（Why）**：减少 MoE 权重显存、避免每个 rank 保存全部专家，并让 token 被送到真正拥有目标 expert 的 rank。**输入（Input）**是 `[num_tokens, hidden_size]`、router logits、`topk_ids/topk_weights`；**处理（Logic）**由 `MoERunner` 调用 `prepare/dispatch → expert kernel → finalize/combine`；**输出（Output）**恢复为 `[num_tokens, hidden_size]`，继续进入下一层 Transformer。**流向（Flow）**会根据是否启用 DP、EP、SP 选择 AllReduce、AllGather/ReduceScatter 或真正的 All-to-All。

### 1. 并行配置如何归一化

| 模式 | MoE 内部配置 | 专家权重布局 | 主要通信 |
|---|---|---|---|
| TP，关闭 EP | `moe_tp = TP, moe_ep = 1` | 每个 rank 保存全部专家，但 `w13/w2` 沿 intermediate 维切分 | 最后 TP AllReduce |
| DP，关闭 EP | `moe_tp = DP × TP, moe_ep = 1` | 专家权重沿 `DP × TP` 切分 | DP AllGather + ReduceScatter |
| EP，DP=1 | `moe_tp = 1, moe_ep = TP` | 每个 rank 保存一部分完整专家 | 通常本地 expert 计算后 TP AllReduce；具体取决于 expert kernel |
| DP+EP | `moe_tp = 1, moe_ep = DP × TP` | 每个 rank 保存完整的 local experts | EP group 上的 All-to-All 或 AG/RS |
| DP+EP+SP | `moe_tp = 1, moe_ep = DP × TP, moe_sp = TP` | 专家按 EP 分布，token 先按 TP 切分 | SP token chunk + EP dispatch/combine + TP AllGather |

EP 的配置转换在 `FusedMoEParallelConfig.make()` 中完成：关闭 EP 时，把 DP/PCP/TP 展平成 MoE 的 `tp_size`；开启 EP 时，则把这个展开后的规模改成 `ep_size`，并令 MoE 内部 `tp_size=1`。

### 2. TP：只做 Tensor Parallel

以 TP=2、关闭 EP 为例：

```text
完整 hidden_states
        |
        v
每个 TP rank 都拥有相同 token
        |
        v
Router 在每个 rank 独立计算 Top-K
        |
        v
每个 rank 执行本地 expert 的 TP 权重分片
w13: [E, 2I/TP, H]
w2 : [E, H, I/TP]
        |
        v
各 rank 得到部分 expert output
        |
        v
TP AllReduce
        |
        v
完整 MoE output
```

这种模式不需要 MoE token dispatch，因为每个 rank 都有所有 expert，只是 expert 的中间维度被切开。`MoERunner._maybe_reduce_final_output()` 会在 `tp_size > 1` 时执行 TP AllReduce。

### 3. DP：关闭 EP 时的行为

DP 并不是简单地让每个 rank 完全复制一份 MoE 权重。当前源码会把 MoE 的 TP 规模展平成 `DP × TP`。

例如 DP=2、TP=1：

```text
DP rank 0: 本地 token A
DP rank 1: 本地 token B

        |
        v

DP AllGather
        |
        v

每个 rank 都得到 A+B
但每个 rank 只拥有 expert 权重的一部分
        |
        v

本地 expert 计算
        |
        v

DP ReduceScatter
        |
        v

rank 0 得到 A 的结果
rank 1 得到 B 的结果
```

默认路径由 `MoEPrepareAndFinalizeNaiveDPEPModular` 实现：`prepare()` 通过 `get_ep_group().dispatch()` 做 token、Top-K ID 和权重的收集，`finalize()` 通过 `combine()` 做输出合并。底层 `AgRsAll2AllManager` 实际使用 AllGatherV 和 ReduceScatterV。

### 4. EP：专家按 rank 分片

开启 `enable_expert_parallel` 后，例如 TP=4、DP=1：

```text
rank 0: expert 0 ~ E/4
rank 1: expert E/4 ~ E/2
rank 2: expert E/2 ~ 3E/4
rank 3: expert 3E/4 ~ E
```

Router 仍然在各 rank 上计算全局 expert ID。`ExpertMapManager` 将全局 expert ID 映射为本 rank 的 local ID：

```text
global expert id
        |
        v
ExpertMapManager
        |
        +--> 本 rank 拥有：local expert id
        |
        +--> 非本 rank 拥有：-1
```

EP-only 模式下，通用 DP+EP All-to-All 路径不一定启用，因为 `use_all2all_kernels` 的条件是 `dp_size > 1 and use_ep`。此时具体 expert kernel 可能直接接收 `ep_size/ep_rank`，只计算本地专家，再由 `MoERunner` 通过 TP group AllReduce 合并各 rank 的局部 expert 结果。因此不能把“开启 EP”简单等价成“必然调用 All-to-All”。

### 5. DP+EP：真正的 MoE dispatch/combine

DP+EP 是当前上游 MoE 通信抽象最完整的场景。

```text
每个 DP rank 的本地 token
        |
        v
Router: topk_ids/topk_weights
        |
        v
prepare / dispatch
        |
        +--> AllGather + ReduceScatter
        |    或
        +--> DeepEP / MoRI / NIXL / FlashInfer All-to-All
        |
        v
目标 expert 所在 rank
        |
        v
本地 expert GEMM + activation
        |
        v
combine
        |
        +--> 返回原 token 所属 rank
        |
        v
MoE output
```

两种主要实现：

| 通信逻辑 | dispatch | combine |
|---|---|---|
| `allgather_reducescatter` | AllGatherV 收集 token、Top-K ID、权重 | ReduceScatterV |
| DeepEP/MoRI/NIXL/FlashInfer | 根据 expert ownership 直接发送 token | 根据源 rank 和 token 映射返回并合并 |

上游把这部分抽象成：

```text
Router
  -> Quantize/Prepare/Dispatch
  -> Fused Expert GEMM
  -> Finalize/Combine
```

因此不同通信实现可以复用同一个 expert GEMM kernel。

### 6. SP：只改变 token 布局

当前上游的 SP 主要服务于 `DP + EP + TP`，自动开启条件是：

```text
enable_expert_parallel
AND data_parallel_size > 1
AND tensor_parallel_size > 1
AND backend 支持 sequence-parallel MoE
```

原因是 Attention 的输出通常已经经过 TP AllReduce，在每个 TP rank 上是完整 token；如果直接进入 EP，每个 TP rank 可能重复计算同一批 token。SP 会先沿 token 维切分：

```text
完整 hidden_states: [S, H]
        |
        v
TP rank 0: [S/TP, H]
TP rank 1: [S/TP, H]
...
        |
        v
各 rank 独立 Router + EP dispatch
        |
        v
本地 expert 计算
        |
        v
EP combine
        |
        v
TP AllGather
        |
        v
恢复 [S, H]
```

`sequence_parallel_chunk()` 会先把 token 数补齐到 TP 的整数倍，再按 TP rank 切片；MoE 结束后通过 `tensor_model_parallel_all_gather()` 恢复完整序列。因此 SP 不是另一套 expert 权重切分方式，而是 token 维度的重新分布。

对于 `allgather_reducescatter` backend，是否使用 DP group 还是 EP group由 `is_sequence_parallel` 决定：

```text
SP=false: dispatch/combine 使用 DP group
SP=true : dispatch/combine 使用 EP group
```

### 7. 上游支持的通信模式

当前源码中可以归纳为三类：

1. **TP Collective**

   - AllReduce：TP expert partial output 合并。
   - AllGather：SP 后恢复完整 token。
   - ReduceScatter：PCP 或部分 DP 输出切分。

2. **AllGather + ReduceScatter 模拟 All-to-All**

   - backend：`allgather_reducescatter`
   - dispatch：AllGatherV
   - combine：ReduceScatterV
   - 实现简单、兼容性好，但会让每个 rank 看到更多 token。

3. **真正的 All-to-All / 专用通信 kernel**

   - `deepep_high_throughput`
   - `deepep_low_latency`
   - `deepep_v2`
   - `mori_high_throughput`
   - `mori_low_latency`
   - `nixl_ep`
   - `flashinfer_nvlink_two_sided`
   - `flashinfer_nvlink_one_sided`

这些 backend 由 `DeviceCommunicator` 创建对应的 `All2AllManager`，再由 `maybe_make_prepare_finalize()` 创建相应的 `PrepareAndFinalize` 实现。部分 backend 支持异步 dispatch/combine，可以和 shared expert 计算重叠。

### 数据流架构图

```text
[Transformer Attention 输出]
        |
        | TP 输出通常在各 rank 上复制
        v
[MoE Model Forward]
        |
        | SP=true: sequence_parallel_chunk
        v
[Router / Gate]
        |
        | hidden_states + topk_ids + topk_weights
        v
[FusedMoEParallelConfig + ExpertMapManager]
        |
        +--> TP only:
        |       本地 expert TP GEMM
        |       -> TP AllReduce
        |
        +--> DP without EP:
        |       DP AllGather
        |       -> 本地 expert GEMM
        |       -> DP ReduceScatter
        |
        +--> DP + EP:
        |       EP All-to-All 或 AG/RS dispatch
        |       -> local experts
        |       -> combine
        |
        +--> SP:
                TP token chunk
                -> EP dispatch/combine
                -> TP AllGather
        |
        v
[MoE Output]
        |
        v
[下一层 Transformer]
```

### 代码定位

1. [`vllm/vllm/config/parallel.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/config/parallel.py:126) — `ParallelConfig`：定义 DP、EP、All-to-All backend，以及自动启用 SP 的条件（126-195、642-668）。
2. [`vllm/vllm/distributed/parallel_state.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/distributed/parallel_state.py:1760) — `initialize_model_parallel()`：创建 TP、DP、PCP、EP group，其中 EP group 规模为 `DP × PCP × TP`（1760-1896）。
3. [`vllm/vllm/model_executor/layers/fused_moe/config.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/layers/fused_moe/config.py:1188) — `FusedMoEParallelConfig.make()`：关闭 EP 时展开 MoE TP，开启 EP 时改为 `ep_size` 并令 MoE `tp_size=1`（1188-1235）。
4. [`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:814) — `_forward_impl()`：执行 dispatch、Router、expert kernel 和 combine（814-833）；`_maybe_reduce_final_output()` 负责必要的 TP/EP 输出归约（431-453）。
5. [`vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:214) — `ExpertMapManager`：计算 global expert 到 local expert 的映射，并将非本 rank expert 映射为 `-1`（214-255、297-307、440-453）。
6. [`vllm/vllm/distributed/device_communicators/all2all.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/distributed/device_communicators/all2all.py:42) — `AgRsAll2AllManager`：用 AllGatherV 做 dispatch、ReduceScatterV 做 combine（42-138）；其他 All-to-All backend 在同文件中实现。

验证范围：以上是当前 checkout 的静态源码分析，未在 Docker/NPU 上执行 E2E。

## 共享专家DP特性

有，但它不是一个单独叫 `shared_expert_dp` 的开关，而是共享专家天然随 DP 副本执行：

- 每个 DP rank 都保存一份共享专家权重；
- 每个 DP rank 只计算自己的本地 token；
- 共享专家不会像 routed experts 一样经过 EP dispatch；
- 共享专家内部仍可使用普通 TP：`gate_up_proj` 按 TP 切分，`down_proj` 产生 TP partial output；
- 最后由 `MoERunner` 与 routed expert 输出相加，再执行必要的 TP AllReduce；SP 模式则由模型侧 TP AllGather 恢复 token。

例如 DP=2、TP=2、EP 开启：

```text
DP0:
  TP0/TP1 保存共享专家的 TP 权重分片
  计算 DP0 的 token

DP1:
  复制同样的共享专家 TP 权重分片
  计算 DP1 的 token

共享专家不走 EP dispatch
```

因此这里的“共享专家 DP”更准确地说是：

```text
DP 维度：共享专家权重复制、token 数据分片
TP 维度：共享专家 MLP 权重切分
EP 维度：只负责 routed experts
```

共享专家可以和 routed expert 的 dispatch/combine 异步重叠，但这是执行重叠优化，不是额外的共享专家 DP 通信机制。

```text
[DP rank 本地 hidden_states]
        |
        +----------------------+
        |                      |
        v                      v
[Shared Expert]         [Router + EP Routed Expert]
 TP 权重分片             dispatch 到目标 expert
        |                      |
        +----------+-----------+
                   v
          [shared + routed output]
                   |
          TP AllReduce / SP AllGather
                   |
             [MoE output]
```

代码定位：

1. [`qwen3_moe.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/models/qwen3_moe.py:191) — 创建共享专家；`shared_expert_gate` 是复制的，shared MLP 设置 `reduce_results=False`。
2. [`qwen3_moe.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/models/qwen3_moe.py:104) — 共享 MLP 使用标准 `MergedColumnParallelLinear + RowParallelLinear`，因此内部仍有 TP。
3. [`moe_runner.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:525) — shared expert 与 routed expert 分开执行；shared expert 不进入 routed dispatch。
4. [`shared_experts.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:155) — 共享专家在当前 rank 的输入上执行，并可使用独立 CUDA stream 与 dispatch/combine 重叠。
5. [`moe_runner.py`](/home/x50063850/vllm-ascend-workspace/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:411) — 根据 routed kernel 是否已经归约，决定对 shared output 或最终结果执行 TP AllReduce。