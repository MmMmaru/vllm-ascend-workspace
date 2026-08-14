# Ascend `torch.compile` 与 Pass 机制：以 SP 为例

本文以 Ascend 的 Sequence Parallelism（SP）为例，说明一次模型 forward 如何进入
`torch.compile`，Ascend 编译后端如何接管 FX 图，以及 SP pattern pass 如何改写图中的
通信和 RMSNorm。这里的 SP pass 是编译期图优化，和 eager 路径中由 Linear/custom op
直接执行的 FlashComm1 不是同一层机制。

## 1. 先区分两种 SP

当前代码中有两条相关但不同的路径：

```text
运行时 SP / FlashComm1
  enable_flashcomm1 或 VLLM_ASCEND_ENABLE_FLASHCOMM1
  → enable_sp()
  → Ascend Linear / Embedding / MoE custom op
  → reduce-scatter、all-gather、padding

编译期 SP pass
  compilation_config.pass_config.enable_sp = true
  → torch.compile 捕获 FX 图
  → SequenceParallelismPass / SequenceParallelismMoePass
  → pattern matcher 改写 all-reduce/all-gather + RMSNorm
```

两者可以同时存在：编译期 pass 负责把已经捕获的 Python/算子调用图改写成更适合
NPU 的形式；运行时 custom op 负责执行图中留下的 `torch.ops.vllm.*`、`_C_ascend`
以及 HCCL/NPU 通信。

## 2. Ascend 编译入口

Ascend 平台通过平台接口告诉 vLLM 使用自己的 PassManager 和 compiler：

- [`platform.py`](../vllm-ascend/vllm_ascend/platform.py) 的 `pass_key` 返回
  `graph_fusion_manager`；
- `get_pass_manager_cls()` 返回
  `vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager`；
- `get_compile_backend()` 返回
  `vllm_ascend.compilation.compiler_interface.AscendCompiler`。

对应关系如下：

```text
current_platform.get_pass_manager_cls()
  → GraphFusionPassManager

current_platform.get_compile_backend()
  → AscendCompiler

torch.compile(..., backend=AscendCompiler)
  → AscendCompiler.compile(...)
```

平台配置阶段还会设置 Ascend 的 OOT compiler，并在特定配置下关闭上游 CUDA/Inductor
路径。若 `cudagraph_mode == NONE`，平台代码会把 compilation mode 设为 `NONE`，此时
不会进入 `torch.compile` 图编译；因此“配置了 SP pass”不等于“本次请求一定执行了
pass”。

## 3. `torch.compile` 是在哪里触发的

上游模型用 `@support_torch_compile` 标记可编译的模型。例如 Qwen3-MoE：

```python
@support_torch_compile
class Qwen3MoeModel(...): ...
```

参见上游
[`qwen3_moe.py`](../vllm/vllm/model_executor/models/qwen3_moe.py) 和
[`decorators.py`](../vllm/vllm/compilation/decorators.py)。装饰器并不是立即编译，
它主要做三件事：

1. 把 `TorchCompileWithNoGuardsWrapper` 注入类继承关系；
2. 保存动态 shape 信息，例如输入 token 维度；
3. 改写实例的 `__call__`，第一次真正执行时再调用 `torch.compile`。

简化后的运行过程是：

```text
Model.__call__(input_ids, positions, ...)
  │
  ├─ compilation mode 为 NONE/不满足 enable_if
  │    └─ 直接执行 forward
  │
  └─ compilation mode 可编译
       ├─ 首次调用：标记动态维度并调用 torch.compile
       ├─ Dynamo 捕获 forward
       ├─ 生成 torch.fx.GraphModule
       └─ 交给 AscendCompiler
```

在当前 v1 ModelRunner 中，模型 forward 的调用入口是
[`model_runner_v1.py`](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py) 的
`_model_forward`。如果模型类已被 `support_torch_compile` 包装，`run_model()` 的
第一次调用会触发上述捕获；之后使用编译后的 callable。

## 4. AscendCompiler 如何把 Pass 插入编译流程

[`compiler_interface.py`](../vllm-ascend/vllm_ascend/compilation/compiler_interface.py)
中有两条后端路径：

### 4.1 `enable_npugraph_ex=True`

调用 `npugraph_ex_compile`，由 `npugraph_ex`/`torchair` 接管 NPU 图编译。这个路径
主要负责 NPU 图和静态 kernel；是否执行 vLLM 的 FX pattern pass，要看后端以及当前
配置是否把图交给 Ascend 的 pass manager。

### 4.2 普通 Ascend fusion pass 路径

`AscendCompiler.compile` 在没有启用 `npugraph_ex` 时调用 `fusion_pass_compile`：

```python
def compile(...):
    if enable_npugraph_ex:
        return npugraph_ex_compile(...)
    return fusion_pass_compile(...)
```

`fusion_pass_compile` 的关键逻辑是：

```python
def compile_inner(graph, example_inputs):
    current_pass_manager = compiler_config[COMPILATION_PASS_KEY]
    graph = current_pass_manager(graph)
    return graph
```

最终调用链为：

```text
torch.compile
  → AscendCompiler.compile
  → fusion_pass_compile
  → GraphFusionPassManager(graph)
  → 各个 VllmInductorPass
  → graph.recompile()
  → NPU backend lower/compile
```

上游 vLLM 的 compilation backend 会在配置阶段执行
`pass_manager.configure(vllm_config)`，并把 PassManager 放入
`inductor_config[pass_key]`。Ascend 使用自己的 `GraphFusionPassManager`，因为当前
NPU 路径不能直接复用上游 CUDA/Inductor 的完整 PassManager。

## 5. GraphFusionPassManager 如何选择 Pass

实现位于
[`graph_fusion_pass_manager.py`](../vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py)。

`configure()` 先添加通用融合 pass，例如：

```text
fuse_norm_quant       → AddRMSNormQuantFusionPass
fuse_qknorm_rope      → QKNormRopeFusionPass
fuse_allreduce_rms    → MatmulAllReduceAddRMSNormPass
fuse_muls_add         → MulsAddFusionPass
```

当下面配置为 true 时，才添加两个 SP pass：

```python
if config.compilation_config.pass_config.enable_sp:
    self.passes.append(SequenceParallelismPass(config))
    self.passes.append(SequenceParallelismMoePass(config))
```

因此 pass 的第一层开关是：

```text
compilation_config.pass_config.enable_sp
```

每次图编译时，PassManager 从 `get_pass_context().compile_range` 取当前编译的 token
范围，然后逐个检查：

```python
if pass_.is_applicable_for_range(compile_range):
    pass_(graph)
```

SP pass 的适用条件是：

```python
compile_range.start >= self.min_tokens
```

当前 Ascend 实现中，`get_sp_min_token_num()` 对 MoE 返回 `1`，对 dense 模型默认返回
`1000`。所以一个小 batch 的 dense 图可能不会执行 SP pass，而 MoE 图更容易执行。

## 6. SequenceParallelismPass 的结构

代码位于
[`sequence_parallelism.py`](../vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism.py)。

### 6.1 Pattern helper

`_SequenceParallelPatternHelper` 保存 TP 通信所需的信息：

```python
self.tp_group = get_tp_group()
self.tp_size = get_tensor_model_parallel_world_size()
self.tp_rank = get_tp_group().rank_in_group
```

并提供四个图内 helper：

```python
_all_reduce(x)
_reduce_scatter(x)
_all_gather(x)
empty(...)
```

注意：这些函数是在 pattern/replacement 函数中被调用的。它们不是运行时先执行一次
的 Python 通信，而是作为节点被 Dynamo/FX 记录到图中。

### 6.2 Pattern 的注册

以 `MiddleAllReduceRMSNormPattern` 为例：

```python
def register(self, pm_pass):
    def pattern(input, weight, residual): ...

    def replacement(input, weight, residual): ...

    pm.register_replacement(
        pattern,
        replacement,
        self.get_inputs(),
        pm.fwd_only,
        pm_pass,
    )
```

`get_inputs()` 提供 `[8, 16]`、`[16]` 等示例输入，PatternMatcher 用它建立可匹配的
FX pattern。`pm.fwd_only` 表示只改 forward 图，不处理反向图。

构造 `SequenceParallelismPass` 时，会针对 `1e-5` 和 `1e-6` 两种 epsilon 注册：

```text
MiddleAllReduceRMSNormPattern
LastAllReduceRMSNormPattern
Qwen3VLMiddleAllReduceRMSNormPattern
```

## 7. Dense SP：AllReduce + RMSNorm 如何被改写

### 7.1 中间层原始图

原始图可以抽象成：

```text
input [T, H]
  │
  ├─ tensor_model_parallel_all_reduce
  │       [T, H]
  │
  └─ npu_add_rms_norm_bias(x, residual, weight)
          │
          ├─ result
          └─ residual
```

对应 pattern：

```python
x = self._all_reduce(input)
result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
    x, residual, weight, None, self.eps
)
```

### 7.2 SP replacement 图

替换后：

```text
input [T, H]
  │
  ├─ reduce_scatter
  │       [ceil(T / TP), H]
  │
  ├─ maybe_chunk_residual
  │       residual → [ceil(T / TP), H]
  │
  ├─ npu_add_rms_norm_bias(local_x, local_residual, weight)
  │       local result [ceil(T / TP), H]
  │
  └─ all_gather(result)
          [T, H]
```

对应 replacement：

```python
reduce_scatter = self._reduce_scatter(input)
residual = torch.ops.vllm.maybe_chunk_residual(reduce_scatter, residual)
result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
    reduce_scatter, residual, weight, None, self.eps
)
all_gather = self._all_gather(result)
return all_gather, residual
```

这里的核心优化是：

```text
原始：先 all-reduce，再对完整 [T, H] 做 RMSNorm
替换：先 reduce-scatter，在本卡局部 token 上做 RMSNorm，再按需要 all-gather
```

`residual` 不能直接沿用完整布局，因此 replacement 显式调用
`maybe_chunk_residual`。这和运行时 FlashComm 中 residual 的本地布局约束是一致的。

### 7.3 最后一层

`LastAllReduceRMSNormPattern` 的输入替换相同，但最后一层不需要继续返回 residual：

```text
reduce-scatter
  → local RMSNorm
  → all-gather result
```

## 8. Qwen3-VL 的 SP pattern

多模态中间层多了 `deepstack_input_embeds`：

```text
原始：
all_reduce(input) + deepstack_input_embeds
  → npu_add_rms_norm_bias

替换：
reduce_scatter(input)
  + deepstack_input_embeds.chunk(TP)[tp_rank]
  → local RMSNorm
  → all_gather
```

对应代码是 `Qwen3VLMiddleAllReduceRMSNormPattern`。它必须对 DeepStack embedding
也做 TP chunk，否则 hidden states 是 `[T/TP, H]`，而视觉侧输入仍是 `[T, H]`，无法逐
token相加。

这和 Worker patch 中对 `_get_deepstack_input_embeds` 的切分是互补关系：

```text
Worker patch：在模型执行前把 DeepStack buffer 变成 TP-local 布局
Compile pass：在 FX 图中把 all-reduce + add + RMSNorm 改成 local add + RMSNorm
```

## 9. SequenceParallelismMoePass：消除多余 AllGather

代码位于
[`sequence_parallelism_moe.py`](../vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism_moe.py)。

它主要处理已经具有 SP local 输入的 MoE/多模态图。

### 9.1 AllGather + Slice + RMSNorm

原始图：

```text
input [T/TP, H]
  → all_gather [T, H]
  → [:num_tokens]
  → RMSNorm
```

替换为：

```text
input [T/TP, H]
  → maybe_chunk_residual
  → RMSNorm(local)
  → all_gather(result)
```

也就是说，Pass 将“先恢复完整 token、再截取本地 token”的顺序改成“直接在本地 token
上计算”。

### 9.2 AllGather + sequence_parallel_chunk

另一个 pattern 是：

```text
all_gather(input)
  → sequence_parallel_chunk(...)
```

如果输入本来就是正确的 TP-local 布局，这两个操作可以整体替换为：

```text
input
```

这就是 `AllGatherChunkNoOpPattern`，目标是删除通信后又立即切回本地的冗余路径。

## 10. 运行时 custom op 与编译 pass 的关系

Pass replacement 中出现了两类算子：

```python
torch.ops._C_ascend.npu_add_rms_norm_bias
torch.ops.vllm.maybe_chunk_residual
torch.ops.vllm.reduce_scatter
torch.ops.vllm.all_gather
```

它们的职责不同：

| 调用 | 作用 |
| --- | --- |
| `torch.ops._C_ascend.*` | Ascend C++/CANN/NPU 原生算子 |
| `torch.ops.vllm.*` | vLLM-Ascend 注册的 Python/PrivateUse1 算子或通信封装 |
| Pattern pass | 在 FX 图中识别并替换这些调用的排列方式 |

例如，`maybe_chunk_residual` 的算子注册在
[`register_custom_ops.py`](../vllm-ascend/vllm_ascend/ops/register_custom_ops.py)。
Pass 并不执行自己的 HCCL 实现，而是将图改写成后续 backend 能够编译和执行的通信
算子组合。

## 11. `update_pass_config` 与 FlashComm1 的桥接

Ascend ModelRunner 中有一个临时兼容函数：

[`model_runner_v1.py`](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py) 的
`update_pass_config`。

它做的事情是：

```python
original = pass_config.enable_sp
pass_config.enable_sp = enable_sp(vllm_config)
try:
    yield
finally:
    pass_config.enable_sp = original
```

这说明当前代码同时维护两套 SP 状态：

```text
pass_config.enable_sp  → 是否向 GraphFusionPassManager 注册 SP pass
enable_sp(config)      → 是否开启 FlashComm1/eager SP
```

`enable_sp_by_pass` 则是一个桥接状态：当不是 enforce-eager 且 compilation pass 的
`enable_sp` 为 true 时，它允许部分 custom op/通信路径识别“SP 是由 pass 开启的”。

调试时不能只看 `enable_flashcomm1`，也要同时检查：

```text
vllm_config.compilation_config.mode
vllm_config.compilation_config.pass_config.enable_sp
get_ascend_config().enable_sp_by_pass
compile_range
当前模型是 dense 还是 MoE
```

## 12. 一次完整的 SP 编译流程

```text
1. NPUPlatform.apply_config_platform_defaults
   └─ 设置 SP pass 默认 token threshold

2. ModelRunner 构造模型
   └─ Qwen3MoeModel 的 forward 被 support_torch_compile 包装

3. 第一次 model.__call__
   └─ TorchCompileWithNoGuardsWrapper
      └─ torch.compile(..., backend=AscendCompiler)

4. TorchDynamo 捕获 forward
   └─ 生成 FX GraphModule

5. AscendCompiler.compile
   └─ fusion_pass_compile
      └─ GraphFusionPassManager(graph)

6. GraphFusionPassManager
   ├─ NoOp/RMSNorm/其他融合 pass
   ├─ SequenceParallelismPass
   └─ SequenceParallelismMoePass

7. SequenceParallel pass
   ├─ pattern matcher 查找 all-reduce/all-gather + RMSNorm
   ├─ 替换为 reduce-scatter + local RMSNorm + 必要 all-gather
   └─ graph.recompile()

8. Ascend NPU backend
   └─ 将改写后的 FX 图编译为 NPU 可执行图

9. 后续请求
   └─ 复用编译缓存/compiled callable
```

## 13. 如何确认 Pass 是否真的生效

建议按以下顺序排查：

### 配置层

```python
print(vllm_config.compilation_config.mode)
print(vllm_config.compilation_config.pass_config.enable_sp)
print(vllm_config.compilation_config.oot_compiler)
```

应确认 compilation mode 不是 `NONE`，compiler 是 Ascend compiler，且 pass flag 为
true。

### 日志层

`SequenceParallelismPass` 中已有日志：

```text
SequenceParallelismPass compile_range=... applicable=...
Replaced N patterns
after apply replacement ...
```

可以打开 debug 日志，搜索：

```text
npu_sequence_parallelism_pass
SequenceParallelismPass
SequenceParallelismMoePass
Replaced
```

### 图层

改写前后重点观察：

```text
all_reduce(input)
all_gather(input)
sequence_parallel_chunk(all_gather(...))
```

是否变成：

```text
reduce_scatter(input)
maybe_chunk_residual(...)
local npu_add_rms_norm_bias(...)
all_gather(result)
```

如果 `matched_count == 0`，通常不是算子没有注册，而是实际 FX 图的节点顺序、参数
签名、epsilon、shape 或 compile range 没有匹配 pattern。

## 14. 常见误区

### 误区一：打开 FlashComm1 就一定会触发 SP pass

不一定。FlashComm1 是运行时通信路径；SP pass 还要求 compilation mode 可用、
`pass_config.enable_sp=True`、compile range 达到阈值，并且 FX 图结构匹配。

### 误区二：Pass 会改变模型 Python forward 源码

不会。Pass 只修改本次 `torch.compile` 产生的 FX 图，原始
`qwen3_moe.py` 文件和 Python 类定义不变。

### 误区三：Pass 中的 `_all_gather` 会在注册阶段执行

不会。`register()` 阶段只是把 pattern/replacement 注册到
`PatternMatcherPass`；真正的节点替换发生在 `self.patterns.apply(graph)`。

### 误区四：所有 SP 都由 `SequenceParallelismPass` 实现

不是。Embedding、Linear、MoE、LayerNorm 等 eager/custom-op 路径仍有自己的
FlashComm/SP 逻辑；Pass 只覆盖它注册的图模式。

## 15. 相关代码入口

| 内容 | 文件 |
| --- | --- |
| Ascend platform 编译接口 | [`platform.py`](../vllm-ascend/vllm_ascend/platform.py) |
| Ascend compiler | [`compiler_interface.py`](../vllm-ascend/vllm_ascend/compilation/compiler_interface.py) |
| Ascend PassManager | [`graph_fusion_pass_manager.py`](../vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py) |
| Dense SP pass | [`sequence_parallelism.py`](../vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism.py) |
| MoE/AllGather SP pass | [`sequence_parallelism_moe.py`](../vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism_moe.py) |
| No-op cleanup | [`noop_elimination.py`](../vllm-ascend/vllm_ascend/compilation/passes/noop_elimination.py) |
| vLLM compile decorator | [`decorators.py`](../vllm/vllm/compilation/decorators.py) |
| vLLM compile backend | [`backends.py`](../vllm/vllm/compilation/backends.py) |
| SP runtime custom op | [`register_custom_ops.py`](../vllm-ascend/vllm_ascend/ops/register_custom_ops.py) |
| SP Linear runtime路径 | [`linear_op.py`](../vllm-ascend/vllm_ascend/ops/linear_op.py) |
