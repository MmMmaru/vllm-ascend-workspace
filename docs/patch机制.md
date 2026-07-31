# vllm-ascend Patch 机制

本文说明 vllm-ascend 如何在不直接修改上游 vLLM 源码的情况下，接管上游的类、函数和算子，并将 NPU、HCCL、SP、FlashComm 等实现接入 vLLM 执行链路。

本文以当前工作区代码为准，运行环境和版本约束见 [`CONFIG.md`](../CONFIG.md)。

## 1. 总体认识

vllm-ascend 中的“patch”主要是 Python 运行时替换机制：

```text
上游 vLLM 类/函数
        │
        │ 进程启动时导入 patch 模块
        ▼
运行时替换为 Ascend 实现
        │
        ├── Python monkey patch
        ├── OOT CustomOp/PluggableLayer
        ├── torch.ops.vllm.* 自定义算子
        └── _C_ascend C++/CANN/ATB 算子
```

因此，`vllm/model_executor/models/qwen3_moe.py` 可以保持上游写法，但模型实例化和 forward 执行时，实际使用的可能已经是 Ascend 类和 Ascend 通信逻辑。

需要区分：

| 机制 | 作用 | 典型例子 |
| --- | --- | --- |
| Monkey patch | 直接替换上游 Python 类、方法或函数 | `Qwen3Attention.forward = ...` |
| OOT CustomOp | 创建上游层时，将实例替换为 Ascend 层 | `QKVParallelLinear -> AscendQKVParallelLinear` |
| `torch.ops.vllm.*` | 注册可被 PyTorch 调度的 Python/NPU 自定义算子 | `maybe_all_gather_and_maybe_unpad` |
| `_C_ascend` | 调用构建好的 C++/CANN/ATB 扩展 | `torch.ops._C_ascend.*` |
| Compilation Pass | 编译图阶段做 pattern replacement | RMSNorm、SP 融合 pass |

## 2. Patch 的加载时机

### 2.1 vLLM 插件入口

Ascend 通过 `vllm.platform_plugins` 注册平台入口，见 [`setup.py`](../vllm-ascend/setup.py#L540)：

```python
"vllm.platform_plugins": ["ascend = vllm_ascend:register"]
```

`register()` 返回平台类路径：

```python
return "vllm_ascend.platform.NPUPlatform"
```

在平台初始化前，vLLM 会调用 `NPUPlatform.pre_register_and_update()`。Ascend 在这里加载全局 patch，见 [`platform.py`](../vllm-ascend/vllm_ascend/platform.py#L182)：

```python
@classmethod
def pre_register_and_update(cls, parser=None):
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
```

### 2.2 `adapt_patch` 的实现

[`utils.py`](../vllm-ascend/vllm_ascend/utils.py#L511) 中的 `adapt_patch` 本身只负责导入两个 patch 包：

```python
def adapt_patch(is_global_patch=False):
    if is_global_patch:
        from vllm_ascend.patch import platform
    else:
        from vllm_ascend.patch import worker
```

真正的替换动作发生在被导入模块的顶层代码中。因此这是“import side effect”机制，而不是一个显式的 `patch_manager.apply_all()` 调度器。

### 2.3 Platform patch 和 Worker patch

[`patch/__init__.py`](../vllm-ascend/vllm_ascend/patch/__init__.py#L18) 明确区分了两类 patch：

```text
platform/  在 Worker 启动前加载，影响 Engine、Scheduler、配置和 KV Cache
worker/    在每个 NPU Worker 进程中加载，影响模型、算子和 Worker 行为
```

Platform patch 由 [`patch/platform/__init__.py`](../vllm-ascend/vllm_ascend/patch/platform/__init__.py) 中的 import 触发，例如：

```python
import vllm_ascend.patch.platform.patch_distributed
import vllm_ascend.patch.platform.patch_kv_cache_interface
import vllm_ascend.patch.platform.patch_scheduler
```

Worker patch 由 [`patch/worker/__init__.py`](../vllm-ascend/vllm_ascend/patch/worker/__init__.py) 中的 import 触发，例如：

```python
import vllm_ascend.patch.worker.patch_distributed
import vllm_ascend.patch.worker.patch_qwen3vl
import vllm_ascend.patch.worker.patch_cudagraph
```

### 2.4 Worker 进程中的加载顺序

[`worker.py`](../vllm-ascend/vllm_ascend/worker/worker.py#L88) 中的 `NPUWorker.__init__` 首先加载 Worker patch：

```python
from vllm_ascend.utils import adapt_patch
adapt_patch()
```

之后才注册算子和 Ascend OOT 层：

```python
from vllm_ascend import ops

ops.register_dummy_fusion_op()
_register_atb_extensions()
register_ascend_customop(vllm_config)
```

这保证 Worker 后续创建模型时，相关 patch、`torch.ops` 和 OOT 层已经准备完成。

多进程场景下，父进程的 Python monkey patch 不一定可靠地传递给子进程，因此每个 Worker 都需要执行一次 `adapt_patch()`。部分 EngineCore 相关 patch 还会通过通用插件入口重新加载，以覆盖子进程启动路径。

## 3. Patch 的几种实现方式

### 3.1 直接替换函数或方法

最直接的方式是给上游类重新绑定方法。例如 [`patch_qwen3vl.py`](../vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py#L35)：

```python
def forward_with_split_qkv_rmsnorm_mrope(self, positions, hidden_states):
    ...

Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope
```

之后所有 `Qwen3Attention` 和 `Qwen3MoeAttention` 实例都会通过新的方法查找规则调用 Ascend forward。

其他直接替换的例子：

```python
Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
```

以及 310P 上对 `torch.distributed.broadcast`、`torch.distributed.all_reduce` 的通信适配。

### 3.2 替换类

如果上游类需要大量 Ascend 专用状态，通常定义一个子类，然后替换上游模块中的类对象。

例如 Worker 分布式 patch 将上游的 `GroupCoordinator` 换成 Ascend 版本：

```python
vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
```

`GroupCoordinatorPatch` 增加了：

- HCCL ProcessGroup 创建和复用；
- NPU device communicator；
- `all_to_all`；
- HCCL group 的销毁和引用计数管理。

KV Cache patch 也采用类似方式，将上游的 `MLAAttentionSpec` 替换为 Ascend 版本，以支持 NPU 的 MLA/DSA cache 字段和 page size。

### 3.3 OOT CustomOp/PluggableLayer 类替换

这类机制不直接修改每个模型的构造代码，而是在上游层实例化时做替换。

上游的 `CustomOp` 和 `PluggableLayer` 会在 `__new__` 中检查 OOT registry。Ascend 在 [`utils.py`](../vllm-ascend/vllm_ascend/utils.py#L660) 建立映射：

```python
REGISTERED_ASCEND_OPS = {
    "QKVParallelLinear": AscendQKVParallelLinear,
    "RowParallelLinear": AscendRowParallelLinear,
    "RMSNorm": AscendRMSNorm,
    "FusedMoE": AscendFusedMoE,
    "VocabParallelEmbedding": AscendVocabParallelEmbedding,
    ...
}
```

随后调用：

```python
CustomOp.register_oot(
    _decorated_op_cls=op_cls,
    name=name,
)
```

因此，虽然 `qwen3_moe.py` 中仍然导入上游类：

```python
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
```

但模型创建时实际可能变成：

```text
QKVParallelLinear       -> AscendQKVParallelLinear
RowParallelLinear       -> AscendRowParallelLinear
RMSNorm                 -> AscendRMSNorm
FusedMoE                -> AscendFusedMoE
VocabParallelEmbedding -> AscendVocabParallelEmbedding
```

这就是上游模型代码不写 Ascend 特定通信，而 Ascend 仍然可以接管 Linear、Embedding、Norm 和 MoE 的原因。

### 3.4 导入钩子

[`patch_weight_utils.py`](../vllm-ascend/vllm_ascend/patch/worker/patch_weight_utils.py#L13) 提供 `ImportPatchDecorator`，按模块名保存 patch 函数。

它还替换 Python 的 `__import__`：

```python
__builtins__["__import__"] = patched_import
```

这样可以处理两种情况：

1. 目标模块已经加载：立即对 `sys.modules` 中的模块执行 patch；
2. 目标模块以后加载：import 完成后自动执行 patch。

这种方式适合权重加载模块、模型模块存在延迟导入的情况。

### 3.5 同步替换 `from ... import ...` 的局部引用

如果某个模块已经执行过：

```python
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
```

此时再修改 `kv_cache_utils.resolve_kv_cache_block_sizes`，并不会自动更新已经保存的局部引用。

所以 [`patch_kv_cache_utils.py`](../vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_utils.py) 同时修改：

```python
vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = patched_fn
vllm.v1.engine.core.resolve_kv_cache_block_sizes = patched_fn
```

类似地，`patch_kv_cache_coordinator.py` 会检查 `sys.modules`，必要时更新已经导入旧函数的 `kv_cache_manager`。

## 4. Qwen3-MoE 中 patch 后的实际执行路径

### 4.1 模型构造阶段

上游 [`qwen3_moe.py`](../vllm/vllm/model_executor/models/qwen3_moe.py) 创建：

```text
Qwen3MoeAttention
    ├── QKVParallelLinear
    ├── RMSNorm
    └── RowParallelLinear

Qwen3MoeMLP
    ├── MergedColumnParallelLinear
    ├── SiluAndMul
    └── RowParallelLinear

Qwen3MoeSparseMoeBlock
    └── FusedMoE
```

由于 OOT registry 已经在 Worker 初始化阶段注册，这些层会实例化为 Ascend 实现。

### 4.2 Ascend Linear 的选择

Ascend Linear 构造时调用 [`linear.py`](../vllm-ascend/vllm_ascend/ops/linear.py#L159)：

```python
self.custom_op, _, tp_size = get_parallel_op(
    disable_tp, prefix, self, "column"
)
```

真正的执行在：

```python
if self.custom_op is not None:
    return self.custom_op.apply(input_)
```

`get_parallel_op` 会根据 layer prefix、SP、FlashComm2、MLP TP、DSA/CP 等配置选择通信实现。

典型 SP 路径是：

```text
qkv_proj / gate_up_proj
    -> SequenceColumnParallelOp

o_proj / down_proj
    -> SequenceRowParallelOp
```

### 4.3 Column Parallel 与 FlashComm

[`linear_op.py`](../vllm-ascend/vllm_ascend/ops/linear_op.py#L423) 中的 `SequenceColumnParallelOp`：

```python
input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
    input_, label=need_all_gather
)
output_parallel = self.quant_method.apply(self.layer, input_, bias)
```

FlashComm 下，输入可能是每个 TP rank 的局部 token 布局，需要在特定 Linear 边界恢复完整 token 视图，再进行矩阵乘法。

### 4.4 Row Parallel 与 FlashComm

[`linear_op.py`](../vllm-ascend/vllm_ascend/ops/linear_op.py#L476) 中的 `SequenceRowParallelOp`：

```python
output = torch.ops.vllm.matmul_and_reduce(
    input_parallel,
    self.unique_prefix,
)
```

`matmul_and_reduce` 会根据 `ForwardContext` 决定：

```text
FlashComm 开启:
    matmul + reduce-scatter
    [T, H] -> [ceil(T / TP), H]

FlashComm 未开启:
    matmul + all-reduce
    [T, H] -> [T, H]
```

因此 FlashComm 的本质不是在模型 `forward` 外面统一切一次，而是通过 Embedding、Linear、MoE、Norm 等层内部的通信算子共同维护 local token layout。

## 5. `patch_qwen3vl.py` 的具体作用

该 patch 在非 310P Worker 中导入，主要解决 Qwen3-VL 与 FlashComm、Ascend fused kernel、PP layer range 的兼容问题。

### 5.1 替换 Qwen3 attention forward

[`patch_qwen3vl.py`](../vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py#L35) 中的新 forward：

```text
qkv_proj(hidden_states)
        │
        ├── AscendMRotaryEmbedding
        │       └── triton_split_qkv_rmsnorm_mrope
        │
        └── 普通 split + q/k RMSNorm + RoPE
        │
attention(q, k, v)
        │
o_proj(attn_output)
```

使用 MRoPE 时调用：

```python
torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(...)
```

该算子融合了 QKV 拆分、Q/K RMSNorm 和 MRoPE 处理；非 MRoPE 情况保留等价的普通 PyTorch 路径。

### 5.2 切分 Qwen3-VL DeepStack embedding

Qwen3-VL 的 DeepStack embedding 原本是完整 token 布局。patch 包装：

```python
Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
```

当 FlashComm1 开启时：

```python
deepstack_input_embeds.tensors = {
    k: v.chunk(tp_size)[tp_rank]
    for k, v in deepstack_input_embeds.tensors.items()
}
```

形状变化为：

```text
完整 DeepStack embedding: [T, H]
每个 TP rank:             [ceil(T / TP), H]
```

这样 DeepStack 输入可以和 FlashComm 产生的 local hidden states 对齐。

### 5.3 暴露 PP layer range

Qwen3-VL-MoE 的 `start_layer`、`end_layer` 位于内部语言模型对象，而 PP 检查使用外层 `Qwen3MoeLLMForCausalLM`。

patch 因此在外层增加转发属性：

```python
Qwen3MoeLLMForCausalLM.start_layer = property(
    lambda self: self.model.start_layer
)
Qwen3MoeLLMForCausalLM.end_layer = property(
    lambda self: self.model.end_layer
)
```

## 6. `torch.ops.vllm.*` 与 patch 的关系

Worker 初始化时：

```python
from vllm_ascend import ops
ops.register_dummy_fusion_op()
register_ascend_customop(vllm_config)
```

`vllm_ascend.ops` 的导入会触发：

- `ops/register_custom_ops.py`：FlashComm、padding、reduce-scatter、all-gather 等算子；
- `ops/linear.py`：`unquantized_gemm`；
- `ops/mla.py`：MLA forward；
- Triton linearnorm：QKV/RMSNorm/RoPE 融合算子；
- `ops/vocab_parallel_embedding.py`：Ascend Embedding 和 LM Head。

例如 Ascend Linear 的无量化路径：

```python
def apply(self, layer, x, bias=None):
    return torch.ops.vllm.unquantized_gemm(
        x, layer.weight, bias
    )
```

所以完整调用关系是：

```text
上游 qwen3_moe.py
        │
        ▼
OOT 替换后的 Ascend Linear/Norm/MoE
        │
        ▼
Ascend linear_op.py 通信策略
        │
        ▼
torch.ops.vllm.*
        │
        ▼
PrivateUse1 / Triton / CANN / ATB NPU 实现
```

`torch.ops.vllm.*` 是算子注册机制，不等同于 Python monkey patch；patch 负责把上游调用导向 Ascend 实现，算子负责具体执行。

## 7. 条件加载和版本差异

Patch 包不是所有场景都会无条件加载，常见条件包括：

- `is_310p()`：310P 使用不同实现；
- `HAS_TRITON`：只有 Triton 可用时才加载相关 patch；
- `DYNAMIC_EPLB`、`EXPERT_MAP_RECORD`：动态 EPLB 时加载多进程 executor patch；
- `vllm_version_is("0.22.1")`：当前版本下 v2 ModelRunner patch 被关闭；
- `enable_sp()`、FlashComm2、DSA/CP 等运行时配置：决定 Linear 使用哪种通信实现。

例如 [`patch/worker/__init__.py`](../vllm-ascend/vllm_ascend/patch/worker/__init__.py) 中：

```python
_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.22.1")
```

所以在当前 v0.22.1 环境中，部分 `patch_v2.*` 不会生效，主要使用 v1 Worker/ModelRunner 路径。

## 8. Patch 的幂等性和常见问题

Python import 默认有模块缓存，因此同一个进程内重复执行：

```python
import vllm_ascend.patch.worker
```

通常不会重复执行模块顶层代码。除此之外，部分 patch 自己还设置了：

- `_PATCHED` 全局标志；
- 函数属性标志，如 `_vllm_ascend_*_patched`；
- `vllm_ascend.__init__` 中的 `_GLOBAL_PATCH_APPLIED`。

排查 patch 不生效时，重点检查：

1. 当前进程是否执行了 `adapt_patch()`；
2. 是否被设备类型或 vLLM 版本条件跳过；
3. patch 是否发生在目标模块导入之后；
4. 是否存在 `from ... import ...` 造成的旧局部引用；
5. 当前走的是 v1 还是 v2 ModelRunner；
6. OOT registry 和 `torch.ops.vllm.*` 是否已经注册。

可以在 Worker 进程中打印：

```python
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.models.qwen3 import Qwen3Attention

print(Qwen3Attention.forward.__module__)
print(op_registry_oot.get("QKVParallelLinear"))
print(op_registry_oot.get("RMSNorm"))
```

期望看到类似：

```text
vllm_ascend.patch.worker.patch_qwen3vl
<class 'vllm_ascend.ops.linear.AscendQKVParallelLinear'>
<class 'vllm_ascend.ops.layernorm.AscendRMSNorm'>
```

## 9. 代码入口索引

| 关注内容 | 文件 |
| --- | --- |
| 全局 patch 入口 | [`vllm_ascend/platform.py`](../vllm-ascend/vllm_ascend/platform.py) |
| `adapt_patch` | [`vllm_ascend/utils.py`](../vllm-ascend/vllm_ascend/utils.py) |
| Platform patch 导入 | [`patch/platform/__init__.py`](../vllm-ascend/vllm_ascend/patch/platform/__init__.py) |
| Worker patch 导入 | [`patch/worker/__init__.py`](../vllm-ascend/vllm_ascend/patch/worker/__init__.py) |
| Worker 初始化顺序 | [`worker/worker.py`](../vllm-ascend/vllm_ascend/worker/worker.py) |
| OOT 层注册 | [`utils.py:register_ascend_customop`](../vllm-ascend/vllm_ascend/utils.py#L638) |
| Qwen3-VL patch | [`patch/worker/patch_qwen3vl.py`](../vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py) |
| Ascend Linear | [`ops/linear.py`](../vllm-ascend/vllm_ascend/ops/linear.py) |
| SP/FlashComm Linear | [`ops/linear_op.py`](../vllm-ascend/vllm_ascend/ops/linear_op.py) |
| 直接注册算子 | [`ops/register_custom_ops.py`](../vllm-ascend/vllm_ascend/ops/register_custom_ops.py) |
| 上游 OOT registry | [`vllm/model_executor/custom_op.py`](../vllm/vllm/model_executor/custom_op.py) |

