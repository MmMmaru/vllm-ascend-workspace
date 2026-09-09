# SP 与 FlashComm 配置及传递链路

本文说明 vLLM Ascend 中用户如何打开 SP/FlashComm，以及配置从命令行或环境变量进入进程后，如何传到 `AscendConfig`、worker、`ForwardContext` 和通信算子。

相关实现入口：

- [additional_config 用户指南](../vllm-ascend/docs/source/user_guide/configuration/additional_config.md)
- [EngineArgs 参数定义与 `VllmConfig` 构造](../vllm/vllm/engine/arg_utils.py)
- [AscendConfig](../vllm-ascend/vllm_ascend/ascend_config.py)
- [Ascend 运行时开关](../vllm-ascend/vllm_ascend/utils.py)
- [Ascend ForwardContext](../vllm-ascend/vllm_ascend/ascend_forward_context.py)

## 1. 用户侧有哪些开关

### 1.1 Ascend FlashComm1：推荐使用 `additional_config`

FlashComm1 的用户配置键是 `enable_flashcomm1`，默认关闭：

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_flashcomm1": true}'
```

离线 API 使用同一个 `VllmConfig.additional_config` 字段：

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=2,
    additional_config={"enable_flashcomm1": True},
)
```

迁移期间仍可使用旧环境变量：

```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2
```

环境变量在 [envs.py](../vllm-ascend/vllm_ascend/envs.py#L72) 中转换为 `bool(int(...))`，代码已标记为 deprecated；新部署应优先使用 `additional_config`。

### 1.2 上游原生 SP：`compilation_config.pass_config.enable_sp`

上游 vLLM 的编译 pass 也有一个名为 `enable_sp` 的开关。它通过 `--compilation-config` 传入：

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --compilation-config \
  '{"pass_config":{"enable_sp":true,"sp_min_token_num":1}}'
```

该字段定义在 [PassConfig](../vllm/vllm/config/compilation.py#L107)，表示启用上游编译/图变换的 Sequence Parallelism。它与 Ascend 的 `additional_config.enable_flashcomm1` 不是同一个字段，也不是简单的别名：

| 配置 | 所在对象 | 主要作用 |
|---|---|---|
| `additional_config.enable_flashcomm1` | `VllmConfig.additional_config` | Ascend FlashComm1 运行时通信路径（代码内部的 `enable_sp()`） |
| `compilation_config.pass_config.enable_sp` | `VllmConfig.compilation_config.pass_config` | 上游 SP pass、编译和图捕获逻辑 |

Ascend 代码同时兼容两者。不要因为两个字段都叫 SP，就认为打开一个一定会自动打开另一个。

### 1.3 FlashComm2

FlashComm2 的配置键是 `enable_flashcomm2_parallel_size`，正整数表示 O-proj 的 FlashComm2 TP group size，`0` 表示关闭：

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --additional-config \
  '{"enable_flashcomm1":true,"enable_flashcomm2_parallel_size":2}'
```

旧环境变量为 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`。代码实际读取的 key 是带 `enable_` 前缀的 `enable_flashcomm2_parallel_size`（[ascend_config.py](../vllm-ascend/vllm_ascend/ascend_config.py#L171)）；如果其他文档写成 `flashcomm2_parallel_size`，应以当前代码为准。

## 2. 配置优先级

对 FlashComm1 和 FlashComm2 这类同时支持 JSON 与环境变量的 Ascend 开关，优先级是：

```text
additional_config.<key>
        > 对应环境变量
        > envs.py 中的默认值
```

例如：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
vllm serve ... --additional-config '{"enable_flashcomm1":true}'
```

最终 FlashComm1 为 `True`。这个优先级由 [AscendConfig._get_config_value](../vllm-ascend/vllm_ascend/ascend_config.py#L305) 实现；如果使用环境变量，会打印迁移提示。

`enable_sp(vllm_config)` 的解析顺序与此一致：

1. 当前 `vllm_config.additional_config` 明确包含 `enable_flashcomm1` 时，直接使用该值；
2. 否则读取已经初始化的 `AscendConfig.enable_flashcomm1`；
3. 如果 `AscendConfig` 尚未初始化，则回退到 `VLLM_ASCEND_ENABLE_FLASHCOMM1`。

因此，调试时应同时查看原始配置和解析后的配置，不能只看环境变量。

## 3. 配置是怎样层层传递的

整体路径可以概括为：

```text
CLI/环境变量
    │
    ├─ --additional-config (JSON)
    ├─ --compilation-config (JSON)
    └─ VLLM_ASCEND_ENABLE_FLASHCOMM1 / FLASHCOMM2...
    │
    ▼
EngineArgs
    │  additional_config / compilation_config
    ▼
VllmConfig
    │  __post_init__：平台默认值、pass_config 校验、图/并行参数调整
    ▼
NPUPlatform.check_and_update_config()
    │
    └─ init_ascend_config(vllm_config)
          ▼
       AscendConfig
       ├─ enable_flashcomm1
       ├─ enable_flashcomm2_parallel_size
       └─ enable_sp_by_pass
    │
    ▼
NPUWorker 初始化
    │  注册 Ascend custom ops，并复用 AscendConfig singleton
    ▼
NPUModelRunner / ForwardContext
    │  enable_sp() → flash_comm_v1_enabled、pad_size
    ▼
FlashComm 算子路径
    ├─ maybe_all_gather_and_maybe_unpad
    ├─ maybe_pad_and_reduce (reduce-scatter)
    ├─ SequenceColumnParallelOp
    └─ SequenceRowParallelOp
```

### 3.1 `EngineArgs` 接收用户输入

`arg_utils.py` 将 `--additional-config` 和 `--compilation-config` 注册为 JSON 参数（[参数注册](../vllm/vllm/engine/arg_utils.py#L1518)）。解析后，`EngineArgs` 中分别保存：

```python
EngineArgs.additional_config: dict[str, Any]
EngineArgs.compilation_config: CompilationConfig
```

在创建引擎配置时，`EngineArgs` 将它们原样传入 `VllmConfig(...)`（[构造位置](../vllm/vllm/engine/arg_utils.py#L2315)）。这里不会把 `enable_flashcomm1` 改写成 `pass_config.enable_sp`。

### 3.2 `VllmConfig.__post_init__` 处理上游配置

`VllmConfig` 初始化时先调用平台默认值钩子，再做通用配置校验（[初始化流程](../vllm/vllm/config/vllm.py#L1133)）。对上游 `pass_config.enable_sp`，会执行以下逻辑：

- `fuse_gemm_comms=True` 会隐式要求 `pass_config.enable_sp=True`；
- TP 等于 1 时自动关闭原生 SP；
- 未指定 `sp_min_token_num` 时，根据 hidden size、TP 和 dtype 计算阈值；
- 阈值不适合时，原生 SP 会被关闭。

Ascend 平台的 [apply_config_platform_defaults](../vllm-ascend/vllm_ascend/platform.py#L248) 还会为原生 SP 补齐 Ascend 的 `sp_min_token_num`。这些操作只针对 `compilation_config.pass_config`，不等价于打开 FlashComm1。

### 3.3 平台校验并创建 `AscendConfig`

随后 vLLM 调用 [NPUPlatform.check_and_update_config](../vllm-ascend/vllm_ascend/platform.py#L411)。该函数调用：

```python
ascend_config = init_ascend_config(vllm_config)
```

`AscendConfig` 从 `vllm_config.additional_config` 读取 Ascend 专属字段，并按“JSON 覆盖环境变量”的优先级解析 FlashComm1/2（[构造函数](../vllm-ascend/vllm_ascend/ascend_config.py#L34)）。同时计算：

```python
enable_sp_by_pass = (
    not model_config.enforce_eager and compilation_config.pass_config.enable_sp
)
```

这说明 `enable_sp_by_pass` 是对上游 pass 开关的一个 Ascend 侧视图，而 `enable_sp()` 主要反映 `enable_flashcomm1`。

平台会在配置阶段检查 FlashComm1 的基本约束（[校验代码](../vllm-ascend/vllm_ascend/platform.py#L733)）：

- `tensor_parallel_size` 必须大于 1；
- MoE 模型启用 FlashComm1 时必须同时启用 `--enable-expert-parallel`。

FlashComm2 还会检查 group size 不超过全局 TP、全局 TP 能整除该 group size，并禁止与某些 O-proj TP 配置冲突（[校验函数](../vllm-ascend/vllm_ascend/utils.py#L1234)）。

`init_ascend_config` 使用 singleton。相同 `vllm_config` 且没有 `additional_config.refresh=true` 时，会复用原对象；清理时 [clear_ascend_config](../vllm-ascend/vllm_ascend/ascend_config.py#L952) 也会清理 `enable_sp()` 的缓存。

### 3.4 worker 进程中的初始化

每个 NPU worker 在初始化时：

1. 注册 Ascend custom ops；
2. 调用 `init_ascend_config(vllm_config)`；
3. 创建 model runner。

入口在 [worker.py](../vllm-ascend/vllm_ascend/worker/worker.py#L105)。多进程执行器把同一份序列化的 `VllmConfig` 传给各 worker；每个进程有自己的 Python singleton 和 NPU/HCCL 通信组。

### 3.5 `ForwardContext` 决定当前 batch 是否真正走 FlashComm

配置为真不代表每个 batch 都一定使用 FlashComm1。model runner 在执行模型前调用 [set_ascend_forward_context](../vllm-ascend/vllm_ascend/ascend_forward_context.py#L57)，根据模型类型和本次 token 数量生成运行时字段：

```python
forward_context.flash_comm_v1_enabled
forward_context.flashcomm_v2_enabled
forward_context.pad_size
```

当前规则是：

| 场景 | `flash_comm_v1_enabled` |
|---|---|
| MoE 主模型 | `enable_sp(vllm_config)` 且 `num_tokens is not None` |
| dense 主模型 | `enable_sp(vllm_config)` 且 `num_tokens > 1000` |
| draft model | 关闭 |

因此 dense 模型的小 batch 可能配置已打开，但本轮仍走普通 AllReduce；这是代码中的性能阈值，不是配置丢失。

如果 FlashComm1 或 FlashComm2 打开，`pad_size` 会把 token 数补齐到 TP 的整数倍；DP 场景还会按 DP ranks 中最大 token 数统一 `padded_length`。model runner 在更早的 batch 阶段也通过 [_pad_for_sequence_parallelism](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2939) 对 token 数做 TP 对齐。

## 4. 运行时开关如何进入通信算子

### 4.1 Column parallel：必要时 All-Gather 输入

当 `enable_sp()` 为真且线性层前缀匹配（如 `qkv_proj`、`gate_up_proj`、`in_proj`），[linear_op.py](../vllm-ascend/vllm_ascend/ops/linear_op.py#L600) 选择 `SequenceColumnParallelOp`。

其执行顺序是：

```text
本卡 local hidden states
    → maybe_all_gather_and_maybe_unpad
    → column-parallel GEMM
    → 本卡 column 输出
```

`maybe_all_gather_and_maybe_unpad` 在 [register_custom_ops.py](../vllm-ascend/vllm_ascend/ops/register_custom_ops.py#L39) 中读取 `ForwardContext.flash_comm_v1_enabled`：需要时做 TP/EP All-Gather，并依据 `pad_size` 去掉补齐 token。

### 4.2 Row parallel：Pad 后 Reduce-Scatter

当层前缀匹配 `o_proj`、`down_proj` 等，[linear_op.py](../vllm-ascend/vllm_ascend/ops/linear_op.py#L641) 选择 `SequenceRowParallelOp`。它调用 `matmul_and_reduce`：

```text
local input
    → row-parallel GEMM
    → （必要时 pad）
    → TP reduce-scatter
    → 每卡 local token 布局
```

`maybe_pad_and_reduce` 的实现见 [register_custom_ops.py](../vllm-ascend/vllm_ascend/ops/register_custom_ops.py#L72)：FlashComm1 关闭时是 TP AllReduce，打开时是 pad + Reduce-Scatter。custom op 的 `fake_impl` 只为 tracing/shape 推导提供形状相容的假实现，不会替代真实 NPU 通信实现。

### 4.3 Embedding 和 MoE

- 词表 embedding 在 [vocab_parallel_embedding.py](../vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py#L229) 中先执行本卡词表查找，再调用 `maybe_pad_and_reduce`，因此 embedding 输出也能进入 FlashComm 的 local token 布局。
- fused MoE 的 prepare/router 需要跨 TP/EP 收集 token 或路由信息；finalize 阶段再通过 `maybe_pad_and_reduce` 汇回 local 布局。它们使用的是否为 FlashComm 路径，同样由 `ForwardContext` 和 `enable_sp_by_pass()` 决定。

### 4.4 输出边界与 PP

model runner 在最后一个 PP stage 得到完整 hidden states 后，如果当前 context 开启 FlashComm1，会执行一次边界 All-Gather（[model_runner_v1.py](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2935)）。

中间 PP stage 则有 Ascend 专门的 slice 逻辑：

- FlashComm1 不会像上游原生 SP 那样在 PP send 前 scatter residual；
- 所以 `sync_and_gather_intermediate_tensors` 在 Ascend 被重定向到 `sync_and_slice_intermediate_tensors`；
- 开启 FlashComm1 时只复制/返回 `ceil(num_tokens / TP)` 行，而不是先做 residual All-Gather。

实现见 [PP+SP override](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2947)。这也是为什么 PP+FlashComm 的 PP receive 可以保持 FlashComm local token 布局。

## 5. 原生 SP 与 FlashComm1 的交互

代码中存在两个判断：

```python
enable_sp(vllm_config)  # Ascend FlashComm1 开关
enable_sp_by_pass()  # compilation_config.pass_config.enable_sp 的 Ascend 视图
```

`register_custom_ops.py` 在 EP 通信场景会使用 `enable_sp_by_pass()` 作为额外条件；model runner 的 padding 也接受二者之一。因此同时打开时，编译 pass 和 Ascend 通信路径都会参与。

为兼容历史 FlashComm1，model runner 在部分图/ACL graph 解析期间还用 `update_pass_config` 临时把：

```python
compilation_config.pass_config.enable_sp = enable_sp(vllm_config)
```

包在 context manager 中，退出后恢复原值（[update_pass_config](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L5222)）。这只是图解析期间的桥接，不代表两个用户配置字段已经合并。

## 6. 配置修改、缓存和多进程注意事项

`enable_sp()` 使用模块级 `_ENABLE_SP` 缓存。运行中的进程里直接修改 `vllm_config.additional_config` 后，旧缓存可能仍生效；测试或 RLHF 场景需要：

```python
additional_config = {
    "enable_flashcomm1": True,
    "refresh": True,
}
```

`refresh=true` 会让 `enable_sp()` 重新解析；`init_ascend_config()` 也会重建 AscendConfig。生产服务通常应在启动前一次性确定配置，不建议在请求期间修改。

worker 是多进程的，每个 worker 都需要完成 custom op 注册、AscendConfig 初始化和通信组初始化；只在 driver 进程中临时设置 Python 变量，不能替代 `VllmConfig.additional_config` 或继承到 worker 的环境变量。

## 7. 排查配置是否真正生效

建议按以下层次打印或断点检查：

```python
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_sp, enable_sp_by_pass
from vllm.forward_context import get_forward_context

print(vllm_config.additional_config)
print(get_ascend_config().enable_flashcomm1)
print(get_ascend_config().enable_flashcomm2_parallel_size)
print(enable_sp(vllm_config))
print(enable_sp_by_pass())

ctx = get_forward_context()
print(ctx.flash_comm_v1_enabled, ctx.flashcomm_v2_enabled, ctx.pad_size)
```

解释结果时按这个顺序判断：

1. `additional_config` 是否真的进入 `VllmConfig`；
2. `AscendConfig` 是否按预期覆盖了环境变量；
3. TP 是否大于 1，MoE 是否启用了 EP；
4. 当前 batch 是否满足 dense 模型的 `num_tokens > 1000` 门槛；
5. `ForwardContext.flash_comm_v1_enabled` 是否为真；
6. 算子是否命中 `SequenceColumnParallelOp`/`SequenceRowParallelOp`，以及 `maybe_pad_and_reduce` 是否走 Reduce-Scatter 分支。

常见误判是只看到 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 就认为当前 batch 已经在走 FlashComm。环境变量只提供默认输入，最终是否走通信优化还要经过 `AscendConfig`、TP/EP 校验、batch token 门槛和 `ForwardContext` 的运行时决策。

