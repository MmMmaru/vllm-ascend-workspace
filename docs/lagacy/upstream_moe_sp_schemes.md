# vLLM 上游新 MoE 模型的 SP 方案

本文基于当前工作区的上游 `vllm` 子模块（commit `fe784ff22`）整理，覆盖：

- DeepSeek V4（下文简称 DS V4）
- Qwen3.5 MoE
- Kimi-K2.5
- GLM-5.2

这里的 SP 特指 vLLM MoE 代码中的 `use_sequence_parallel_moe`，即“把 token 维度按 TP rank 分片后再进入 MoE”。它不等同于 attention 的 Context Parallel、PCP、DCP，也不等同于只有专家权重切分的 Expert Parallel（EP）。

## 结论先看

| 模型 | 当前上游实际模型实现 | 是否接入通用 MoE-SP | 主要方案 |
| --- | --- | --- | --- |
| DS V4 | `vllm.models.deepseek_v4.nvidia.model.DeepseekV4ForCausalLM` | 否，当前模型层没有接入通用 `use_sequence_parallel_moe` token 流 | 重点是 EP；`deep_gemm_mega_moe` 路径要求 EP，并由 MegaMoE/DeepGEMM 管理本地专家和专家组 |
| Qwen3.5 MoE | `Qwen3_5MoeForConditionalGeneration` → `Qwen3_5DecoderLayer` → `Qwen3NextSparseMoeBlock` | 是 | 仅在 MoE layer、无 PP 时，在 attention 输出后 reduce-scatter；MoE 使用本地 token shard，后续需要完整 token 序列时再 all-gather |
| Kimi-K2.5 | 多模态外壳 `KimiK25ForConditionalGeneration`；语言模型显式初始化为 `DeepseekV2ForCausalLM` | 是，复用 DeepseekV2 的 MoE-SP | Kimi-K2.5 的文本 backbone 走 `DeepseekV2DecoderLayer`，不是 `kimi_linear.py`；attention→MoE 使用 reduce-scatter/chunk，MoE 后保持 sequence-sharded |
| GLM-5.2 | `GlmMoeDsaForCausalLM`，实际继承 `DeepseekV2ForCausalLM` | 是，复用 DeepseekV2 的 MoE-SP | `glm_moe_dsa` 只改变 GLM/DSA 配置和模型注册；MoE-SP 的 token、attention、residual、最终 gather 逻辑来自 DeepseekV2 |

因此，四者不能简单归纳为“都有 SP”：当前上游是“DS V4 主要走 EP/MegaMoE，另外三者接入同一套通用 MoE-SP”。

## 1. 通用 MoE-SP 的触发条件

`ParallelConfig.use_sequence_parallel_moe` 的触发条件是：all2all backend 在支持列表中、开启 `enable_expert_parallel`，并且 `tensor_parallel_size > 1`。它的直接动机是 attention 的输出 all-reduce 会让每个 TP rank 持有重复 token；进入 EP MoE 前先切成 token shard，可以避免每个 rank 对同一批 token 重复执行专家计算。[parallel.py](../vllm/vllm/config/parallel.py#L633-L655)

具体的 `sequence_parallel_chunk()` 会在 token 数不能被 TP 整除时先在 token 维补齐，再按 `tp_rank` 取本 rank 的连续 chunk；因此 SP 输入的第一维通常是 `ceil(num_tokens / TP)`。[utils.py](../vllm/vllm/model_executor/models/utils.py#L871-L897)

`FusedMoE` 创建并行配置时，SP 会把 `sp_size` 设为 TP size；如果同时开启 EP，则把原来的 TP rank flatten 到 EP rank，并将 MoE 内部的 `tp_size` 变成 1，每个 rank 持有一组完整专家。SP 是否生效由传入的 `is_sequence_parallel` 决定，而不是仅由全局配置或 EP 自动推断。[layer.py](../vllm/vllm/model_executor/layers/fused_moe/layer.py#L44-L70)、[config.py](../vllm/vllm/model_executor/layers/fused_moe/config.py#L1031-L1054)、[config.py](../vllm/vllm/model_executor/layers/fused_moe/config.py#L1202-L1248)

### 通用数据流

```text
[embedding / 上一层输出: [T, H]，TP rank 间重复]
                 |
                 | MoE layer 且 use_sequence_parallel_moe=true
                 v
[attention 输出 pad + TP reduce-scatter]
                 |
                 v
[residual sequence_parallel_chunk: 每 rank [ceil(T/TP), H]]
                 |
                 v
[FusedMoE: 本地 router + EP dispatch/combine + 本地专家]
                 |
                 | 仍是 token shard；不在该 MLP 处无条件 all-gather
                 v
[下一层：若需要完整序列，TP all-gather hidden_states/residual]
                 |
                 v
[最终 norm / logits 消费完整 token 序列]
```

## 2. Qwen3.5 MoE：混合注意力模型中的 MoE-SP

### 接入位置

Qwen3.5 MoE 注册到 `qwen3_5.py`，`Qwen3_5MoeForConditionalGeneration` 继承 `Qwen3_5ForCausalLMBase` 并设置 MoE 元数据；其 `Qwen3_5Model` 继承 `Qwen3NextModel`，因此复用后者的模型级 gather 边界。[registry.py](../vllm/vllm/model_executor/models/registry.py#L568-L577)、[qwen3_5.py](../vllm/vllm/model_executor/models/qwen3_5.py#L209-L249)、[qwen3_5.py](../vllm/vllm/model_executor/models/qwen3_5.py#L376-L385)

`Qwen3_5DecoderLayer` 根据 `config.model_type == "qwen3_5_moe_text"` 创建 `Qwen3NextSparseMoeBlock`。只有当当前 layer 是 MoE layer、全局启用了 `use_sequence_parallel_moe` 且 PP size 为 1 时，才设置 `use_attn_reduce_scatter_for_moe`。[qwen3_5.py](../vllm/vllm/model_executor/models/qwen3_5.py#L112-L160)

### 运行时顺序

`Qwen3NextSparseMoeBlock` 将全局开关保存为 `self.is_sequence_parallel`，并把它同时传给 shared expert 和 `FusedMoE`。如果调用者没有提前完成分片，MoE 自己先调用 `sequence_parallel_chunk()`；如果调用者已经在 decoder layer 的 attention 边界完成了分片，则通过 `already_sequence_parallel=True` 避免重复 chunk。MoE 独立调用路径完成后才会 all-gather；而 Qwen3.5 的 MoE decoder layer 走的是 already-SP 路径。[qwen3_next.py](../vllm/vllm/model_executor/models/qwen3_next.py#L102-L117)、[qwen3_next.py](../vllm/vllm/model_executor/models/qwen3_next.py#L166-L227)

decoder layer 在 attention 后先对 hidden states 做 TP reduce-scatter，并对 residual 做同样的 sequence chunk，然后调用 post-attention RMSNorm 和已经是 sequence-parallel 输入的 MoE。[qwen3_next.py](../vllm/vllm/model_executor/models/qwen3_next.py#L491-L553) Qwen3.5 的 linear attention 和 full attention 都复用这个边界判断；区别只在 attention 实现，不改变 MoE-SP 的通信位置。[qwen3_5.py](../vllm/vllm/model_executor/models/qwen3_5.py#L127-L159)

当下一层不是 SP-MoE layer，或者模型到达最终 norm 时，`Qwen3NextModel` 才把 hidden states 和 residual 合并后 all-gather，并截断掉 padding token。[qwen3_next.py](../vllm/vllm/model_executor/models/qwen3_next.py#L665-L710)

### 方案特点

- SP 只服务于稀疏 MoE 层，不是把整个 Qwen3.5 hybrid backbone 的每一层都永久保持 token shard。
- attention 输出的 reduce-scatter 是进入 MoE 的前置边界；MoE 本身不再重复 chunk。
- shared expert 也按 SP 方式构造，避免 routed experts 是 shard、shared expert 却重复处理完整 token。
- PP size 大于 1 时该路径被关闭；源码明确把 PP-SP 边界留作后续工作。

## 3. Kimi-K2.5：多模态外壳 + DeepseekV2 文本 SP

### 先区分两个 Kimi 文件

当前 `KimiK25ForConditionalGeneration` 是多模态外壳。它在构造语言模型时明确调用 `init_vllm_registered_model(... architectures=["DeepseekV2ForCausalLM"])`，所以 Kimi-K2.5 的文本 backbone 使用 DeepseekV2 的模型层。[kimi_k25.py](../vllm/vllm/model_executor/models/kimi_k25.py#L328-L377)

`kimi_linear.py` 中虽然也有一个 `KimiMoE`，但该实现没有接入 `use_sequence_parallel_moe`，不能把它当作 Kimi-K2.5 当前实际使用的 SP 证据。Kimi-K2.5 的实际 SP 入口应沿 `kimi_k25.py` → `DeepseekV2ForCausalLM` → `DeepseekV2DecoderLayer` 追踪。

### 复用的 DeepseekV2-SP

DeepseekV2 的 `DeepseekV2MoE` 保存全局 `parallel_config.use_sequence_parallel_moe`，并把它传给 shared expert 与 `FusedMoE`；MoE forward 在需要时先 chunk token，执行专家后再按调用路径决定是否 all-gather。[deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py#L276-L298)、[deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py#L396-L425)

在 decoder layer 中，若上一层已经产生 token shard，先 all-gather 回完整 token 序列以执行当前层 attention；attention 输出随后 pad 并 reduce-scatter，residual 同步 chunk，之后 MLP/MoE 以 `already_sequence_parallel=True` 执行。[deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py#L1271-L1332)

模型循环会在非 SP-MoE layer 前和最终输出前 all-gather hidden states/residual；这保证了 dense layer、辅助 hidden state 和最终 norm 看到完整 token 序列。[deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py#L1449-L1492)

### 方案特点

- Kimi-K2.5 的视觉塔/多模态 projector 与语言 MoE-SP 是两条不同并行路径；本文只讨论语言 backbone 的 SP。
- Kimi-K2.5 的 SP 不是 Kimi 专属新实现，而是复用 DeepseekV2 的 MLA + MoE-SP 边界。
- 不能仅看到 `KimiK25ForConditionalGeneration` 就推断其使用 `KimiMoE`；当前实际语言模型初始化代码已经给出相反证据。

## 4. GLM-5.2：`glm_moe_dsa` 复用 DeepseekV2-SP

当前模型注册把 `GlmMoeDsaForCausalLM` 映射到 `deepseek_v2.GlmMoeDsaForCausalLM`；这个类本身只是 `DeepseekV2ForCausalLM` 的子类，没有重写 forward 或 MoE-SP。[registry.py](../vllm/vllm/model_executor/models/registry.py#L90-L117)、[deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py#L1911-L1920)

GLM-5.2 的 `model_type` 是 `glm_moe_dsa`。上游对它做了 DSA/配置兼容处理，例如将 router dtype 固定为 FP32，并把该 model type 纳入 MLA 配置识别；这些是模型算子/配置差异，不会改变 DeepseekV2 的 MoE-SP token 流。[deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py#L120-L128)、[config.py](../vllm/vllm/transformers_utils/config.py#L255-L270)

因此 GLM-5.2 的 SP 时序与 Kimi-K2.5 的文本 backbone 相同：

```text
[完整 hidden states]
        |
        v
[DSA/MLA attention，输出 TP-replicated]
        |
        v
[pad + TP reduce-scatter hidden；chunk residual]
        |
        v
[DeepseekV2MoE：SP token shard + EP dispatch/combine]
        |
        v
[继续保持 shard，遇到非 SP-MoE 或 final norm 时 all-gather]
```

需要注意，当前 registry 中 GLM-5.2 的实际入口是 `deepseek_v2.py`；不能仅因为仓库中存在 `vllm/models/deepseek_v32/` 就把该目录当成 GLM-5.2 的当前运行入口。

## 5. DeepSeek V4：当前是 EP/MegaMoE，不是通用 MoE-SP

### 模型层没有接入通用 SP

DS V4 通过独立的 `DeepseekV4ForCausalLM` 实现注册。[registry.py](../vllm/vllm/model_executor/models/registry.py#L90-L95)、[deepseek_v4/__init__.py](../vllm/vllm/models/deepseek_v4/__init__.py#L14-L38)

其 decoder layer 直接创建 `DeepseekV4MoE`，没有像 DeepseekV2/Qwen3Next 那样保存 `use_sequence_parallel_moe`、在 attention 后做 reduce-scatter，也没有把该开关传给 `FusedMoE`。[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L794-L817)

普通 fused-MoE 路径创建 `FusedMoE` 时也没有传 `is_sequence_parallel=True`；因此 `FusedMoE` 的默认 SP 参数仍为 false。[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L647-L691)、[layer.py](../vllm/vllm/model_executor/layers/fused_moe/layer.py#L120-L130)

`DeepseekV4MLP` 的构造函数确实保留了一个 `is_sequence_parallel` 参数，并在为 true 时使用 replicated weights、关闭线性层 TP；但当前 `DeepseekV4MoE` 创建 shared expert 时没有传入这个参数。这个 helper 能表达“SP 下的 MLP 形态”，不等于 DS V4 当前 forward 已经接入 SP。[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L82-L125)、[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L585-L603)

### MegaMoE 路径的实际重点

当 `moe_backend == "deep_gemm_mega_moe"` 时，DS V4 要求开启 `--enable-expert-parallel`；MegaMoE 根据 EP group 计算每个 rank 的本地物理专家范围，并用 DeepGEMM 的对称 buffer 执行专家计算。[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L512-L531)、[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L605-L645)、[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L357-L383)

DS V4 的 forward 是 router → top-k → 本地 MegaMoE experts → shared expert 合并；代码中没有通用 MoE-SP 的 `sequence_parallel_chunk`、TP reduce-scatter 或 model-level all-gather 边界。[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L693-L753)

当前文件中 `parallel_config.use_sequence_parallel_moe` 出现于权重加载时的 block-FP8 shared-expert padding 判断，而不是 forward 的 token 通信路径。[deepseek_v4/nvidia/model.py](../vllm/vllm/models/deepseek_v4/nvidia/model.py#L1163-L1184)

### 结论

对 DS V4，准确表述应是：

> 当前上游提供 DS V4 的 EP/MegaMoE 路径，但没有把通用 `use_sequence_parallel_moe` 接入 DS V4 decoder 的 MoE token 流。因此不能仅依据 `--enable-expert-parallel` 或 `DeepseekV4MLP` 中存在 `is_sequence_parallel` 参数，就声称 DS V4 已支持同样的 MoE-SP。

## 6. 四者方案的统一比较

| 比较项 | DS V4 | Qwen3.5 MoE | Kimi-K2.5 | GLM-5.2 |
| --- | --- | --- | --- | --- |
| 模型层是否读取 `use_sequence_parallel_moe` | forward 不读取；仅 loader 有相关判断 | 是 | 是，继承 DeepseekV2 | 是，继承 DeepseekV2 |
| 是否调用 `sequence_parallel_chunk` | 当前 DS V4 forward 没有 | 由 decoder layer 先完成 residual chunk；MoE 用 `already_sequence_parallel` | DeepseekV2 decoder 完成 residual chunk | DeepseekV2 decoder 完成 residual chunk |
| attention→MoE 边界 | 没有通用 SP reduce-scatter | attention 输出 TP reduce-scatter | attention 输出 TP reduce-scatter | attention 输出 TP reduce-scatter |
| MoE 实现 | 普通 `FusedMoE` 或 MegaMoE | `Qwen3NextSparseMoeBlock` + `FusedMoE` | `DeepseekV2MoE` + `FusedMoE` | `DeepseekV2MoE` + `FusedMoE` |
| SP token shard 的回收 | 不适用通用 SP 流程 | 非 SP 层/最终 norm 前 all-gather | 非 SP 层/最终 norm 前 all-gather | 非 SP 层/最终 norm 前 all-gather |
| PP 限制 | MegaMoE 自身有独立 EP 约束；无通用 MoE-SP | `pipeline_parallel_size == 1` 才启用 | `pipeline_parallel_size == 1` 才启用 | `pipeline_parallel_size == 1` 才启用 |

## 7. 实际判断方法

分析任意新 MoE 模型是否真的支持这套 SP，建议按以下顺序查源码：

1. 查模型注册入口，确认最终实例化的是哪个 model class，不要只看 Hugging Face 模型名称。
2. 查 decoder layer 是否根据 `parallel_config.use_sequence_parallel_moe` 设置 layer-level 开关。
3. 查 `FusedMoE(... is_sequence_parallel=...)` 是否显式传入该开关；只看到 `enable_expert_parallel` 不能证明有 SP。
4. 查 attention 后是否有 pad + `tensor_model_parallel_reduce_scatter`，以及 residual 是否调用 `sequence_parallel_chunk`。
5. 查模型循环何时 `tensor_model_parallel_all_gather`；如果 MoE 输出仍是 token shard，后续消费者必须能接受 shard，或在边界处显式 gather。
6. 如果模型走自定义 MegaMoE/自定义 expert kernel，单独确认其 EP dispatch、local expert ownership 和输出布局，不能套用 `FusedMoE` 的 SP 结论。

## 验证范围

本文结论来自当前上游源码静态核对和注册关系追踪，未宣称四个模型在 Ascend NPU 上已经完成运行时 E2E 或性能验证。`use_sequence_parallel_moe` 的具体 all2all kernel 行为还会随 `all2all_backend`、EP/TP/DP 拓扑和硬件后端变化；若要验证运行结果，应另外做 SP on/off 的数值 E2E，并检查实际 after-graph 和通信 trace。
