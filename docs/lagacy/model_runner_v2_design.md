# vLLM 上游 Model Runner V2 设计

本文基于当前工作区上游 `vllm/` 子模块的静态源码整理，目标是说明 Model Runner V2（下文简称 MRV2）如何被选择、如何接收调度结果、如何构造一次模型执行，以及如何把采样结果交还给 Scheduler。当前上游 checkout：`7b3d595eb197d714052ce296cc8b124f0dc8af31`。

## 一、先给结论

**目的（Why）**：MRV2 是 vLLM V1 Engine/Scheduler 下的一套新的 GPU 执行实现，主要把请求状态、输入缓冲区、KV/Attention 元数据、模型 forward 和采样拆成更明确的生命周期，并为 CUDA Graph、PP、DP、speculative decoding 和异步 D2H 输出提供统一执行状态。

**输入（Input）**：Scheduler 每一步产生一个 `SchedulerOutput`，其中包括新请求数据、已缓存请求的增量、每个请求要执行的 token 数、KV block 变更、完成/抢占请求以及 speculative tokens。MRV2 由 `VllmConfig.use_v2_model_runner` 选择，默认由模型/特性能力判定，也可以通过 `VLLM_USE_V2_MODEL_RUNNER` 显式覆盖。

**处理（Logic）**：`GPUWorker.execute_model()` 先把 PP 输入接收好，再调用 MRV2 的 `execute_model()`；MRV2 更新持久化请求状态，生成 `InputBatch` 和 Attention metadata，按 `FULL` CUDA Graph、`PIECEWISE` 或 eager 执行模型。最后一个 PP rank 把隐藏状态保存在 `ExecuteModelState`，随后由独立的 `sample_tokens()` 完成 logits、grammar mask、采样、请求状态更新和 speculative proposal。

**输出（Output）**：普通生成返回 `AsyncOutput`，其中 GPU 上的 sampled token、logprobs 和计数在独立 copy stream 上异步复制到 CPU，`get_output()` 等待事件后形成 `ModelRunnerOutput`。非最后 PP rank 返回 `IntermediateTensors` 给下一 rank；pooling 模型走 `pool()`。

**流向（Flow）**：`EngineCore.step()` 调度并异步调用 `execute_model()`，同时计算 grammar bitmask；若 execute 阶段返回 `None`，立即调用 `sample_tokens()`；最终 `Scheduler.update_from_output()` 消费 `ModelRunnerOutput`，推进请求状态和下一轮调度。

## 二、名称和代码边界

这里的“V2”不是新的 Engine 版本。当前代码仍在 `vllm/v1` 命名空间下：

| 角色 | MRV2 源码 | 作用 |
| --- | --- | --- |
| Engine loop | [`vllm/vllm/v1/engine/core.py`](../vllm/vllm/v1/engine/core.py#L576-L606) | 调度、执行、采样、回写 Scheduler |
| 调度输出 | [`vllm/vllm/v1/core/sched/output.py`](../vllm/vllm/v1/core/sched/output.py#L190-L260) | 跨 Scheduler/Worker 传递一步的计划 |
| GPU Worker 入口 | [`vllm/vllm/v1/worker/gpu_worker.py`](../vllm/vllm/v1/worker/gpu_worker.py#L1000-L1090) | 处理 PP 通信并转发到 runner |
| MRV2 | [`vllm/vllm/v1/worker/gpu/model_runner.py`](../vllm/vllm/v1/worker/gpu/model_runner.py#L120-L256) | 请求状态、输入、forward、sampling |
| MRV1/legacy | [`vllm/vllm/v1/worker/gpu_model_runner.py`](../vllm/vllm/v1/worker/gpu_model_runner.py#L447-L490) | 旧 GPU runner 实现 |
| 输出契约 | [`vllm/vllm/v1/outputs.py`](../vllm/vllm/v1/outputs.py#L231-L307) | `ModelRunnerOutput` / 异步输出接口 |

`GPUWorker` 根据同一个接口选择实现：MRV2 导入 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`，旧实现导入 `vllm.v1.worker.gpu_model_runner.GPUModelRunner`。[`GPUWorker.__init__()`](../vllm/vllm/v1/worker/gpu_worker.py#L174-L176) 保存开关，[`GPUWorker.init_device()`](../vllm/vllm/v1/worker/gpu_worker.py#L401-L416) 完成实际选择。

## 三、MRV2 的启用决策

### 3.1 配置到 runner 的链路

```text
VLLM_USE_V2_MODEL_RUNNER
          |
          v
envs.VLLM_USE_V2_MODEL_RUNNER
          |
          v
VllmConfig.use_v2_model_runner
          |
          +--> GPUWorker.model_runner = MRV2 / MRV1
          +--> Scheduler 的 V2 调度分支
          +--> InputProcessor 的能力检查
```

环境变量默认是 `None`，表示不强制选择；配置属性优先处理显式环境变量，然后对 DSpark、混合 sliding/full DFlash、diffusion 等 V2-only 情况强制选择，再对默认架构、Triton 和不支持特性进行判断。没有显式强制时，不满足能力条件会回退 MRV1；配置最终校验阶段若已经选择 MRV2，则缺少 Triton 或存在不支持特性会直接抛错。[环境变量定义](../vllm/vllm/envs.py#L274-L275)、[环境变量读取](../vllm/vllm/envs.py#L1921-L1924)、[选择逻辑](../vllm/vllm/config/vllm.py#L549-L591)。

### 3.2 当前主要限制

配置层集中列出了 MRV2 尚不支持的特性，包括：prefill context parallelism、stock `torch.compile`、TP>1 下的 sequence parallelism、external launcher 下的 PP、ngram speculative decoding、DBO、elastic EP、custom logits processors、prompt embeds、部分 logprobs mode、KV sharing fast prefill 和 EC transfer 等。[完整检查列表](../vllm/vllm/config/vllm.py#L2127-L2220)。因此“把环境变量设为 1”不是无条件开启：配置验证仍会在明确选择 MRV2 后拒绝不兼容组合。[校验函数](../vllm/vllm/config/vllm.py#L2222-L2237)。

## 四、核心对象：持久状态、单步输入和跨阶段状态

### 4.1 持久状态

MRV2 初始化时从 `VllmConfig` 保存 model/cache/parallel/scheduler/speculative 等配置，并创建以下长期存在的对象：

- `RequestState`：按 runner 内部 index 保存 request token、`num_computed_tokens`、prefill 状态、采样状态和 draft token。
- `InputBuffers`：复用 GPU input、position、padding、query-start 等缓冲区，减少每步重新分配。
- `BlockTables`、KV cache 和 `KVConnector`：把 Scheduler 分配的 block IDs 绑定到 Attention/KV 实现。
- `ModelState`：模型相关的 multimodal、Attention、Mamba/hybrid 等状态适配。
- 最后 PP rank 上的 `Sampler`、`RejectionSampler`、`PromptLogprobsWorker` 和 `StructuredOutputsWorker`。
- `execute_model_state`：把一次 forward 产生的 hidden states 和 Attention 元数据留给紧接着的 `sample_tokens()`。

这些对象的初始化和 ownership 位于 [MRV2 构造函数](../vllm/vllm/v1/worker/gpu/model_runner.py#L120-L256)；模型加载后才创建 model-dependent state 和 sampler，见 [`load_model()`](../vllm/vllm/v1/worker/gpu/model_runner.py#L273-L369)。KV cache 初始化会建立 block table、attention backend、caches、connector 以及 V2 的 CUDA Graph manager，见 [`initialize_kv_cache()`](../vllm/vllm/v1/worker/gpu/model_runner.py#L405-L502)。

### 4.2 SchedulerOutput：Scheduler 只发送变化量

`SchedulerOutput` 的关键设计是：新请求发送完整的 `NewRequestData`，已存在请求只发送 `CachedRequestData` 的增量；注释明确说明 Worker 会缓存请求数据，以减少每一步通信。[`SchedulerOutput` 字段](../vllm/vllm/v1/core/sched/output.py#L190-L256)。

MRV2 的新请求额外携带 `prefill_token_ids`，用于在 Worker 的 `RequestState` 中建立完整 token 状态；Scheduler 在 V2 分支把 resumed requests 合并到 new requests，并通过 `NewRequestData.from_request(..., req._all_token_ids)` 填充该字段。[字段定义](../vllm/vllm/v1/core/sched/output.py#L33-L67)、[Scheduler 的 V2 构造分支](../vllm/vllm/v1/core/sched/scheduler.py#L1080-L1107)。相对地，`CachedRequestData.all_token_ids` 标注为 MRV1-only；`preempted_req_ids` 标注为 MRV2-only，表明抢占后的 runner 状态回收由 MRV2 自己处理。[`CachedRequestData` 注释](../vllm/vllm/v1/core/sched/output.py#L113-L128)、[V2 专用字段](../vllm/vllm/v1/core/sched/output.py#L227-L235)。

### 4.3 InputBatch：把 request-level 计划变成 token-level 执行输入

`prepare_inputs()` 的主要转换如下：

1. 按 decode、short extend、prefill 对 request 排序，生成 `req_ids`、runner index 映射和每请求 scheduled token 数。
2. 根据 speculative tokens 计算 expanded mapping、logits 数量和 `cu_num_logits`。
3. 生成 `query_start_loc`，准备 prefill token、position、sequence length 和 DCP local sequence length。
4. 从 last sampled token、prefill token 和 draft token 组合本轮 `input_ids`，同时计算 `logits_indices`，决定哪些 hidden states 进入采样。
5. 把 padding 后 token 数、Attention 所需序列信息、structured-output 标志等封装成 `InputBatch`。[`prepare_inputs()`](../vllm/vllm/v1/worker/gpu/model_runner.py#L861-L1039)。

因此 `InputBatch` 不是 Scheduler 的原始输出，也不是单纯的 token tensor；它是“请求持久状态 + 本轮调度计划 + padding/graph 描述”组合后的执行视图。

## 五、一次执行的完整时序

```text
[Scheduler.schedule()]
       | SchedulerOutput
       v
[EngineCore.step()]
       | execute_model(non_block=True)
       v
[GPUWorker.execute_model()]
       | PP irecv / intermediate tensors
       v
[MRV2.execute_model()]
       | 更新 RequestState / BlockTables
       | dispatch_cg_and_sync_dp
       | prepare_inputs + prepare_attn
       | model forward: FULL / PIECEWISE / EAGER
       v
 [最后 PP rank: ExecuteModelState(hidden_states)]
       | GrammarOutput
       v
[MRV2.sample_tokens()]
       | logits -> grammar mask -> sampler/rejection sampler
       | postprocess request state / speculative proposal
       v
[AsyncOutput.get_output()]
       | CPU sampled tokens / logprobs
       v
[Scheduler.update_from_output()]
```

### 5.1 Engine 和 Worker 的两阶段契约

`EngineCore.step()` 先调用 `model_executor.execute_model(..., non_block=True)`，再取得 grammar bitmask；当 forward 没有直接返回最终 output 时，调用 `sample_tokens(grammar_output)`，最后把结果交给 Scheduler。[Engine 主循环](../vllm/vllm/v1/engine/core.py#L576-L606)。抽象 Worker 接口也明确规定：`execute_model()` 返回 `None` 时，必须紧接着调用 `sample_tokens()`。[接口定义](../vllm/vllm/v1/worker/worker_base.py#L142-L157)。

GPU Worker 在此之上处理 PP：非首 rank 先异步接收前一 rank 的 `IntermediateTensors`；调用 runner 后，如果得到的是中间 tensor，则非阻塞发送给下一 rank，否则把 `ModelRunnerOutput`/异步 output 原样返回。[PP 接收和 runner 调用](../vllm/vllm/v1/worker/gpu_worker.py#L1047-L1090)。

### 5.2 `execute_model()`：从计划到 hidden states

MRV2 的 forward 阶段顺序是：

1. 非 dummy run 时，先 `update_pp_decode_requests()`，清理 finished/preempted request，添加新请求，更新 cached requests 和 block table；无 scheduled token 时直接走 KV connector 的 no-forward 路径。[状态更新入口](../vllm/vllm/v1/worker/gpu/model_runner.py#L1133-L1153)。
2. 计算请求数、token 数、uniform token count，并调用 `dispatch_cg_and_sync_dp()`，让 DP rank 对 batch descriptor 和 token 数达成一致。[batch dispatch](../vllm/vllm/v1/worker/gpu/model_runner.py#L1155-L1189)。
3. 真实 batch 调用 `prepare_inputs()`、`prepare_attn()` 和 model-state pre-attention；dummy batch 则创建符合 graph shape 的 dummy inputs。[输入/Attention 准备](../vllm/vllm/v1/worker/gpu/model_runner.py#L1191-L1247)。
4. 首 PP rank 处理 multimodal embeddings，所有 rank 由 `model_state.prepare_inputs()` 组合模型参数；非首 rank 把接收到的 intermediate tensors 拷贝到 persistent capture buffer。[模型输入构造](../vllm/vllm/v1/worker/gpu/model_runner.py#L1249-L1304)。
5. `FULL` 直接 replay 完整 graph；`PIECEWISE` 调用 graph manager；`NONE` 在 forward context 下直接调用 `self.model(**model_inputs)`。[执行模式分支](../vllm/vllm/v1/worker/gpu/model_runner.py#L1307-L1345)。
6. 最后 PP rank 保存 hidden states，非最后 rank 保存 `IntermediateTensors`；两者都写入 `ExecuteModelState`，随后最后 rank 返回 `None`，非最后 rank返回中间 tensor。[跨阶段状态](../vllm/vllm/v1/worker/gpu/model_runner.py#L1346-L1374)。

### 5.3 `sample_tokens()`：采样、回写和异步输出

最后 PP rank 从 `ExecuteModelState` 取出 hidden states，根据 `logits_indices` 选择采样位置，调用 `model.compute_logits()`，应用 grammar bitmask，再选择普通 sampler 或 speculative rejection sampler。[logits/采样](../vllm/vllm/v1/worker/gpu/model_runner.py#L1068-L1100)。之后 MRV2：

- 创建兼容 Scheduler 的 `ModelRunnerOutput`，其中 `req_id_to_index` 当前只是兼容字段；
- 在 `AsyncOutput` 构造阶段，让 copy stream 等待 main stream，并非阻塞复制 sampled tokens、logprobs、NaN 计数和 prompt logprobs；
- 在异步复制已经排队后更新 `num_computed_tokens`、采样历史和 model state，再执行 speculative proposal；
- 完成 KV connector 的 post-forward，并返回 async output。[采样后处理](../vllm/vllm/v1/worker/gpu/model_runner.py#L1411-L1514)。

`AsyncOutput.get_output()` 才会同步 copy event，把 GPU/CPU tensor 转成 Scheduler 兼容的 Python/NumPy 结构；这使 D2H 可以与后续 proposal 或其它 GPU 工作重叠。[异步 output](../vllm/vllm/v1/worker/gpu/async_utils.py#L12-L70)。`AsyncModelRunnerOutput` 的契约明确：`get_output()` 是可能阻塞、且每个 output 只能调用一次的消费点。[输出接口](../vllm/vllm/v1/outputs.py#L231-L307)。

非最后 PP rank 不执行采样，而是接收最后 rank 广播的 sampled tokens，乐观更新 computed tokens，再只返回 KV connector output；pooling 模型则由 `pool()` 消费同一份 `ExecuteModelState`。[非最后 rank 与 pooling 分支](../vllm/vllm/v1/worker/gpu/model_runner.py#L1393-L1410)、[`pool()`](../vllm/vllm/v1/worker/gpu/model_runner.py#L1519-L1558)。

## 六、CUDA Graph、warmup 和并发模型

MRV2 在 KV cache 初始化阶段根据 attention backend 支持度和 decode query length 解析 graph mode，并创建 `ModelCudaGraphManager`。[graph manager 初始化](../vllm/vllm/v1/worker/gpu/model_runner.py#L459-L474)。Worker 初始化 KV cache 后，如果选择 MRV2，会调用 `warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)`，即用完整的 forward+sample 链路预热 Triton/kernel，而不是只单独预热 sampler。[V2 warmup](../vllm/vllm/v1/worker/gpu_worker.py#L861-L868)。

V2 的 async scheduling/PP 并发数也由配置显式区分：`max_concurrent_batches` 在 V2 下按 `pp_size + 1` 计算，以便 PP pipeline 和异步重叠；旧 runner 对 PP async 有更严格限制。[并发配置](../vllm/vllm/config/vllm.py#L511-L533)。这也是为什么 MRV2 的输入缓冲区、persistent intermediate tensors 和 `ExecuteModelState` 必须支持多个 in-flight batch，不能把一次 step 的 tensor 生命周期简单等同于一次 Python 调用。

## 七、与 MRV1 的关键差异

| 维度 | MRV2 | MRV1/legacy |
| --- | --- | --- |
| 实现位置 | `v1/worker/gpu/model_runner.py` | `v1/worker/gpu_model_runner.py` |
| 请求数据 | Worker 持久缓存 `RequestState`，新请求携带 `prefill_token_ids`，cached request 主要发增量 | 仍保留更多旧式 per-step/request bookkeeping；Scheduler 明确保留 `all_token_ids` 的 MRV1 路径 |
| forward/sample | `execute_model()` 保存 `ExecuteModelState`，`sample_tokens()` 后置消费 | 也实现共同 Worker 接口，但内部状态和兼容逻辑不同 |
| PP 输出 | V2 runner 以 `IntermediateTensors` 或 `None + sample_tokens()` 配合 PP handler | Worker 中仍存在只支持 MRV1 的 sequence-parallel PP 预处理分支；见 [`GPUWorker.execute_model()`](../vllm/vllm/v1/worker/gpu_worker.py#L1018-L1045) |
| 输出搬运 | `AsyncOutput` 在 copy stream 上异步 D2H | legacy runner 使用自己的 async output/采样路径 |
| graph/warmup | `use_v2_model_runner=True` 参与 graph mode 解析，warmup 覆盖 forward+sample | legacy 有单独的 dummy run、sampler buffer 和兼容分支 |

因此扩展 MRV2 时，不能只修改 `gpu_model_runner.py` 期待自动生效；必须先确认 `GPUWorker` 实际选择的 runner，再沿 `SchedulerOutput → RequestState/InputBatch → ExecuteModelState → ModelRunnerOutput` 链路修改对应层。

## 八、给硬件后端/插件开发的阅读边界

上游 MRV2 的硬件抽象边界主要是 `GPUWorker`、attention backend、model state、KV connector、CUDA graph manager 和模型自身的 `forward/compute_logits`。如果硬件插件仍实现自己的 `worker/model_runner_v1.py`，它属于插件侧 V1 runner，不等价于上游 `vllm/v1/worker/gpu/model_runner.py`；是否接入 MRV2，需要单独实现或适配上游 Worker/runner 接口，不能仅依据“V1 Engine”这个命名判断。

静态源码能证明调用关系、字段所有权和分支条件；它不能证明特定硬件上的 kernel 是否真正执行、CUDA Graph/后端 graph 是否成功 capture，也不能替代功能 E2E、数值精度和性能 profiling。

## 九、最小排查清单

遇到“MRV2 没有生效”或输出异常时，按以下顺序定位：

1. 检查 `VllmConfig.use_v2_model_runner` 的最终值，而不是只看环境变量。
2. 确认 `GPUWorker` 实际 import 的 runner 路径和启动日志 `Using V2 Model Runner`。
3. 对照 `SchedulerOutput.num_scheduled_tokens`、`scheduled_new_reqs`、`preempted_req_ids`，确认请求增量是否完整。
4. 检查 `InputBatch.req_ids` 排序、`query_start_loc`、padding 后 token 数和 `logits_indices` 是否一致。
5. 区分 `execute_model()` 的 forward 问题、`sample_tokens()` 的 logits/grammar/sampler 问题和 `AsyncOutput.get_output()` 的 D2H/同步问题。
6. 多卡场景分别检查 PP intermediate tensor、最后 rank 的 token broadcast、DP batch descriptor 同步和 KV connector 的 pre/post-forward。

## 十、数据流架构图（精简版）

```text
[请求 + SamplingParams]
          |
          v
[Scheduler.schedule()]
          | SchedulerOutput: req/token/block 增量
          v
[EngineCore.step()]
          | execute_model + GrammarOutput
          v
[GPUWorker]
          | PP recv/send
          v
[MRV2.execute_model()]
          | RequestState -> InputBatch -> Attn metadata
          | FULL / PIECEWISE / EAGER model forward
          v
[ExecuteModelState]
          | hidden states / intermediate tensors
          v
[MRV2.sample_tokens()]
          | logits -> grammar -> sampler -> state update
          v
[AsyncOutput.get_output()]
          | sampled tokens / logprobs / KV connector output
          v
[Scheduler.update_from_output()]
```

## 十一、代码定位索引

1. [`vllm/vllm/config/vllm.py:549-591`](../vllm/vllm/config/vllm.py#L549-L591) — `VllmConfig.use_v2_model_runner`：环境变量、强制场景、默认模型和能力回退。
2. [`vllm/vllm/v1/worker/gpu_worker.py:401-416`](../vllm/vllm/v1/worker/gpu_worker.py#L401-L416) — `GPUWorker.init_device()`：选择 MRV2 或 legacy runner。
3. [`vllm/vllm/v1/core/sched/output.py:190-256`](../vllm/vllm/v1/core/sched/output.py#L190-L256) — `SchedulerOutput`：一步调度计划和 V2 专用状态字段。
4. [`vllm/vllm/v1/worker/gpu/model_runner.py:861-1039`](../vllm/vllm/v1/worker/gpu/model_runner.py#L861-L1039) — `prepare_inputs()`：请求计划到 token-level `InputBatch`。
5. [`vllm/vllm/v1/worker/gpu/model_runner.py:1133-1374`](../vllm/vllm/v1/worker/gpu/model_runner.py#L1133-L1374) — `execute_model()`：状态更新、batch dispatch、输入准备、forward 和跨阶段状态。
6. [`vllm/vllm/v1/worker/gpu/model_runner.py:1378-1514`](../vllm/vllm/v1/worker/gpu/model_runner.py#L1378-L1514) — `sample_tokens()`：采样、状态回写、speculative proposal 和 async output。
7. [`vllm/vllm/v1/engine/core.py:576-606`](../vllm/vllm/v1/engine/core.py#L576-L606) — `EngineCore.step()`：Engine 对 execute/sample/output 的编排。

以上行号是当前工作区 checkout 的静态定位；源码变更后应重新用 `rg -n` 和 `nl -ba` 核对。
