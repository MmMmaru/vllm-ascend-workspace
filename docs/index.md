# vLLM Workspace 文档与代码导航

本文档是 `/home/x50063850/vllm-workspace` 的宏观索引。它列出主要子项目、关键子目录和常用入口文件，帮助定位 Engine、Worker、模型、TP/PP、FlashComm、NPU custom op、多模态以及测试代码。

说明：目录树中省略 `__pycache__`、构建产物、缓存和 generated 文件；关键文件路径以当前工作区实际代码为准。

换仓后的容器编译、custom-op 重建、SP benchmark 和 E2E 验证流程见
[`container_compile_flow.md`](container_compile_flow.md)。

## 1. 工作区总览

```text
vllm-workspace/
├── vllm/                         上游 vLLM 主项目
├── vllm-ascend/                  Ascend NPU 插件项目
├── Mooncake/                     独立的 KV/传输相关项目（如任务涉及再进入）
├── scripts/                      当前工作区的启动、调试、请求脚本
├── docs/                         当前工作区的专题文档和索引
├── .vscode/                      VS Code 调试配置
├── logs/                         运行日志目录
├── extra-info/                   额外环境/问题记录
├── AGENTS.md                     开发前的协作说明
├── CONFIG.md                     容器、权重和运行环境说明
├── vllm-workspace.code-workspace  VS Code 工作区文件
└── TODO.md                       当前待办事项
```

工作区中的 [`Mooncake/`](../Mooncake/) 是独立的 KV cache/数据传输项目；它不属于 vLLM Python 主调用链，但在 KV transfer、分离式推理和传输性能问题中可能被同时查看：

```text
Mooncake/
├── mooncake-transfer-engine/       Transfer Engine（C++/Rust/示例/测试）
├── mooncake-store/                 Store 服务、客户端和测试
├── mooncake-common/                公共 C++ 组件、etcd 和测试
├── mooncake-integration/           与 EP、Store、Transfer Engine 的集成
├── mooncake-ep/                    Expert Parallel 相关组件
├── mooncake-p2p-store/             P2P Store
├── mooncake-pg/                    PG 组件
├── mooncake-asio/                  Asio 网络组件
├── mooncake-wheel/                 Python wheel 封装
├── benchmarks/                     传输/Store 基准
├── docs/source/                    架构、部署、API 和故障排查文档
├── monitoring/                     Prometheus/Grafana 监控
├── image/                          文档图片
└── scripts/                        运行和测试脚本
```

## 2. 上游 `vllm/` 目录

`vllm/` 提供通用推理框架、调度、分布式并行、模型实现和 v1 执行链路。Ascend 插件通常通过继承、custom op 或 patch 接入这里的接口。

### 2.1 上游项目的主要子目录

```text
vllm/
├── vllm/                         Python 主包
│   ├── v1/                       v1 Engine/Worker/Executor/Attention 链路
│   ├── engine/                   传统 Engine 组件
│   ├── entrypoints/              CLI、OpenAI API、serve、generate 等入口
│   ├── distributed/              TP/PP/DP/EP 通信和并行状态
│   │   ├── device_communicators/ 设备通信、共享内存广播等
│   │   ├── elastic_ep/            弹性 Expert Parallel
│   │   ├── eplb/                  Expert load balancing
│   │   ├── kv_transfer/            KV 传输
│   │   └── weight_transfer/        权重传输
│   ├── model_executor/            模型执行框架
│   │   ├── layers/                 Linear、Attention、Embedding、Norm、MoE
│   │   ├── models/                 Qwen、DeepSeek 及其他模型 forward
│   │   ├── model_loader/           权重加载
│   │   ├── kernels/                模型执行 kernel
│   │   └── quantization/           量化接口和实现
│   ├── multimodal/                多模态 registry、processor、media 输入
│   ├── compilation/                torch.compile、graph、编译 pass
│   │   └── passes/                 编译期优化 pass
│   ├── config/                     vLLM 配置对象
│   ├── inputs/                     输入协议和输入处理
│   ├── transformers_utils/         Transformers 配置、processor、chat template
│   ├── platforms/                  硬件平台抽象
│   ├── kernels/                    Triton/Helion 等通用 kernel
│   ├── lora/                       LoRA 层和管理
│   ├── profiler/                   profiling
│   ├── parser/                     reasoning/tool parser
│   ├── utils/                      通用工具
│   ├── ir/                         中间表示和 IR ops
│   ├── models/                     跨模型公共组件
│   ├── renderers/                  输入渲染和模型输入准备
│   ├── plugins/                    IO processor、LoRA resolver 等插件
│   ├── tokenizers/                 tokenizer 封装
│   ├── tool_parsers/、reasoning/   工具调用和 reasoning parser
│   └── tracing/、usage/            tracing 与 usage 统计
├── tests/                          上游测试
├── benchmarks/                     基准和性能测试
├── examples/                       示例
├── docs/                           上游文档
├── requirements/                   依赖文件
├── scripts/                        上游构建/开发脚本
├── docker/                          上游 Docker 配置
├── cmake/                           CMake 构建配置
├── csrc/                           C/CUDA 源码
├── rust/                           Rust 组件
└── tools/                          工具和开发辅助
```

### 2.2 上游关键文件

| 关注内容 | 关键文件 |
| --- | --- |
| AsyncLLM 请求入口 | [`vllm/v1/engine/async_llm.py`](../vllm/vllm/v1/engine/async_llm.py) |
| EngineCore 调度与执行 | [`vllm/v1/engine/core.py`](../vllm/vllm/v1/engine/core.py) |
| Scheduler | [`vllm/v1/core/sched/scheduler.py`](../vllm/vllm/v1/core/sched/scheduler.py) |
| 多进程 Worker 启动和 RPC | [`vllm/v1/executor/multiproc_executor.py`](../vllm/vllm/v1/executor/multiproc_executor.py) |
| 通用 Worker | [`vllm/v1/worker/gpu_worker.py`](../vllm/vllm/v1/worker/gpu_worker.py) |
| 通用 ModelRunner | [`vllm/v1/worker/gpu_model_runner.py`](../vllm/vllm/v1/worker/gpu_model_runner.py) |
| Worker 辅助函数 | [`vllm/v1/worker/utils.py`](../vllm/vllm/v1/worker/utils.py) |
| TP/PP group、send/recv | [`vllm/distributed/parallel_state.py`](../vllm/vllm/distributed/parallel_state.py) |
| 共享内存广播 | [`vllm/distributed/device_communicators/shm_broadcast.py`](../vllm/vllm/distributed/device_communicators/shm_broadcast.py) |
| Qwen3 dense forward | [`vllm/model_executor/models/qwen3.py`](../vllm/vllm/model_executor/models/qwen3.py) |
| Qwen3-MoE forward | [`vllm/model_executor/models/qwen3_moe.py`](../vllm/vllm/model_executor/models/qwen3_moe.py) |
| Qwen3-VL 多模态 forward | [`vllm/model_executor/models/qwen3_vl.py`](../vllm/vllm/model_executor/models/qwen3_vl.py) |
| PP layer 切分工具 | [`vllm/model_executor/models/utils.py`](../vllm/vllm/model_executor/models/utils.py) |
| 词表并行 Embedding/LMHead | [`vllm/model_executor/layers/vocab_parallel_embedding.py`](../vllm/vllm/model_executor/layers/vocab_parallel_embedding.py) |
| Linear/TP 层 | [`vllm/model_executor/layers/linear.py`](../vllm/vllm/model_executor/layers/linear.py) |
| 多模态输入 registry | [`vllm/multimodal/`](../vllm/vllm/multimodal/) |
| torch.compile 装饰器 | [`vllm/compilation/decorators.py`](../vllm/vllm/compilation/decorators.py) |

### 2.3 上游 v1 执行子目录

```text
vllm/vllm/v1/
├── engine/       AsyncLLM、EngineCore、输入/输出处理
├── core/         block pool、KV cache、scheduler
├── executor/     uniprocess、multiprocess、Ray executor
├── worker/       Worker、GPU ModelRunner、InputBatch、PP 工具
├── attention/    attention backend、ops、paged attention
├── sample/       v1 sampling 和 logits 处理
├── spec_decode/  speculative decoding
├── metrics/      v1 指标
├── kv_offload/   KV offload
├── pool/         pooling runner
└── structured_output/  structured output
```

## 3. `vllm-ascend/` 目录

`vllm-ascend/` 是 Ascend 硬件插件，负责 NPU 平台、HCCL 通信、FlashComm、Ascend custom ops、NPU ModelRunner 以及针对模型/编译/量化的适配。

### 3.1 Ascend Python 包目录

```text
vllm-ascend/vllm_ascend/
├── worker/                       NPU Worker、ModelRunner、输入 batch
│   ├── v2/                       v2 ModelRunner 链路
│   ├── worker.py                 Worker 主循环、PP send/recv、NPU 初始化
│   ├── model_runner_v1.py        v1 preprocess、forward、padding、PP intermediate
│   ├── npu_input_batch.py         NPU 输入 buffer 和 token/position 管理
│   └── pcp_utils.py               PCP 辅助
├── _310p/                        310P 专用 attention/ops/MoE/量化/采样适配
├── _cann_ops_custom/             CANN custom op 源码和厂商注册目录
├── include/                      C++ kernel 头文件
├── lib/                          已构建的 Ascend runtime/kernel 库
├── xlite/                        XLite 相关适配
├── device_allocator/             NPU device allocator
├── ops/                          Ascend custom op 和算子封装
│   ├── fused_moe/                 MoE routing、prepare/finalize、expert dispatch
│   ├── triton/                    NPU/Triton 辅助 kernel
│   ├── register_custom_ops.py     reduce/all-gather/chunk residual 等注册
│   ├── vocab_parallel_embedding.py  Ascend 词表并行 Embedding/LMHead
│   ├── linear_op.py                Ascend TP/FlashComm Linear
│   ├── layernorm.py                RMSNorm/residual 处理
│   ├── mla.py                      MLA 和 FlashComm 首层处理
│   ├── linear.py、linear_op.py      TP/FlashComm Linear 相关算子
│   ├── dsa.py、mla.py               DSA/MLA 相关算子
│   ├── layer_shard_linear.py        layer-shard Linear
│   ├── flashcomm2_oshard_manager.py FlashComm2 OShard 管理
│   └── rotary_embedding.py         RoPE 和 position 处理
├── attention/                     NPU attention backend
│   ├── attention_v1.py             通用 attention v1
│   ├── mla_v1.py                   MLA
│   ├── sfa_v1.py、dsa_v1.py        SFA/DSA
│   ├── context_parallel/           CP attention
│   └── kvcomp_attn/                KV compression attention
├── distributed/                   Ascend 并行状态和通信
│   ├── parallel_state.py            Ascend 专用并行 group
│   ├── device_communicators/        HCCL/NPU communicator
│   ├── kv_transfer/                 Ascend KV transfer
│   └── weight_transfer/             权重传输
├── compilation/                   NPU graph 和编译优化
│   ├── acl_graph.py                 ACL graph
│   ├── compiler_interface.py        编译接口
│   ├── graph_fusion_pass_manager.py pass 管理
│   └── passes/                      sequence parallel、RMSNorm、RoPE 等融合
├── patch/                          对上游 vLLM 的平台/Worker patch
│   ├── platform/                    executor、scheduler、distributed、KV 等 patch
│   └── worker/                      模型、CUDAGraph、spec decode、NPU 行为 patch
├── models/                         Ascend 自定义模型/模型适配
├── model_executor/                 Ascend model executor 扩展
├── model_loader/                   Ascend 权重加载和网络加载
│   ├── netloader/                   网络权重加载
│   └── rfork/                       rfork 加载支持
├── quantization/                   Ascend 量化方法和适配器
│   └── methods/                     具体量化方法
├── spec_decode/                    Ascend speculative decoding
├── sample/                         Ascend sampler
├── eplb/                           Expert load balancing
├── lora/                           Ascend LoRA
├── kv_offload/                     KV offload
├── simple_kv_offload/               简化 KV offload
├── profiler/                       NPU profiling
├── device/                         NPU device 管理
├── core/                           Ascend core 组件
├── platform.py                     Ascend platform 注册和能力声明
├── ascend_config.py                additional_config/Ascend 配置
├── ascend_forward_context.py        FlashComm/graph/DP forward context
├── envs.py                         Ascend 环境变量集中定义
├── flash_common3_context.py        FlashCommon3 context
├── batch_invariant.py               batch-invariant 支持
├── profiling_config.py              profiling 配置
├── cpu_binding.py                   CPU 绑核辅助
├── logger.py                        Ascend 日志封装
├── utils.py                        enable_sp、FlashComm 和通用工具
└── meta_registration.py            NPU meta/fake 注册
```

### 3.2 Ascend 关键文件

| 关注内容 | 关键文件 |
| --- | --- |
| NPU Worker 主流程 | [`vllm_ascend/worker/worker.py`](../vllm-ascend/vllm_ascend/worker/worker.py) |
| v1 NPU ModelRunner | [`vllm_ascend/worker/model_runner_v1.py`](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py) |
| NPU 输入 batch | [`vllm_ascend/worker/npu_input_batch.py`](../vllm-ascend/vllm_ascend/worker/npu_input_batch.py) |
| FlashComm forward context | [`vllm_ascend/ascend_forward_context.py`](../vllm-ascend/vllm_ascend/ascend_forward_context.py) |
| FlashComm 开关和判断 | [`vllm_ascend/utils.py`](../vllm-ascend/vllm_ascend/utils.py) |
| Ascend 配置 | [`vllm_ascend/ascend_config.py`](../vllm-ascend/vllm_ascend/ascend_config.py) |
| 环境变量 | [`vllm_ascend/envs.py`](../vllm-ascend/vllm_ascend/envs.py) |
| PP+FlashComm receive/send | [`vllm_ascend/worker/worker.py`](../vllm-ascend/vllm_ascend/worker/worker.py) |
| PP+FlashComm slice/gather | [`vllm_ascend/worker/model_runner_v1.py`](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py) |
| custom op 注册 | [`vllm_ascend/ops/register_custom_ops.py`](../vllm-ascend/vllm_ascend/ops/register_custom_ops.py) |
| 词表并行 Embedding | [`vllm_ascend/ops/vocab_parallel_embedding.py`](../vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py) |
| TP/FlashComm Linear | [`vllm_ascend/ops/linear_op.py`](../vllm-ascend/vllm_ascend/ops/linear_op.py) |
| MoE prepare/finalize | [`vllm_ascend/ops/fused_moe/prepare_finalize.py`](../vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py) |
| MoE 主执行 | [`vllm_ascend/ops/fused_moe/fused_moe.py`](../vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py) |
| attention v1 | [`vllm_ascend/attention/attention_v1.py`](../vllm-ascend/vllm_ascend/attention/attention_v1.py) |
| MLA/SFA/DSA | [`vllm_ascend/attention/mla_v1.py`](../vllm-ascend/vllm_ascend/attention/mla_v1.py)、[`sfa_v1.py`](../vllm-ascend/vllm_ascend/attention/sfa_v1.py)、[`dsa_v1.py`](../vllm-ascend/vllm_ascend/attention/dsa_v1.py) |
| HCCL communicator | [`vllm_ascend/distributed/device_communicators/`](../vllm-ascend/vllm_ascend/distributed/device_communicators/) |
| 多进程 executor patch | [`vllm_ascend/patch/platform/patch_multiproc_executor.py`](../vllm-ascend/vllm_ascend/patch/platform/patch_multiproc_executor.py) |
| Qwen3-VL patch | [`vllm_ascend/patch/worker/patch_qwen3vl.py`](../vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py) |
| sequence-parallel 编译 pass | [`vllm/compilation/passes/fusion/sequence_parallelism.py`](../vllm/vllm/compilation/passes/fusion/sequence_parallelism.py) |

### 3.3 Ascend 辅助目录

```text
vllm-ascend/
├── tests/                 Ascend UT/E2E 测试
├── docs/source/           Ascend 用户文档、feature guide、support matrix
├── examples/              Ascend 示例
├── benchmarks/            Ascend benchmark
├── csrc/                  C++/NPU 源码
├── cmake/                 构建配置
├── tools/                 开发和环境工具
├── Dockerfile*            不同硬件/系统的容器构建文件
├── pyproject.toml         Python 项目配置
├── setup.py               安装入口
└── requirements*.txt      依赖文件
```

## 4. 当前工作区脚本与编辑器配置

```text
scripts/
├── vllm_run.sh                    普通 vLLM 服务启动
├── debug_vllm.sh                  debugpy + vLLM 服务启动
├── send_req_2vllm.sh              向服务端口发送请求
└── repro_issue_11548.sh           问题复现脚本

.vscode/
├── launch.json                    debugpy attach/launch 配置
└── settings.json                  Remote/工作区设置
```

调试端口时同时检查：

```text
scripts/debug_vllm.sh 中的 debugpy --listen 端口
scripts/debug_vllm.sh 中的 vLLM --port 端口
.vscode/launch.json 中的 connect.host/connect.port
```

## 5. 测试目录

### 5.1 Ascend 测试

```text
vllm-ascend/tests/
├── ut/                       单元测试
│   ├── _310p/、_fake_weight/  特殊设备/权重测试辅助
│   ├── _tools/                测试工具
│   ├── attention/
│   ├── compilation/
│   ├── core/
│   ├── device/、device_allocator/
│   ├── distributed/
│   ├── ops/
│   ├── patch/
│   ├── quantization/
│   ├── spec_decode/
│   ├── worker/
│   └── 其他 NPU 功能目录
└── e2e/                      端到端测试
    ├── pull_request/          PR 门禁测试
    ├── nightly/               nightly 模型/多节点测试
    ├── weekly/                weekly 测试
    ├── models/                模型测试资源
    ├── vllm_interface/        vLLM 接口测试
    ├── prompts/               测试 prompt
    └── coverage.md            功能覆盖矩阵
```

与当前专题最相关的测试入口：

- [`tests/e2e/pull_request/two_card/test_sequence_parallel_linear.py`](../vllm-ascend/tests/e2e/pull_request/two_card/test_sequence_parallel_linear.py)：TP/EP sequence-parallel 基础验证。
- [`tests/e2e/pull_request/four_card/test_pipeline_parallel.py`](../vllm-ascend/tests/e2e/pull_request/four_card/test_pipeline_parallel.py)：PP 验证。
- [`tests/e2e/nightly/single_node/models/configs/Qwen3-VL-32B-Instruct-W8A8.yaml`](../vllm-ascend/tests/e2e/nightly/single_node/models/configs/Qwen3-VL-32B-Instruct-W8A8.yaml)：Qwen3-VL 配置示例。

### 5.2 上游测试

```text
vllm/tests/
├── v1/                       v1 Engine/Worker/Attention/Executor 测试
├── distributed/              分布式测试
├── engine/、entrypoints/     Engine/API 测试
├── model_executor/、models/  模型和层测试
├── multimodal/               多模态测试
├── compile/                  编译和 graph 测试
├── kernels/、quantization/   kernel/量化测试
├── spec_decode/、samplers/   speculative/sampling 测试
└── 其他功能测试目录
```

Ascend 文档源码目录：

```text
vllm-ascend/docs/source/
├── _templates/                文档模板
├── assets/                    架构/功能配图
├── community/                 社区和贡献者文档
├── developer_guide/           开发者指南
├── tutorials/                 教程
├── user_guide/                用户指南
├── logos/                     项目 Logo
├── locale/                    多语言资源
├── quick_start.md             快速开始
├── installation.md            安装
└── index.md                   文档主页
```

## 6. 关键代码流入口

### 6.1 普通请求到模型 forward

```text
AsyncLLM.generate
  → vllm/vllm/v1/engine/core.py
  → vllm/vllm/v1/executor/multiproc_executor.py
  → vllm-ascend/vllm_ascend/worker/worker.py
  → vllm-ascend/vllm_ascend/worker/model_runner_v1.py
  → vllm/vllm/model_executor/models/<model>.py
```

### 6.2 PP + FlashComm

```text
Worker PP irecv/isend
  → ModelRunner padding
  → _preprocess
  → sync_and_slice_intermediate_tensors
  → model forward
  → custom op reduce-scatter/all-gather
  → 非最后 PP stage 返回 IntermediateTensors
  → 下一 PP stage
```

详细 shape 和通信说明见 [flashcomm_pp_sync.md](flashcomm_pp_sync.md)。

### 6.3 多模态输入

```text
GPUModelRunner._preprocess
  → _execute_mm_encoder
  → _gather_mm_embeddings
  → model.embed_input_ids(... multimodal_embeddings=...)
  → inputs_embeds
  → model forward
```

多模态与 FlashComm 的布局差异、VocabParallelEmbedding 设计见
[flashcomm_pp_sync.md](flashcomm_pp_sync.md) 以及上游 `vllm/multimodal/`、Ascend `ops/vocab_parallel_embedding.py`。

## 7. 按问题定位代码

| 问题 | 首先查看 |
| --- | --- |
| 服务无法启动/端口不通 | `scripts/*.sh`、`.vscode/launch.json`、`vllm/vllm/entrypoints/` |
| Worker 没有响应/忙循环 | `vllm/vllm/v1/executor/multiproc_executor.py`、`vllm-ascend/vllm_ascend/worker/worker.py` |
| 调度 token 数、padding 不对 | `vllm/vllm/v1/core/sched/`、`vllm-ascend/vllm_ascend/worker/model_runner_v1.py` |
| PP rank、send/recv、IntermediateTensors | `vllm/vllm/distributed/parallel_state.py`、Ascend `worker.py`、`model_runner_v1.py` |
| FlashComm 开关/context | `vllm_ascend/utils.py`、`ascend_forward_context.py`、`register_custom_ops.py` |
| Embedding/LMHead shape | 上游和 Ascend `vocab_parallel_embedding.py` |
| Linear reduce-scatter/all-gather | `vllm_ascend/ops/linear_op.py`、上游 `model_executor/layers/linear.py` |
| MoE routing/EP 通信 | `vllm_ascend/ops/fused_moe/` |
| Attention/MLA/SFA/DSA | `vllm_ascend/attention/`、`vllm_ascend/ops/mla.py` |
| 多模态 embedding/vision | 上游 `vllm/multimodal/`、`qwen3_vl.py`、Ascend `patch_qwen3vl.py` |
| graph/compile shape | 上游 `vllm/compilation/`、Ascend `compilation/` |
| HCCL/设备通信 | `vllm/distributed/device_communicators/`、Ascend `distributed/device_communicators/` |

`torch.compile`、Ascend GraphFusionPassManager 和 SP pattern 改写见
[torch_compile_pass_sp.md](torch_compile_pass_sp.md)。

本轮 SP/FlashComm pass、实际 FX graph 打印以及 ACL graph capture/replay 证据见
[flashcomm_sp_cuda_graph_report.md](flashcomm_sp_cuda_graph_report.md)。

## 8. 推荐阅读顺序

```text
1. 本文档：确认项目范围和代码入口
2. CONFIG.md：确认当前容器、PYTHONPATH 和权重路径
3. docs/flashcomm_pp_sync.md：理解当前 FlashComm + PP 设计
4. docs/torch_compile_pass_sp.md：理解 torch.compile、FX graph 和 SP pass
5. docs/flashcomm_sp_cuda_graph_report.md：查看本轮实际 graph 日志和验证边界
6. 先读上游 vllm/vllm/v1 的对应流程，再读 vllm-ascend 的覆盖/patch
7. 查看 scripts/ 和 tests/ 复现、验证具体问题
```

开发前还必须遵守根目录 [`AGENTS.md`](../AGENTS.md) 中的说明。
