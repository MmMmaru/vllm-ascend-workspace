## 环境配置

- 工作区根目录：`/home/x50063850/vllm-ascend-workspace`。
- 工作区不是单一 Python 仓库，而是一个 scaffold，包含两个 Git submodule：
  - [`vllm/`](vllm/)：上游 vLLM；当前 checkout 为 `7b3d595eb197d714052ce296cc8b124f0dc8af31`。
  - [`vllm-ascend/`](vllm-ascend/)：Ascend NPU 插件；当前 checkout 为 `9e65daf44bc4752121ac1a1c50325cc4652556e9`。
- 所有 torch/torch_npu、UT、E2E 和 benchmark 必须在 Docker 容器 `xrs_vllm_main` 中执行；本地宿主机不具备 NPU 运行条件。
- 容器中需要保留 CANN 原有环境，并把当前工作区源码追加到 `PYTHONPATH`：

  ```bash
  export PYTHONPATH=/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend-workspace/vllm:${PYTHONPATH:-}
  ```
- 如果连接不上尝试使用export https_proxy=localhost:33210端口
- 

- 常用环境：`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.1`、`SOC_VERSION=ascend910b1`，权重目录为 `/home/weights`。
- 多卡调试通常使用 `VLLM_WORKER_MULTIPROC_METHOD=spawn`、`OMP_NUM_THREADS=1`；`fork` 继承父进程 NPU/线程池状态，曾出现 OpenMP thread-pool abort。
- 临时缓存放 `.temp/`，日志、benchmark JSON 和 profiler 产物放 `.log/`。每个 benchmark 变体必须使用独立的 `VLLM_CACHE_ROOT`、日志和 JSON。
- 换仓/重建/运行验收流程见 [`docs/container_compile_flow.md`](docs/container_compile_flow.md)。关键原则：`.so` 存在只说明构建完成，`Replaced N patterns` 只说明 FX 图被改写，只有请求完成、JSON 和 E2E 输出一致才说明真实 NPU 执行成功。

## 项目架构

这是一个 vLLM 上游框架加 Ascend 硬件插件的双仓工作区。请求从 vLLM v1 EngineCore 进入 scheduler，经过 NPU Worker/ModelRunner 调度，进入模型 forward；Ascend 插件通过 platform interface、patch、custom op 和编译后端接入上游执行链。

```text
vllm entrypoint / OpenAI API / bench
        |
        v
vllm/vllm/v1/engine + core/sched/scheduler
        |
        v
vllm-ascend/worker/worker.py
        |
        v
vllm-ascend/worker/model_runner_v1.py
  preprocess / padding / attention metadata / model forward
        |
        v
上游模型 forward: Qwen3 / Qwen3-MoE / Qwen3-VL / DeepSeek
        |
        +--> Ascend ops: Linear, RMSNorm, Attention, MoE
        |       |
        |       +--> torch.ops.vllm.* custom ops
        |       +--> HCCL TP/DP/EP communication
        |
        +--> torch.compile / Dynamo FX graph
                |
                v
        vllm-ascend AscendCompiler
                |
                v
        GraphFusionPassManager
          SP / MoE SP / fusion passes
                |
                v
        modified FX graph -> Ascend/NPU backend compile
                |
                v
        ACLGraphWrapper / torch.npu.NPUGraph capture-replay
```

核心概念要分开：

- `enable_sp()` 是运行时 FlashComm1/运行时 SP 开关，主要由 `additional_config.enable_flashcomm1` 或 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 决定。
- `enable_sp_by_pass` 是编译期 SP 开关，要求非 eager 且 `compilation_config.pass_config.enable_sp=True`。
- `torch.compile` 负责 Dynamo FX 捕获、调用 AscendCompiler 和生成 compiled callable；它不是 ACL graph capture 本身。
- `SequenceParallelismPass` 修改 FX 图；`FULL_DECODE_ONLY` 只决定 decode 是否捕获完整 ACL graph。Prefill 可以经过 SP pass，但通常以 `NONE` 运行时模式执行。
- `FULL`/`FULL_DECODE_ONLY` 是完整 forward/decode graph 模式；`PIECEWISE` 只捕获可捕获的编译区域；日志中的 `Capturing a aclgraph` 是 capture 动作本身。

## 源码理解

### 当前主要修改文件

- [`vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism.py`](vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism.py)：SP pattern pass；当前未提交改动增加 `applicable=False`、零替换和正替换时的 warning/info，向日志明确报告 SP 替换或 TP fallback。
- [`vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py`](vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py)：MoE AllGather/ReduceScatter prepare/finalize；当前未提交改动增加 MoE SP 状态和 EP/DP 路径 debug 日志，并将 `_use_ep_sequence_parallel()` 当前实现收敛为返回 `enable_sp()`。
- [`vllm-ascend/tests/ut/compilation/test_sequence_parallelism.py`](vllm-ascend/tests/ut/compilation/test_sequence_parallelism.py)：当前未提交新增 UT，覆盖命中替换、零命中和 token range 不适用三类日志状态。
- [`vllm-ascend/tests/e2e/pull_request/two_card/test_sp_pass.py`](vllm-ascend/tests/e2e/pull_request/two_card/test_sp_pass.py)：当前未提交增加 `SP_TEST_MODEL` 环境覆盖和短输入 TP fallback 正确性 E2E。

### 相关文件

- [`vllm-ascend/vllm_ascend/compilation/compiler_interface.py`](vllm-ascend/vllm_ascend/compilation/compiler_interface.py)：`AscendCompiler.compile()`；`enable_npugraph_ex=False` 时进入 `fusion_pass_compile`，将 PassManager 插入 `torch.compile` 的 inner compiler。
- [`vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py`](vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py)：配置并按 `compile_range` 依次执行 fusion、SP 和 MoE SP pass；只有 `pass_config.enable_sp` 为真时才加入两个 SP pass。
- [`vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism_moe.py`](vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism_moe.py)：处理 MoE/all-gather epilogue 和 sequence-parallel chunk 消除；必须和主 SP pass 的 after-graph 一起判断。
- [`vllm-ascend/vllm_ascend/compilation/acl_graph.py`](vllm-ascend/vllm_ascend/compilation/acl_graph.py)：`ACLGraphWrapper` 在 runtime mode 匹配且 batch descriptor 命中时创建 `torch.npu.NPUGraph`，随后 replay；这是编译之后的运行时阶段。
- [`vllm-ascend/vllm_ascend/ascend_config.py`](vllm-ascend/vllm_ascend/ascend_config.py)：计算 `enable_sp_by_pass`，当前条件是 model config 存在、非 enforce eager、且 pass config enable_sp。
- [`vllm-ascend/vllm_ascend/utils.py`](vllm-ascend/vllm_ascend/utils.py)：`enable_sp()` 读取 FlashComm1/运行时 SP 开关；不要把它和 `enable_sp_by_pass()` 混为一个状态。
- [`vllm-ascend/vllm_ascend/ops/register_custom_ops.py`](vllm-ascend/vllm_ascend/ops/register_custom_ops.py)：注册 `maybe_chunk_residual`、`maybe_all_gather_and_maybe_unpad`、`maybe_pad_and_reduce` 等 `torch.ops.vllm`；真实实现和 fake 实现必须保持 shape/通信分支一致。
- [`vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py`](vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)：MoE 主执行，消费 prepare 输出并调用 expert/routing，随后交给 finalize。
- [`vllm/vllm/model_executor/layers/fused_moe/config.py`](vllm/vllm/model_executor/layers/fused_moe/config.py)：`FusedMoEConfig.is_sequence_parallel` 的定义；本质是 `sp_size > 1`，表示 MoE 自己是否拥有 sequence-parallel token 分片。
- [`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`](vllm-ascend/vllm_ascend/worker/model_runner_v1.py)：NPU v1 preprocess、SP padding、forward、cudagraph runtime mode 和 attention metadata。
- [`vllm/vllm/v1/engine/core.py`](vllm/vllm/v1/engine/core.py) 和 [`vllm/vllm/v1/core/sched/scheduler.py`](vllm/vllm/v1/core/sched/scheduler.py)：EngineCore 调度、请求批处理和 KV cache 生命周期。
- [`vllm-ascend/vllm_ascend/attention/attention_v1.py`](vllm-ascend/vllm_ascend/attention/attention_v1.py)：从 `query_start_loc` 构造 attention metadata，并调用 NPU fused attention；长 prefill 的 query token 与 `actual_seq_lengths_q` 必须同时检查。
- [`scripts/bench_sp_tpot.sh`](scripts/bench_sp_tpot.sh)：当前 Qwen3-30B-A3B TP2/EP SP 长输入吞吐 benchmark，16K input、1 output、100 prompts、10 warmups、FULL_DECODE_ONLY、SP threshold 1024。
- [`scripts/bench_flashcomm_tpot.sh`](scripts/bench_flashcomm_tpot.sh)：FlashComm1 对照 benchmark；当前实验需要额外检查 custom OPP、`libcust_opapi.so` 和 `vllm_ascend_C` 的加载路径。
- [`scripts/serve_sp.sh`](scripts/serve_sp.sh) 和 [`scripts/bench_serve.sh`](scripts/bench_serve.sh)：在线服务和 TTFT benchmark 入口；TTFT 使用 `vllm bench serve`，关注 mean/median/p50/p90/p99 TTFT。
- [`docs/torch_compile_pass_sp.md`](docs/torch_compile_pass_sp.md)：torch.compile、FX pass 和 SP 的编译链说明。
- [`docs/sp_debug.md`](docs/sp_debug.md)：SP 性能回退排查、FX after-graph、profiler 和 MoE 重复 gather 判定标准。
- [`docs/pr_sequence_parallelism_moe_fix.md`](docs/pr_sequence_parallelism_moe_fix.md)：SP/MoE shape、`prepare_finalize` 和 residual 修复的历史设计说明；其中部分内容描述的是较早的期望实现，阅读当前 diff 时以源码为准。
- [`docs/flashcomm_sp_cuda_graph_report.md`](docs/flashcomm_sp_cuda_graph_report.md)：FlashComm/SP/ACL graph 调研和历史实验报告。
- [`docs/data_flow.md`](docs/data_flow.md)：简化的 NPU Worker 到 custom op 数据流。
