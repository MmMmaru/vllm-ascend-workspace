## 环境配置
阅读docs/instructions获取最新的环境配置方案。

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

- `enable_sp()` 是运行时 上游 parallel_config 驱动的 SP 开关，主要由 `vllm_config.parallel_config.use_sequence_parallel_moe` 决定。
- `torch.compile` 负责 Dynamo FX 捕获、调用 AscendCompiler 和生成 compiled callable；它不是 ACL graph capture 本身。
- 上游 vLLM 的 `SequenceParallelismPass` 修改 FX 图；`FULL_DECODE_ONLY` 只决定 decode 是否捕获完整 ACL graph。Prefill 可以经过 SP pass，但通常以 `NONE` 运行时模式执行。
- `FULL`/`FULL_DECODE_ONLY` 是完整 forward/decode graph 模式；`PIECEWISE` 只捕获可捕获的编译区域；日志中的 `Capturing a aclgraph` 是 capture 动作本身。

## 源码理解

### 当前主要修改文件

- [`vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py`](vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)：`AscendMoERunner._maybe_reduce_final_output()` 在原生 MoE SP 时跳过最终 TP all-reduce，保留 token-sharded 结果给 Qwen3 MoE 后续 all-gather。
- [`vllm-ascend/vllm_ascend/ops/register_custom_ops.py`](vllm-ascend/vllm_ascend/ops/register_custom_ops.py)：SP+EP 的 all-gather 后按 DP local sizes 去除无效尾部，reduce-scatter 前恢复等长 padding；fake shape 使用 EP group world size。
- [`vllm-ascend/tests/ut/ops/test_fused_moe.py`](vllm-ascend/tests/ut/ops/test_fused_moe.py)：回归测试 SP 时不调用最终 all-reduce、非 SP 时保留 all-reduce。
- [`vllm-ascend/tests/e2e/pull_request/two_card/test_sp_pass.py`](vllm-ascend/tests/e2e/pull_request/two_card/test_sp_pass.py)：新增 DP2/TP2/EP/SP 与 SP-off 输出对照的功能 E2E。

### 相关文件

- [`vllm-ascend/vllm_ascend/compilation/compiler_interface.py`](vllm-ascend/vllm_ascend/compilation/compiler_interface.py)：`AscendCompiler.compile()`；`enable_npugraph_ex=False` 时进入 `fusion_pass_compile`，将 PassManager 插入 `torch.compile` 的 inner compiler。
- [`vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py`](vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py)：配置并按 `compile_range` 执行 Ascend fusion pass；SP pass 由上游 vLLM PassManager 管理。
- [`vllm-ascend/vllm_ascend/compilation/acl_graph.py`](vllm-ascend/vllm_ascend/compilation/acl_graph.py)：`ACLGraphWrapper` 在 runtime mode 匹配且 batch descriptor 命中时创建 `torch.npu.NPUGraph`，随后 replay；这是编译之后的运行时阶段。
- [`vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py`](vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py)：MoE prepare/finalize；`_use_ep_sequence_parallel()` 根据当前 MoE sequence-parallel token 布局选择 EP 路径。
- [`vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py`](vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)：MoE 主执行，消费 prepare 输出并调用 expert/routing，随后交给 finalize。
- [`vllm/vllm/model_executor/layers/fused_moe/config.py`](vllm/vllm/model_executor/layers/fused_moe/config.py)：`FusedMoEConfig.is_sequence_parallel` 的定义；本质是 `sp_size > 1`，表示 MoE 自己是否拥有 sequence-parallel token 分片。
- [`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`](vllm-ascend/vllm_ascend/worker/model_runner_v1.py)：NPU v1 preprocess、SP padding、forward、cudagraph runtime mode 和 attention metadata。
- [`vllm/vllm/v1/engine/core.py`](vllm/vllm/v1/engine/core.py) 和 [`vllm/vllm/v1/core/sched/scheduler.py`](vllm/vllm/v1/core/sched/scheduler.py)：EngineCore 调度、请求批处理和 KV cache 生命周期。
- [`vllm-ascend/vllm_ascend/attention/attention_v1.py`](vllm-ascend/vllm_ascend/attention/attention_v1.py)：从 `query_start_loc` 构造 attention metadata，并调用 NPU fused attention；长 prefill 的 query token 与 `actual_seq_lengths_q` 必须同时检查。
- [`scripts/bench_sp_tpot.sh`](scripts/bench_sp_tpot.sh)：当前 Qwen3-30B-A3B TP2/EP SP 长输入吞吐 benchmark，16K input、1 output、100 prompts、10 warmups、FULL_DECODE_ONLY、SP threshold 1024。
- [`scripts/serve_sp.sh`](scripts/serve_sp.sh) 和 [`scripts/bench_serve.sh`](scripts/bench_serve.sh)：在线服务和 TTFT benchmark 入口；TTFT 使用 `vllm bench serve`，关注 mean/median/p50/p90/p99 TTFT。
- [`docs/torch_compile_pass_sp.md`](docs/torch_compile_pass_sp.md)：torch.compile、FX pass 和 SP 的编译链说明。
- [`docs/sp_debug.md`](docs/sp_debug.md)：SP 性能回退排查、FX after-graph、profiler 和 MoE 重复 gather 判定标准。
- [`docs/pr_sequence_parallelism_moe_fix.md`](docs/pr_sequence_parallelism_moe_fix.md)：SP/MoE shape、`prepare_finalize` 和 residual 修复的历史设计说明；其中部分内容描述的是较早的期望实现，阅读当前 diff 时以源码为准。
- [`docs/flashcomm_sp_cuda_graph_report.md`](docs/flashcomm_sp_cuda_graph_report.md)：FlashComm/SP/ACL graph 调研和历史实验报告。
- [`docs/data_flow.md`](docs/data_flow.md)：简化的 NPU Worker 到 custom op 数据流。

### 当前诊断结论与验证边界

- DP2/TP2/EP4/SP 的乱码根因已确认：EP reduce-scatter 后结果按 token 分片，Ascend override 旧代码又无条件 TP all-reduce，随后 [`Qwen3Moe.forward()`](vllm/vllm/model_executor/models/qwen3_moe.py) 再 TP all-gather，导致不同 token 段被相加。
- 正确通信流为：`sequence_parallel_chunk -> EP all-gather -> 按 local sizes unpad -> MoE -> 按 local sizes zero-pad -> EP reduce-scatter -> Qwen3 TP all-gather`。DP 长度不均衡时，不能直接把 EP all-gather 的等长 buffer 当作有效连续 token 序列。
- 真实服务入口是 [`scripts/serve_sp.sh`](scripts/serve_sp.sh)，在容器 `xrs_vllm_main` 中使用 DP2、TP2、EP、`pass_config.enable_sp=true` 启动；`.log/bench_sp_true_16384.log` 记录了 `sp_size=2`、`ep_size=4` 和不均衡 token 分布。
- 修复后真实服务短请求约 19 token、长请求约 1.4K token 均输出连贯文本，SP-off 对照通过。`max_tokens=16` 会在模型完成最终数字回答前停止，只能用“无乱码/语义连贯”判断本回归。
- 已验证：聚焦 ops UT `27 passed`，compilation UT `26 passed`。完整 ops UT 仍有一个与本次改动无关的 `swiglustep` 小尺寸对齐失败；`typos` pre-commit 因容器内依赖安装卡住未完成，`ruff-check`、`ruff-format`、`codespell` 已通过。
- 修复提交为 [`49c074788`](vllm-ascend/)，后续若继续做性能优化，应重新收集 SP on/off profiler，并分别核对 `op_statistic.csv`、`operator_details.csv`、`communication.json` 与 after-graph；不能仅依据 `Replaced N patterns` 或通信算子总次数判断收益。
