## 环境配置
阅读docs/instructions获取最新的环境配置方案。

### 远端机器与大模型部署方法论（08-13 Kimi-K2.5 验证沉淀）

- 机器选型先看物理/逻辑卡映射：17.111 是 8 物理 NPU × 2 chip = 16 逻辑卡（`logical = physical * 2 + chip`，每逻辑卡 64GB，共 1TB）；17.110 只有 8 物理卡共 512GB，放不下 K2.5 w4a8（仅权重 500GB）。K2.5 权重在 NFS `/mnt/share/weights/kimi-k2.5-w4a8_modelscope`（126 分片，多机可见）。
- 17.111 容器运行时以 `/vllm-workspace` 为准，`/etc/profile.d/vaws-ascend-env.sh` 必须包含：PYTHONPATH 指向该树、ATB 路径（`ATB_HOME_PATH` + `LD_LIBRARY_PATH` 含 `nnal/atb/latest/atb/cxx_abi_1/lib`，缺了 worker 起不来报 `libatb.so` 找不到）、代理 `http_proxy=80.253.137.110:7897` 且 `no_proxy` 必须含本机 IP——否则 `vllm bench serve` 主会话 `trust_env=True` 会把打向本机的请求送进代理，造成 50/50 全挂且服务端无 access 日志。
- parity 快照仓库无 git tag，`vllm.__version__` 变成 `0.1.dev1+...`，`vllm_version_is()` 分支会走错（典型报错：取不到 `FusedMoEFactory`）；在快照运行时上服务必须显式 `--extra-env VLLM_VERSION=0.26.0`。
- parity 全量安装被 requirements pin 阻断时（如 `triton-ascend==3.2.2` 在各 pip 源不存在）的降级路径：parity materialize 源码 → `COMPILE_CUSTOM_KERNELS=0 pip install --no-deps --no-build-isolation -ve .`（vllm 和 vllm-ascend）→ 从旧树复制编译产物（`vllm_ascend_C*.so`、`libvllm_ascend_kernels.so`、`_cann_ops_custom/`）→ `serve_start --skip-parity`。
- 大模型加载慢（K2.5 约 610s），serving 默认 300s 健康超时不够；`bench_run.py` 需显式 `--health-timeout 1800`。
- bench 全挂排障顺序：先查客户端（代理泄漏、tokenizer 需 `--trust-remote-code`、`--backend` 与 `--endpoint` 必须配对如 openai+/v1/completions）→ 再查服务端 stdout 和 stderr 双日志 → 重跑全量前用手工单发→并发最小复现确认服务端健康，避免 10 分钟级启动空转。K2.5 的完整 serve/bench 配置与结果见 [benchmark-delete-flashcomm.md](benchmark-delete-flashcomm.md) model&setup 4。

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

### vLLM 上游与 vllm-ascend 的 CP（context parallel）机制（08-18 静态分析，未实测）

自 08-18 起 `vllm/` 子模块为 v0.27.0rc2-3（上游 main 附近）；旧 `vllm/model_executor/layers/attention/backends/` 目录已删除，attention 实现迁到 `vllm/v1/attention/backends/` + `vllm/v1/attention/ops/`。

**核心结论**：上游没有"all2all 恢复全量序列 → 切 head 维度 → attention → 再 all2all 变回 → 拿部分序列 hidden states 做 o_proj"的 Ulysses 式方案。上游 CP 拆成 DCP + PCP 两条正交轴，Q 序列永不跨 rank 交换，通信只发生在"部分输出 + LSE 合并"阶段：

- **DCP（decode context parallel，`--decode-context-parallel-size/-dcp`）**：KV cache 按序列交错分片存在各 rank（`cp_kv_cache_interleave_size`，interleave=1 时 token i 落 rank `i % dcp`，省 KV 显存）；decode 时每 rank 只处理属于自己的 token，[`get_dcp_local_seq_lens`](vllm/vllm/v1/attention/backends/utils.py:887) 算各 rank 本地 KV 长度，Q 仅做 head 维 all_gather（[`flash_attn.py:1182`](vllm/vllm/v1/attention/backends/flash_attn.py:1182)，入口 `_forward_with_dcp` 在 1134 行）后用本地 query × 本地 KV 分片算"部分 attention 输出 + LSE"。合并默认 [`cp_lse_ag_out_rs`](vllm/vllm/v1/attention/ops/common.py:213)（all_gather LSE → `correct_attn_out` 修正 → head 维 reduce_scatter，每 rank 拿回自己 head 子集）；可选 `--dcp-comm-backend=a2a` 走 [`dcp_a2a_lse_reduce`](vllm/vllm/v1/attention/ops/dcp_alltoall.py:392)——把 output+LSE 打包后单次 all_to_all（**attention 路径唯一的 all_to_all，交换的是输出与 LSE，不是 Q/K/V**），Triton kernel LSE 加权合成（参考 arXiv:2507.07120）。o_proj 吃 head 子集输出，靠 TP 权重切分 + 层末 all_reduce 汇总。
- **PCP（prefill context parallel，`--prefill-context-parallel-size/-pcp`）**：prefill 序列切分并行、KV cache 复制（不省显存）；[`pcp.py:11`](vllm/vllm/model_executor/layers/attention/pcp.py:11) `_gather_prefill_cache_inputs` 在写 cache 前把 prefill 的 K/V 做 dim-0 all_gather 拼成全量序列（这是上游最接近"恢复全量序列"的一步，但发生在 cache 写入而非 attention 前）；MLA decode 按 head 切分计算，算完 [`finalize_mla_pcp_decode`](vllm/vllm/model_executor/layers/attention/pcp.py:83) 用 head 维 all_gather 拼回全 head 再做 v_up（[`mla_attention.py:902-946`](vllm/vllm/model_executor/layers/attention/mla_attention.py:902)）。
- **约束**：DCP 要求 attention 后端返回 softmax LSE（[`cp_utils.py:15`](vllm/vllm/v1/worker/cp_utils.py:15) `check_attention_cp_compatibility`）；DCP/PCP/TP 尺寸整除关系校验在 [`parallel.py:524-539`](vllm/vllm/config/parallel.py:524)。

vllm-ascend 侧已有 NPU 版 CP：[`vllm_ascend/attention/context_parallel/`](vllm-ascend/vllm_ascend/attention/context_parallel/)（`attention_cp.py`/`common_cp.py` 镜像上游 DCP 布局，`common_cp.py:15` 的 `get_dcp_local_seq_lens` 与上游同名函数同语义；`dsa_cp.py`/`sfa_cp.py`/`mla_cp.py` 为 DSA/SFA/MLA 各后端的 CP 封装）。08-18 E2E `context_parallel/test_accuracy.py` 的 DSA-CP/SFA-DCP golden 用例已通过（详见 PROGRESS.md）。若要在 vllm-ascend 实现 Ulysses 式 all2all CP，上游无现成代码可复用，需自行实现。

### 当前诊断结论与验证边界

- DP2/TP2/EP4/SP 的乱码根因已确认：EP reduce-scatter 后结果按 token 分片，Ascend override 旧代码又无条件 TP all-reduce，随后 [`Qwen3Moe.forward()`](vllm/vllm/model_executor/models/qwen3_moe.py) 再 TP all-gather，导致不同 token 段被相加。
- 正确通信流为：`sequence_parallel_chunk -> EP all-gather -> 按 local sizes unpad -> MoE -> 按 local sizes zero-pad -> EP reduce-scatter -> Qwen3 TP all-gather`。DP 长度不均衡时，不能直接把 EP all-gather 的等长 buffer 当作有效连续 token 序列。
- 真实服务入口是 [`scripts/serve_sp.sh`](scripts/serve_sp.sh)，在容器 `xrs_vllm_main` 中使用 DP2、TP2、EP、`pass_config.enable_sp=true` 启动；`.log/bench_sp_true_16384.log` 记录了 `sp_size=2`、`ep_size=4` 和不均衡 token 分布。
- 修复后真实服务短请求约 19 token、长请求约 1.4K token 均输出连贯文本，SP-off 对照通过。`max_tokens=16` 会在模型完成最终数字回答前停止，只能用“无乱码/语义连贯”判断本回归。
- 已验证：聚焦 ops UT `27 passed`，compilation UT `26 passed`。完整 ops UT 仍有一个与本次改动无关的 `swiglustep` 小尺寸对齐失败；`typos` pre-commit 因容器内依赖安装卡住未完成，`ruff-check`、`ruff-format`、`codespell` 已通过。
- 修复提交为 [`49c074788`](vllm-ascend/)，后续若继续做性能优化，应重新收集 SP on/off profiler，并分别核对 `op_statistic.csv`、`operator_details.csv`、`communication.json` 与 after-graph；不能仅依据 `Replaced N patterns` 或通信算子总次数判断收益。
