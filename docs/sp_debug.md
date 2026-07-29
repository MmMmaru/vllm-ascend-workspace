# 诊断序列并行（Sequence Parallelism）性能回退

本文档描述一套可复现的工作流，用于排查在 Ascend NPU 推理场景中
开启序列并行（SP）后性能反而变慢的问题。文中以 TP2/TP4 基准测试中发现的
Qwen3 MoE 性能回退作为参考案例，但该工作流同样适用于 FlashComm、EP 以及其他
通信密集型路径。

## 1. 从受控对比开始

只有当两次运行仅在一个特性上存在差异时，性能对比才有意义。
请在 SP 运行与基线运行之间保持以下配置完全一致：

| 类别 | 需要保持不变的项 |
| --- | --- |
| 模型 | checkpoint、revision、dtype、量化方式 |
| 并行配置 | TP/DP/PP/EP 规模及 rank 映射 |
| 负载 | 输入长度、输出长度、请求数量、随机种子 |
| 调度器 | `max_model_len`、`max_num_batched_tokens`、chunked prefill、warmups |
| 运行时 | 容器镜像、CANN、PyTorch、vLLM、vLLM Ascend 提交版本 |
| 编译 | cudagraph 模式、capture sizes、缓存隔离 |

每个变体都要使用全新的缓存目录以及独立的 JSON/日志路径。否则，
残留的已编译图可能让 SP 开关看起来没有效果，或把不同源码版本的
结果混在一起。

基准测试必须同时产出日志和 JSON 结果。下面是一个最小化的离线
吞吐对比命令。请在支持 NPU 的 Docker 容器内运行，两次运行之间
只修改 `ENABLE_SP`。

```bash
set -euo pipefail

export PYTHONPATH=/workspace/vllm-ascend:/workspace/vllm:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG

ENABLE_SP=true
RUN_NAME=sp_${ENABLE_SP}
export VLLM_CACHE_ROOT=/workspace/.temp/${RUN_NAME}
PROFILE_DIR=/workspace/.log/profile_${RUN_NAME}
# Docker development containers may execute as root while /workspace is a
# host bind mount. Keep benchmark and profiler artifacts usable by the host
# user, including rank directories created later by CANN.
umask 000
mkdir -p /workspace/.log "${VLLM_CACHE_ROOT}" "${PROFILE_DIR}"
chmod a+rwX /workspace/.log "${PROFILE_DIR}"

vllm bench throughput \
  --model /home/weights/Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 20000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 4 \
  --num-prompts 10 \
  --num-warmups 2 \
  --seed 0 \
  --profile \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"warmup_iterations\":1,\"active_iterations\":4,\"max_iterations\":5}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[2,4],\"pass_config\":{\"enable_sp\":${ENABLE_SP},\"sp_min_token_num\":1024}}" \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}' \
  --output-json "/workspace/.log/profile_${RUN_NAME}.json" \
  2>&1 | tee "/workspace/.log/profile_${RUN_NAME}.log"
```

对于 TP4 长上下文基准测试，保持相同的命令结构，只修改
明确记录的 TP/负载参数。不要把 TP2 的 profile 吞吐与 TP4 的生产环境
基准直接对比：profiling 本身会引入额外开销，且两者回答的是
不同的问题。

## 2. 理解每个 profiling 层次能提供什么信息

按以下顺序使用各层次的信息：

1. **基准测试 JSON/日志**：确认请求集已完整执行，并给出端到端吞吐。
   检查 `Processed prompts: 100%`、`Throughput:` 行，以及 JSON 中的
   `elapsed_time`、`total_num_tokens`、`tokens_per_second` 字段。
2. **编译日志**：确认 SP pass 是否被选中、是否真正改动了图。
   `applicable=True` 和 `Replaced N patterns` 是必要但不充分的证据；
   还需检查替换后的图（after-graph）。
3. **算子统计**：找出哪些 kernel 消耗了额外时间。
   `op_statistic.csv` 可用于算子排序，但不显示造成开销的 tensor shape。
4. **Kernel 明细**：将算子与其输入/输出 shape 关联起来。对于
   MoE 回退，重点对比 `MoeGatingTopK`、`MoeInitRoutingCustom`、
   `GroupedMatmul`、`SwiGlu` 和 `MoeTokenUnpermute`。
5. **Step trace 与通信**：判断通信是否与计算重叠。
   `step_trace_time.csv` 汇总报告 `Computing`、`Communication`、
   `Overlapped`、`Free` 和 `Stage` 时间。
6. **Trace 查看器**：只有在 CSV 证据定位出可疑区间后，才使用
   `trace_view.json` 或 MindStudio Insight。时间线最适合用于确认
   执行顺序和重叠情况，而不是替代 shape 与调用次数分析。

## 3. 离线解析 Ascend PyTorch Profiler 输出

worker 可能会打印错误，提示数据无法在 daemon 进程中解析。
对于大型 NPU trace 这是预期行为。请在基准测试退出后，对每个
`*_ascend_pt` 目录执行解析：

```bash
python3 - <<'PY'
import glob
from torch_npu.profiler.profiler import analyse

for path in sorted(glob.glob("/workspace/.log/profile_sp_*/*_ascend_pt")):
    print(f"analysing {path}")
    analyse(path)
PY
```

重要的输出目录是
`*_ascend_pt/ASCEND_PROFILER_OUTPUT/`。无需额外 Python 依赖即可
生成一份精简的算子报告：

```bash
python3 - <<'PY'
import csv
import glob
import os

names = {
    "GroupedMatmul",
    "FusedInferAttentionScore",
    "MoeGatingTopK",
    "MoeInitRoutingCustom",
    "MoeTokenUnpermute",
    "SwiGlu",
}

for mode in ("true", "false"):
    paths = sorted(glob.glob(f"/workspace/.log/profile_sp_{mode}/*_rank0_*_ascend_pt"))
    if not paths:
        continue
    output = os.path.join(paths[0], "ASCEND_PROFILER_OUTPUT")
    print(f"=== SP {mode} ===")
    with open(os.path.join(output, "op_statistic.csv"), newline="") as stream:
        for row in csv.DictReader(stream):
            if row["OP Type"] in names:
                print(row["OP Type"], row["Count"], row["Total Time(us)"], row["Ratio(%)"])
PY
```

对于通信重叠情况，检查 `step_trace_time.csv` 中的那一行数据：

```bash
for mode in true false; do
  echo "=== SP ${mode} ==="
  find "/workspace/.log/profile_sp_${mode}" \
    -path '*/ASCEND_PROFILER_OUTPUT/step_trace_time.csv' -print -exec tail -n 1 {} \;
done
```

对比 `Communication(Not Overlapped)` 和 `Overlapped`。如果前者大幅增加
而后者几乎没有增长，说明新增的集合通信位于关键路径上，
没有被计算掩盖。

## 4. 将 trace 与 FX 编译图关联

编译与运行时采集是两个独立的阶段。请在 debug 日志中检查以下
所有内容：

```bash
LOG=/workspace/.log/bench_sp_true_16384.log

rg -n \
  'SequenceParallelism(Pass|MoePass).*applicable|Replaced [0-9]+ patterns|replaced [0-9]+ patterns' \
  "${LOG}"

rg -n \
  'reduce_scatter|all_gather|all_reduce|moe_forward|GroupedMatmul|unquantized_gemm' \
  "${LOG}" | head -200
```

正确的解读方式是：

- `applicable=False`：该编译区间没有使用 SP；不要将其运行时间
  归因于 SP。
- `applicable=True` 加上 `Replaced N patterns`：pass 确实改动了图。
- after-graph 中包含 `all_gather -> moe_forward`：MoE 在该图边界处
  收到了完整的 token 集合。检查 MoE prepare 是否又执行了一次 gather。
- `SequenceParallelismMoePass replaced 0 patterns`：没有图级别的清理
  消除可能存在的 `all_gather -> sequence_parallel_chunk` 配对。

找到图节点后，继续检查源码调用链：

```bash
rg -n \
  'enable_sp_by_pass|maybe_all_gather_and_maybe_unpad|maybe_pad_and_reduce|_use_ep_sequence_parallel' \
  vllm_ascend vllm
```

对于分布式模型，还应打印实际生效的 MoE 配置。特别地，
`sp_size=1` 意味着 MoE 自身并不持有序列并行的 token 分片，
即使编译 pass 在全局已开启。

## 5. 已诊断出的失败模式

在促成本文档的那次回退中，基准输入包含 8192 个 token。
SP 的 after-graph 为：

```text
reduce_scatter -> RMSNorm -> all_gather -> moe_forward
```

MoE prepare 路径同时又被全局 `enable_sp_by_pass` 标志选中，
调用了 `maybe_all_gather_and_maybe_unpad(..., is_ep_comm=True)`。
在 TP2 下，第二次 gather 把 MoE 输入从 8192 个 token 变成了
16384 个。因此路由和专家 kernel 做了重复的工作。同样的错误
还在 MoE 输出周围引入了多余的 reduce-scatter/all-gather 操作。

修复方法是区分「编译 pass 引入的 SP」和「MoE 原生 SP」：

```text
满足以下条件时使用 EP MoE prepare/finalize：
    启用了原生 FlashComm1
    或（启用了 SP pass 且 moe_config.is_sequence_parallel 为 true）

否则：
    使用常规 MoE prepare/finalize 路径
```

这样既能在 MoE 配置中 `sp_size > 1` 时保留 EP 路径，又能避免
在只开启 RMSNorm 编译 pass 时发生第二次 gather。

## 6. 修复后的验证清单

### 静态与单元验证

- 对 `prepare()` 和 `finalize()` 的路由决策分别做单元测试。
- 测试 `enable_sp=False`、`enable_sp_by_pass=False`。
- 测试原生 FlashComm1/SP 行为。
- 测试 `enable_sp_by_pass=True` 且 `moe_config.is_sequence_parallel=False`。
- 测试 `enable_sp_by_pass=True` 且 `moe_config.is_sequence_parallel=True`。
- 检查 after-graph；仅凭 pass 替换次数是不够的。

### 功能性 E2E 验证

在 SP 关闭和开启两种情况下，使用相同的 prompt 和贪心采样
运行 TP2 MoE 生成，然后对比 token id 和文本。这能发现
仅靠图测试无法检测的 shape 错误和通信错位。

```bash
cd /workspace/vllm-ascend
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MODELSCOPE=false
export SP_TEST_MODEL=/home/weights/Qwen/Qwen3-30B-A3B
pytest -sv \
  tests/e2e/pull_request/two_card/test_sp_pass.py::test_qwen3_moe_sp_pass_matches_no_sp_tp2
```

### 性能验证

重复受控基准测试，并要求以下全部成立：

- `total_num_tokens` 和完成的请求数量相同；
- MoE 输入 shape 恢复到基线的 token 数量；
- 路由专家输出 shape 没有被意外地乘以 TP；
- `Communication(Not Overlapped)` 和集合通信调用次数中不包含
  该 bug 引入的重复 gather；
- 只有在剔除 profiling 和 warmup 开销之后才对比吞吐。

profiling 运行本身是诊断产物，不是最终性能数字。请将基准 JSON、
基准日志、原始 `*_ascend_pt` 目录以及离线解析的
`ASCEND_PROFILER_OUTPUT` 一并保存，以便后续结果可以追溯到
确切的源码和运行时配置。

## 7. 精确定位多余的 all-gather

在最初的 TP2 profile 中，rank 0 报告了 `allReduce=245`、
`reduceScatter=240`、`allGather=485`。多出的 245 次 all-gather
并非全部由 SP 替换本身产生。源码层面的账目如下：

```text
240  SequenceParallelismPass 显式插入的 all_gather
240  MoE all-reduce 之后 maybe_chunk_residual 中隐式的 all_gather
  5  输入/embedding 对齐用的 gather
---
485  profile 中观测到的 all_gather 总数
```

这 240 次隐式调用位于
`vllm_ascend/ops/register_custom_ops.py:_maybe_chunk_residual_impl`：
当 MoE 输出是完整序列而 residual 是序列分片时，它会执行
`tensor_model_parallel_all_gather(residual, 0)`。造成这种 shape
不匹配的调用点是
`vllm_ascend/ops/fused_moe/fused_moe.py:_maybe_reduce_final_output`，
它在下一个 Attention RMSNorm 之前发出了
`maybe_all_reduce_tensor_model_parallel`。在旧图中，确切的链条是：

```text
moe_forward
  -> maybe_all_reduce_tensor_model_parallel
  -> aten.alias
  -> maybe_chunk_residual   # 隐式的 residual all_gather
  -> npu_add_rms_norm_bias
```

SP pass 现在会匹配这条保留 alias 的链条，并将其改为：

```text
moe_forward
  -> reduce_scatter
  -> maybe_chunk_residual
  -> npu_add_rms_norm_bias
  -> all_gather
  -> next Attention
```

这就是为什么必须通过 after-graph 验证 pass，而不能只看替换次数。
在修复后的 E2E 日志中，`SequenceParallelismPass` 替换了 96 个 pattern，
代表性图可见：`reduce_scatter_default_47` 位于
`.log/e2e_sp_maybe_ar_local5.log:13073`，`npu_add_rms_norm_bias_default`
位于 `13075`，两者之间没有
`maybe_all_reduce_tensor_model_parallel` 节点。源码实现位于
`vllm_ascend/compilation/passes/sequence_parallelism.py` 中
`_maybe_all_reduce_search_pattern` 辅助函数及三个已注册的
`maybe_all_reduce_pattern` 规则附近。

如果旧日志中在 `sequence_parallelism.py` 的 logger 行之下仍包含
`maybe_all_reduce_tensor_model_parallel -> aten.alias -> maybe_chunk_residual`，
请检查它是在 `after apply replacement graph()` 之前，还是在某个
`Pattern N:` dump 内部。这些行描述的是输入图或搜索 pattern，
本来就应包含原始链条。验收检查必须使用 `Replaced 96 patterns`
之后的图；在修复后的 E2E 日志中，对应的 MoE 输出在第 13073 行是
`reduce_scatter_default`，第 13075 行是
`npu_add_rms_norm_bias_default`，第 13077 行是
`all_gather_default`。该 after-graph 链条中不存在
`maybe_all_reduce_tensor_model_parallel` 节点。

## 8. 三类通信算子次数与性能解释

### 8.1 当前修复代码的实测账目

对当前修复代码重新采集的 TP2 profile 是
`.log/profile_current_sp_0724`。每个 rank 的原始
`communication.json` 统计如下；这是 profiler active 窗口内的总次数，
不是单层次数：

```text
                         allReduce  reduceScatter  allGather  总数
修复后 SP（rank0/rank1）       101          384         389    874
旧 SP（修复前）               245          240         485    970
无 SP                         485            0           5    490
```

上表的三类计数是按 `hcom_<collective>` key 统计，另外每个 profile
还有一个 `Total Op Info` 汇总项，未计入算子次数。当前 after-graph 中
每个 compiled graph 有 96 个 `reduce_scatter + all_gather` 对；本次
profile 覆盖 4 个 active graph 执行，因此得到 `384 = 96 x 4`。
代表性源码图位于 `.log/profile_current_sp_0724.log:7316-7321`。

注意旧 SP/无 SP 目录的有效通信窗口是 5 个执行，而当前修复 profile
是 4 个执行，因此上表是原始 profile 账目，不能直接用总数比较速度。
按一次 compiled graph 归一化后：当前修复是 `96 RS + 96` 个配对的显式
AG；旧 SP 是 `48 RS + 48 显式 AG + 48 隐式 residual AG`；无 SP 是
`97 AR + 1 AG`。这正好说明修复解决了旧 SP 中的 48 次隐式 residual
AG，但没有把一次通信变成零次通信。

当前 profile 的通信耗时（rank0）为：

```text
allReduce       101 次，  12.036 ms，平均 0.119 ms/次
reduceScatter   384 次， 371.204 ms，平均 0.967 ms/次
allGather       389 次， 345.813 ms，平均 0.889 ms/次
```

`allReduce=101` 不能直接解释成 101 个未替换的 RMSNorm。继续展开
`operator_details.csv` 后，当前 EP profile 的 101 个 communication.json
事件可精确拆成：4 个模型运行时 all-reduce，以及 97 个 1-element 的
HCCL communicator 初始化 warmup。4 个运行时事件在 operator hierarchy
中均表现为
`vllm::maybe_pad_and_reduce -> vllm::all_reduce -> c10d::allreduce_`
，每个 HCCS 数据量为 33.554432 MB，即
`8192 x 2048 x sizeof(bfloat16)`；因此它们是每个 active graph 一次，
不是 96 个 SP pattern 之外残留的 MoE all-reduce。对应的源码调用链是：

```text
AscendVocabParallelEmbedding._forward_origin
  -> maybe_pad_and_reduce(output_parallel)       # vocab_parallel_embedding.py:248
  -> _maybe_pad_and_reduce_impl(..., is_ep_comm=False)
  -> tensor_model_parallel_all_reduce             # register_custom_ops.py:89
  -> get_tp_group().all_reduce                     # communication_op.py:14
  -> torch.ops.vllm.all_reduce -> c10d::allreduce_ -> HcclAllreduce
```

`maybe_pad_and_reduce` 的输入图证据是
`.log/profile_current_ep_sp_0724.log:7429`：`embedding -> masked_fill ->
maybe_pad_and_reduce -> npu_rms_norm`。由于这次运行没有启用 FlashComm，
`register_custom_ops.py:88-89` 走 TP all-reduce 分支。97 个小事件则与
`pyhccl.py:124-128` 中注释明确的 1-element warmup 相符，不能计入模型
前向 all-reduce。判断 SP 是否替换成功，仍应以 after-graph 中的
`reduce_scatter -> RMSNorm -> all_gather` 为准，再用 profiler 时间戳
分析剩余 all-reduce 的调用来源。修复后的 after-graph 已经没有旧的 MoE 输出链；但 runtime 的 MoE dispatcher 仍打印
`MoECommType.ALLGATHER`，见 `.log/profile_current_sp_0724.log:17378`，
这表示 MoE 路由通信和 RMSNorm 前后的 TP SP 通信仍是两套独立机制。
在当前 A2 配置中，`num_tokens=8192` 而 `mc2_capacity=4`，不满足
`ascend_forward_context.py:239-251` 的 MC2 条件，所以选择 ALLGATHER。
随后 `fused_moe.py:510-521` 不会把 ALLGATHER 标记为已完成 TP reduction，
`fused_moe.py:533-539` 是一个静态潜在 all-reduce 调用点；但本次 profile
的最终执行证据并不支持它产生了那 4 个大 all-reduce。那 4 个事件已经由
`maybe_pad_and_reduce` 的父子算子关系和 embedding 前向图精确归因到
`vocab_parallel_embedding.py:248`。因此，排查剩余 all-reduce 时必须同时
看最终 graph 和 operator hierarchy，不能只根据 `fused_moe.py:538` 的源码
存在就判定 MoE 实际发起了 all-reduce。

### 8.1.1 `MoECommType.ALLGATHER` 是否在 MoE 内部产生了 all-gather

这个名称容易造成误判。当前 EP+SP profile 的 `allGather=389` 可以按
HCCL 传输数据量拆成：

```text
16.777216 MB  x 384  = SP pass 插入的显式 TP all_gather
0.151936 MB   x 2    = logits 输出的 TP all_gather
0.303872 MB   x 1    = logits 输出的 TP all_gather
0.455808 MB   x 2    = logits 输出的 TP all_gather
总计                 = 389
```

其中 `16.777216 MB = 8192 x 2048 x sizeof(bfloat16)`，正好对应
`96` 个 SP pattern 在 `4` 个 active graph 中执行，即 `96 x 4 = 384`。
`.log/profile_current_ep_sp_0724.log:5896-5903` 的 after-graph 直接显示
`reduce_scatter -> npu_add_rms_norm_bias -> all_gather -> router GEMM ->
vllm.moe_forward`；所以看起来位于 MoE 前面的这个 all-gather，实际是
SP 为恢复完整 token 序列而插入的 TP all-gather，不是 `moe_forward` 内部
发出的额外 all-gather。

运行时层级也排除了 MoE 内部 HCCL all-gather：EP profile 的
`operator_details.csv` 中 `vllm::all_gather=389`、
`vllm::moe_forward=192`、`vllm::maybe_all_gather_and_maybe_unpad=0`，且
每个 `vllm::moe_forward` 的子算子序列是
`npu_moe_init_routing_custom -> grouped_matmul -> npu_moe_token_unpermute`，
中间没有 `c10d::_allgather_base_` 或 `HcclAllGather`。剩余的 5 次小消息
与 `vocab_parallel_embedding.py:323-337` 的 logits gathering 相符，不能
算作 MoE 通信。

源码中的 `MoECommType.ALLGATHER` 只是选择了
`AllGatherCommImpl`；该实现使用 NPU routing/unpermute 完成 token 到本地
expert 的处理，并不等价于单独调用 HCCL all-gather。当前配置还明确是
`sp_size=1`、`dp_size=1`、`pcp_size=1`。因此
`PrepareAndFinalizeWithAllGather._use_ep_sequence_parallel()` 返回 false，
不会再执行 `maybe_all_gather_and_maybe_unpad`。删除图中 MoE 前的 SP
all-gather 不是安全的冗余删除：当前 MoE 接口接收的是完整 token 序列，
真正消除它需要把 MoE 的 sequence-parallel 边界和 SP pass 合并设计，确保
只保留一次必要的数据重组并验证输出 token 顺序与精度。

### 8.2 为什么 pattern 已替换，收益仍然不高

SP 不是把一次通信删除，而是把 RMSNorm 前的一次
`all_reduce` 改成了 `reduce_scatter`，并在 RMSNorm 后补回一次
`all_gather`：

```text
原路径：input -> all_reduce -> RMSNorm -> next GEMM
SP路径：input -> reduce_scatter -> RMSNorm -> all_gather -> next GEMM
```

因此，在归一化的单次 graph 中，通信从无 SP 的约 97 次 allReduce，
变成当前 SP 的约 192 次 RS/AG；只是通信数据形态和计算所在序列长度
发生了变化。当前 rank0 的
`reduceScatter + allGather` 已占约 `717.017 ms`，而无 SP profile 的
allReduce 约为 `715.906 ms`；也就是说这组通信在当前 A2/TP2 配置上
基本是持平，新增的 collective launch 和同步没有被 GEMM/Attention
计算充分隐藏。

这也解释了 benchmark 只得到约 0.85% 的提升：当前 profile 的
`Stage` 为约 1941 ms，无 SP 为约 1959 ms；SP 节省了一部分计算量，
但通信关键路径没有同步下降。下一步优化重点应从“继续增加 pattern
替换数”转向“减少 collective 次数或让 RS/AG 与计算重叠”，尤其要
单独分析 `ascend_forward_context.py:336` 选择的 MoE ALLGATHER，不能
把它误认为已经被 `SequenceParallelismPass` 优化。

## 9. 开启 EP 后的复测

之前的 profile 确实没有开启 EP：日志显示 `ep_size=1`、`use_ep=False`。
使用相同 workload 重新加入 `--enable-expert-parallel` 后，运行时确认：

```text
TP-only:  tp_size=2, ep_size=1, use_ep=False
TP+EP:   tp_size=1, ep_size=2, use_ep=True
```

但两次 profile 的通信次数完全相同：

```text
                         allReduce  reduceScatter  allGather
TP-only SP                    101          384         389
TP+EP SP                     101          384         389
```

rank0 通信耗时从 `allReduce=12.036 ms,
reduceScatter=371.204 ms` 变为 `allReduce=9.788 ms,
reduceScatter=403.100 ms`；`allGather` 基本不变。Stage 从约
`1941.387 ms` 增加到 `1964.098 ms`，profiling 输出吞吐从
`14297.72` 降到 `14201.02 total tokens/s`。这不是无 profiling 的最终
benchmark，但说明本次 EP 开启没有改善通信关键路径。

原因是 A2 的 MC2 选择条件仍不满足：EP2 时每卡 64 个专家，且
`num_tokens=8192`、`mc2_capacity=4`；`ascend_forward_context.py:239-251`
要求每卡专家数不超过 24、EP world size 至少 16、token 数不超过 capacity，
否则仍返回 `MoECommType.ALLGATHER`。因此 EP 只改变了专家放置方式，
没有改变当前 profile 的 MoE 通信算法，也没有消除剩余 all-reduce。
