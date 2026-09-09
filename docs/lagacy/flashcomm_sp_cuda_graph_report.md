# SP / FlashComm Pass 与 CUDA/ACL Graph 调研报告

> 日期：2026-07-21  
> 分支：`vllm-ascend/flashcomm-by-pass`  
> 说明：当前实验硬件是 Ascend NPU。vLLM 配置仍沿用 `cudagraph_*` 命名，但底层实现是 `torch.npu.NPUGraph`/ACL graph，本文将两者合称为 CUDA/ACL graph。

## 结论摘要

1. 图变换发生在 graph capture 之前：Dynamo 生成 FX graph，Ascend `GraphFusionPassManager` 执行 pass，随后 `AscendCompiler` 编译；`ACLGraphWrapper` 最后对编译后的 callable 做 capture/replay。
2. SP pass 的核心替换是 `all_reduce -> RMSNorm` 改成 `reduce_scatter -> local RMSNorm -> all_gather`，并在 residual 上插入 `maybe_chunk_residual`。Ascend 的 FlashComm1 与上游 `pass_config.enable_sp` 是两个开关。
3. 旧 FlashComm 自定义线性算子已经从 layer 构造路径删除；稳定替换点应是编译图中的普通 `unquantized_gemm`，以及 `unquantized_gemm -> all_reduce`。
4. 真实 Qwen3-30B-A3B TP2/EP E2E 已观察到 ACL graph 的 capture 和 replay；原生 SP 修复后的 E2E 以 `1 passed` 完成，并在两张卡上各命中 `SequenceParallelismPass replaced 48 patterns`。
5. 本轮还修复了两个真实图运行时问题：Dynamo 省略 RMSNorm 默认参数导致的 pattern arity/topology 不匹配，以及 `maybe_pad_and_reduce` fake shape 与真实非 EP 分支不一致；另补齐了 local residual 在后续 full-sequence 输入前的 all-gather。

## 1. 调用链与 graph capture 时序

```text
model forward
  -> torch.compile / Dynamo FX graph
  -> AscendCompiler
  -> GraphFusionPassManager
       -> FlashCommPass
       -> SequenceParallelismPass
       -> SequenceParallelismMoePass
  -> graph.recompile()
  -> compiled callable
  -> ACLGraphWrapper
       -> torch.npu.NPUGraph capture
       -> aclgraph.replay()
```

关键实现位置：

- [`graph_fusion_pass_manager.py`](../vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py)：配置和顺序；FlashComm1 由 `enable_sp(config)` 加入，原生 SP 由 `config.compilation_config.pass_config.enable_sp` 加入。
- [`flashcomm.py`](../vllm-ascend/vllm_ascend/compilation/passes/flashcomm.py)：FlashComm1 FX/NGE replacement。
- [`sequence_parallelism.py`](../vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism.py)：`all_reduce + RMSNorm` 的 SP replacement。
- [`sequence_parallelism_moe.py`](../vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism_moe.py)：all-gather epilogue 和 `all_gather + sequence_parallel_chunk_impl` 消除。
- [`acl_graph.py`](../vllm-ascend/vllm_ascend/compilation/acl_graph.py)：capture/replay；`Capturing a aclgraph` 在约第 159 行，`Replaying aclgraph` 在约第 252 行。

因此，pass 打印的是 FX/编译图，ACL graph 日志证明的是编译结果被捕获和重放；二者不是同一层图。

## 2. SP pass 如何修改图

### 2.1 Middle layer

`MiddleAllReduceRMSNormPattern` 的源码模式为：

```text
input
  -> all_reduce
  -> npu_add_rms_norm_bias(all_reduce(input), residual, weight)
```

替换为：

```text
input
  -> reduce_scatter(dim=0, world_size=TP)
  -> maybe_chunk_residual(reduce_scatter, residual)
  -> npu_add_rms_norm_bias(local_input, local_residual, weight)
  -> all_gather(dim=0, world_size=TP)
```

返回值仍保持 `(result, residual)`。Last layer 使用同一通信顺序，但不返回 residual 的第二项。

### 2.2 MoE / all-gather epilogue

`SequenceParallelismMoePass` 处理另一类已经包含 all-gather 的图：

```text
all_gather(input)
  -> slice[:num_tokens]
  -> npu_add_rms_norm_bias(...)
```

变成：

```text
maybe_chunk_residual(input, residual)
  -> npu_add_rms_norm_bias(input, local_residual, weight)
  -> all_gather(result)
```

同时，`all_gather -> sequence_parallel_chunk_impl` 被替换为 identity。这样可以避免先 gather 全量 token、再切回本 rank 的无效往返。

### 2.3 适用范围

- Dense 模型的 `SequenceParallelismPass` 默认从 `compile_range.start >= 1000` 才启用。
- MoE 模型的最小 token 阈值是 `1`。
- `enable_flashcomm1`/`VLLM_ASCEND_ENABLE_FLASHCOMM1` 与 `pass_config.enable_sp` 必须分别检查；只开前者不会加入原生 `SequenceParallelismPass`。

SP pass 自身通过 DEBUG 日志打印 pass 前后的完整 FX graph，并打印替换计数。本轮真实 Qwen3-30B-A3B TP2/EP E2E 已确认两张卡均命中 48 个 SP replacement；`SequenceParallelismMoePass` 在该模型图上为 0，说明本次命中的是 all-reduce epilogue pattern，而不是 all-gather epilogue pattern。

## 3. 删除 FlashComm 自定义算子后的替换点

### 3.1 删除前后的设计

旧路径在 `linear_op.py` 中通过 `SequenceColumnParallelOp`、`SequenceRowParallelOp` 和 `matmul_and_reduce` 等对象/自定义 op，在 layer 构造或 forward 分支中直接决定通信方式。这会把 FlashComm 逻辑散落到多个 Linear 类型和模型路径中。

当前提交删除了这些旧入口，保留普通编译图可识别的算子：

| 图形态 | pass 替换 | 目的 |
| --- | --- | --- |
| `unquantized_gemm(x, weight)`，列并行层 | `maybe_all_gather_and_maybe_unpad(x, True) -> unquantized_gemm` | 先把 token gather 到完整形状，再做本 rank 的列并行 GEMM |
| `unquantized_gemm(x, weight) -> all_reduce`，行并行层 | `maybe_pad_and_reduce(gemm)` | 安全 fallback，保留 padding/reduce 语义 |
| 同一个行并行形态，非 MoE、TP <= 8 且 range 适用 | `npu_mm_reduce_scatter_base` | 使用 MMRS 将 GEMM 与 reduce-scatter 融合 |

layer 侧只保留 eager fallback：[`linear.py`](../vllm-ascend/vllm_ascend/ops/linear.py) 在 compilation mode 为 `NONE` 时调用 `maybe_all_gather_and_maybe_unpad` 或 `maybe_pad_and_reduce`；进入编译路径后使用稳定的 `unquantized_gemm`，由 `FlashCommPass` 接管。

### 3.2 真实替换点的注意事项

真实 Dynamo 图会省略无 bias 调用中显式的 `None`，日志中看到的是：

```text
unquantized_gemm = torch.ops.vllm.unquantized_gemm.default(view, arg2_1)
all_reduce = torch.ops.vllm.all_reduce.default(unquantized_gemm, 'tp:0')
```

原 pass 使用三参数 pattern，因此 PatternMatcher 不会命中。已将 [`flashcomm.py`](../vllm-ascend/vllm_ascend/compilation/passes/flashcomm.py) 的 column/row pattern 和 replacement 改为两参数形态，并同步把单测中的无 bias graph 改成真实形态。

另外，layer metadata 过滤仍然必要：列并行 pattern 不能把 router GEMM、vision/shared expert 或排除的 `wo_a/q_b/kv_b` 误替换；行并行 pattern 则需要检查模块路径和已启用的 FlashComm2/MLP TP/MatmulAllReduce 开关。

## 4. 实际打印的 FX graph

### 4.1 replacement 函数的可控小图

在 CPU 容器中对 replacement 函数做 `torch.fx.symbolic_trace`，修改 pattern 签名之前打印出的 replacement 图为：

```text
graph():
    %x = placeholder[target=x]
    %weight = placeholder[target=weight]
    %maybe_all_gather_and_maybe_unpad = call_function[
        target=torch.ops.vllm.maybe_all_gather_and_maybe_unpad
    ](args=(%x, True), kwargs={})
    %unquantized_gemm = call_function[
        target=torch.ops.vllm.unquantized_gemm
    ](args=(%maybe_all_gather_and_maybe_unpad, %weight, None), kwargs={})
    return unquantized_gemm
```

这是修改前 replacement 本身的“通信 + GEMM”图；修改后的实现使用两参数无 bias 形态。该打印不证明它已匹配真实模型图；真实模型图的参数签名差异正是本轮发现的问题。

### 4.2 真实 Qwen3-MoE graph 的 pass 前后输出

实验命令开启了 `VLLM_LOGGING_LEVEL=DEBUG`，并在 FlashCommPass 中加入：

```python
logger.debug("FlashCommPass before apply replacement\\n%s", graph)
...
logger.debug("FlashCommPass after apply replacement\\n%s", graph)
```

真实输出节选（日志文件：`.log/flashcomm_e2e.log`；该日志对应“两参数修正前”的一次 instrumentation rerun）：

```text
FlashCommPass compile_range=(1, 8192) applicable=True mmrs=False
FlashCommPass before apply replacement
    view = torch.ops.aten.view.default(arg0_1, [-1, 2048])
    unquantized_gemm = torch.ops.vllm.unquantized_gemm.default(view, arg2_1)
    all_reduce = torch.ops.vllm.all_reduce.default(unquantized_gemm, 'tp:0')
    maybe_chunk_residual = torch.ops.vllm.maybe_chunk_residual.default(all_reduce, arg3_1)
    npu_add_rms_norm_bias = torch.ops._C_ascend.npu_add_rms_norm_bias.default(...)
    ...
FlashCommPass after apply replacement
    view = torch.ops.aten.view.default(arg0_1, [-1, 2048])
    unquantized_gemm = torch.ops.vllm.unquantized_gemm.default(view, arg2_1)
    all_reduce = torch.ops.vllm.all_reduce.default(unquantized_gemm, 'tp:0')
    ...
FlashCommPass replaced 0 patterns (all_gather=0, reduce_scatter=0, all_reduce=1)
```

前后图一致，说明原三参数 pattern 没有命中；`all_reduce=1` 是图中原有节点统计，不是 pass 新增的 reduce-scatter。修正两参数 pattern 后，正确的验收信号应是：

```text
FlashCommPass replaced <positive number> patterns
... all_gather=<positive number> ...
```

## 5. 实际 ACL graph capture/replay 报告

### 5.1 成功的真实 E2E

用例：

```text
tests/e2e/pull_request/two_card/test_flashcomm_distributed.py::
    test_qwen3_moe_flashcomm1_tp2_fx_compile
```

配置：Qwen3-30B-A3B、TP2、EP、`cudagraph_mode=PIECEWISE`、capture size `[2]`、关闭 `npugraph_ex`。首次真实 NPU 运行结果为 `1 passed`。

关键日志：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 0% ...
[acl_graph.py:159] Capturing a aclgraph on
    (PIECEWISE,BatchDescriptor(num_tokens=2, ...))
Graph capturing finished in 7 secs, took 0.02 GiB
Running batch with cudagraph_mode: NONE, ... num_tokens=1002
Running batch with cudagraph_mode: PIECEWISE, ... num_tokens=2
[acl_graph.py:252] Replaying aclgraph
```

解释：1002 token 的 prefill 超过 capture bucket，走 `NONE`；decode 的 2 token 命中 `[2]` bucket，走 PIECEWISE 并 replay。这说明 graph capture/replay 链路工作正常，但不等价于 FlashComm pass 已经替换了图。

### 5.2 metrics 报告如何打开

若要输出 vLLM 的 CUDAGraph bucket 统计，需要同时满足：

```text
--cudagraph-metrics
```

以及未关闭 log stats。日志表记录 unpadded tokens、padded tokens、padding 数、runtime mode 和 count。Ascend 的底层 DEBUG 证据仍应优先看 `acl_graph.py` 的 capture/replay 行。

## 6. 验证结果与剩余风险

已完成：

- CPU 容器：`tests/ut/compilation/test_flashcomm.py`，修复后 `5 passed, 14 warnings`。
- CPU 容器：实际打印 replacement FX graph。
- NPU 容器：原 FlashComm E2E 首次运行 `1 passed`，真实 ACL graph capture/replay 已观察到。
- NPU instrumentation：已打印真实 FlashComm pass 前后图，并确认原实现 `replaced 0 patterns`。
- NPU 容器：SP 修复后 Qwen3-30B-A3B TP2/EP E2E `1 passed`，两张卡分别打印 `SequenceParallelismPass replaced 48 patterns`。
- NPU 容器：SP E2E 日志观察到 `cudagraph_mode: FULL` 的 batch 调度，日志文件为 `.log/sp_e2e_real10.log`；该配置使用 `FULL_DECODE_ONLY`，避免 SP 与 piecewise/full-graph 强制切换产生的 profile shape 冲突。
- 代码：增加 FlashComm/SP pass 的 DEBUG 完整图打印；修正无 bias RMSNorm arity、`maybe_chunk_residual` topology、`maybe_pad_and_reduce` fake shape，以及 local residual 回 full sequence 的通信路径。

实验中定位并解决的错误：

- 原始 SP probe 只打印 `replaced 0 patterns`：真实图包含 `all_reduce -> maybe_chunk_residual -> npu_add_rms_norm_bias`，而模板漏掉了中间节点，并使用了与 Dynamo 默认参数不同的 RMSNorm 参数个数。
- SP 开启后首次运行出现 `shape '[4096, 16, 128]' is invalid ...`：`maybe_pad_and_reduce` fake 实现错误地把普通 TP all-reduce 也推导成 reduce-scatter shape；修正 fake 分支后该错误消失。
- 随后出现 `AddRmsNormBias do tiling failed`：上一个 SP RMSNorm 的 local residual 进入 full-sequence 输入；`maybe_chunk_residual` 现在按大小方向选择 chunk 或 TP all-gather，修复后 E2E 通过。

复现/验收命令：

```bash
export PYTHONPATH=/home/x50063850/vllm-workspace/vllm:/home/x50063850/vllm-workspace/vllm-ascend:/home/x50063850/vllm-workspace/.temp:$PYTHONPATH
export VLLM_LOGGING_LEVEL=DEBUG
export ASCEND_RT_VISIBLE_DEVICES=2,3

python -m pytest -q -s \
  tests/ut/compilation/test_flashcomm.py

python -m pytest -q -s \
  tests/e2e/pull_request/two_card/test_flashcomm_distributed.py \
  -k sequence_parallelism_tp2_fx_compile 2>&1 | tee .log/sp_e2e_after_fix.log
```

剩余风险：`SequenceParallelismMoePass` 在本次 Qwen3-30B-A3B 图上替换数为 0；如果后续模型走 all-gather epilogue，需要单独提供对应真实图和输出等价性验证。当前验证覆盖的是 TP2/EP、SP pass、FULL_DECODE_ONLY graph capture/replay，不等同于所有模型和所有 CUDA/ACL graph 模式。

## 7. DEBUG 日志下的 SP 消融实验（2026-07-22）

本节记录本次实际运行的关闭 SP/开启 SP 对照实验。实验在 Docker 容器 `xrs_vllm_main` 中执行，使用 Ascend NPU；因此日志中的 CUDA graph 实际对应 ACL graph。两组实验除 SP 开关外保持一致：Qwen3-VL-2B-Instruct、TP2、bfloat16、`FULL_DECODE_ONLY`、capture sizes `[2, 4]`、随机输入 512 token、随机输出 128 token、100 个请求、10 个 warmup、seed 0。

运行时环境：

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=DEBUG
```

编译配置分别为：

```json
{"pass_config": {"enable_sp": false}}
{"pass_config": {"enable_sp": true, "sp_min_token_num": 1}}
```

### 7.1 结果

| 配置 | elapsed time | requests/s | total tokens/s | output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| SP 关闭 | 10.5096 s | 9.5151 | 6089.64 | 1217.93 |
| SP 开启 | 12.2877 s | 8.1382 | 5208.47 | 1041.69 |

开启 SP 后，本次 workload 的请求吞吐和输出 token 吞吐均下降 **14.47%**。这是 100 请求、短输入输出的一次单轮结果，SP 通信收益可能尚未覆盖额外的 reduce-scatter/all-gather 成本，不能据此推导长序列或大模型的最终结论；正式结论应增加重复轮次，并分别测试 prefill/decode 和更长上下文。

### 7.2 pass 与 graph 证据

- 关闭 SP 日志确认 `pass_config: {'enable_sp': False}`；未执行 SP pass。
- 开启 SP 日志确认 `enable_sp=True, sp_min_token_num=1`。
- 开启 SP 时，两张 TP worker 均打印 `SequenceParallelismPass ... applicable=True`，并分别打印 `Replaced 53 patterns`；`SequenceParallelismMoePass replaced 0 patterns`。
- 两组都完成 graph capture：分别捕获 FULL graph 的 batch size 4 和 2，并打印 `Graph capturing finished`。关闭 SP capture 用时约 3 秒、占用 0.23 GiB；开启 SP 用时约 2 秒、占用 0.25 GiB。
- 运行期间两组日志都同时出现 `cudagraph_mode: FULL` 和 `cudagraph_mode: NONE`：prefill/不在 capture bucket 的 batch 走 NONE，decode batch 2/4 命中 FULL graph。这是预期调度行为，不表示 graph capture 失败。

### 7.3 产物与复现命令

- 关闭 SP 结果：[bench_no_sp_debug_512.json](../.log/bench_no_sp_debug_512.json)，完整 DEBUG 日志：[bench_no_sp_debug_512.log](../.log/bench_no_sp_debug_512.log)。
- 开启 SP 结果：[bench_sp_debug_512.json](../.log/bench_sp_debug_512.json)，完整 DEBUG 日志：[bench_sp_debug_512.log](../.log/bench_sp_debug_512.log)。

命令模板：

```bash
vllm bench throughput \
  --model /home/weights/Qwen/Qwen3-VL-2B-Instruct \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 1024 \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 100 --num-warmups 10 --seed 0 \
  --compilation-config '<SP_JSON>' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

`<SP_JSON>` 替换为上面的关闭或开启 SP 配置即可。首次尝试使用 `--input-len 512 --output-len 128`，但 random dataset 的 `--random-input-len` 默认值仍为 1024，导致 prompt 1024 加 output 128 超过 `max_model_len=1024`；该失败日志不是 SP 运行失败，最终对照结果使用了显式的 `--random-input-len 512 --random-output-len 128`。
