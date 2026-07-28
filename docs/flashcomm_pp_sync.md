# FlashComm1 + Pipeline Parallelism 通信与 Shape 流程
已解决 PR #9155
## PP=2、TP=2 的完整 shape 示例

假设本轮 `T_real=3`，FlashComm 将输入补齐到 `T_pad=4`：

```text
第一段 PP（rank 0/1）
──────────────────────────────
input_ids                         [4]
embedding output                  [2, H] 每个 TP rank
FlashComm reduce-scatter          [2, H]  
执行 PP stage 0 的 layers
PP send hidden_states/residual    local layout [2, H]

第二段 PP（rank 2/3）
──────────────────────────────
PP receive                        [2, H]  （FlashComm设置后 PP 不做 all-gather， 而是通过pp group内进行点对点通信）
sync_self=True                    copy 到 local buffer
sync_and_slice                    返回 [2, H]
执行 PP stage 1 的 layers
最后 stage norm / logits 前处理
必要的最终 hidden-state all-gather
```

本文以当前工作区中的 vLLM/vLLM-Ascend 代码为准，说明 FlashComm1（Ascend
的 sequence-parallel/collective-fusion 路径）与 Pipeline Parallelism（PP）同时
开启时，一轮 `execute_model` 中数据如何传输，以及每个阶段的 tensor shape 如何
变化。

文中使用以下符号：

| 符号 | 含义 |
| --- | --- |
| `T_real` | scheduler 本轮实际调度的 token 数，即 `scheduler_output.total_num_scheduled_tokens` |
| `T_pad` | 为 FlashComm、DP 或 graph 对齐后的 token 数，传入模型的 token 数 |
| `TP` | tensor parallel size |
| `P` | pipeline parallel size，即 `get_pp_group().world_size` |
| `H` | hidden size |
| `T_local` | FlashComm 在单个 TP rank 上保留的 token 数，通常为 `ceil(T_pad / TP)` |

这里的 `T_real` 是“当前 forward 的 token 数”，不是请求的完整上下文长度。decode
阶段一个请求通常每轮只调度一个 token，因此不能用请求总长度直接推断
`intermediate_tensors.shape[0]`。

## 进程内一轮执行的总流程

Ascend worker 的主路径可以概括为：

```text
Worker.execute_model
  │
  ├─ 非第一段 PP：irecv_tensor_dict（接收上一段 PP 的 activation）
  │
  └─ ModelRunner.execute_model
       │
       ├─ 准备 batch/attention metadata
       ├─ _preprocess
       │    ├─ 第一段 PP：准备 input_ids 或 inputs_embeds
       │    └─ 非第一段 PP：copy + slice intermediate_tensors （目的是准备给_model_forward消费的intermediate_tensors）
       ├─ _model_forward
       │    └─ Qwen3MoeModel.forward
       └─ 后处理
            ├─ 非最后段 PP：返回 IntermediateTensors
            └─ 最后段 PP：必要时 all-gather，继续 logits/sampling
  │
  └─ 非最后段 PP：isend_tensor_dict（异步发送给下一段 PP）
```

Worker 的 PP receive/send 在
[`worker.py`](../vllm-ascend/vllm_ascend/worker/worker.py) 的
`execute_model` 路径中；模型调用前的整理在
[`model_runner_v1.py`](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
中完成。

## PP group 的 rank 拓扑

`parallel_state.py` 以 `TP` 为最后一个维度建立 rank 布局，并将 PP 维和 TP
维转置后构造 PP groups，见
[`initialize_model_parallel`](../vllm/vllm/distributed/parallel_state.py)。

以 `PP=2、TP=2、DP=1、PCP=1` 为例，通常得到：

```text
TP lane 0 的 PP group: [0, 2]
TP lane 1 的 PP group: [1, 3]

PP stage 0: rank 0 / rank 1
PP stage 1: rank 2 / rank 3
```
## 

## 相较于上游vllm的关键改动点
关键数据：hidden_states, residual
SP和PP叠加的实现上，vllm在pp group中传输[T,H], 而vllm-ascend在pp group中传输[T/TP,H].

在_preprocess中的sync_and_gather_intermediate_tensors的实现上.
上游vllm做了两件事：
1. sync：把 PP receive 得到的数据复制到 ModelRunner 的本地 buffer
2. gather：如果 residual 被 native SP 切分，则把 residual all-gather 回完整形状 （hidden_states取决于 PP send/receive的时候是否用了 all_gather_group, 而这个在flashcomm中默认关闭）

vllm-ascend中flashcomm + PP的实现：
使用pp group内的点对点通信，关闭all_gather_group, 直接传输[T/TP,H]，并且不做residual的all-gather.
重写sync_and_gather_intermediate_tensors为sync_and_slice_intermediate_tensors, 只做copy和slice, 保持[T/TP,H]的形状.

`irecv_tensor_dict` 的实现位于
[`parallel_state.py`](../vllm/vllm/distributed/parallel_state.py)。它先通过 CPU
group 传 tensor metadata，再在 device group 上对每个 NPU tensor 执行
`torch.distributed.irecv`。

## 9. PP send：异步发送 hidden_states/residual

非最后 PP stage 执行模型后返回 `IntermediateTensors`，Worker 使用：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(
    output.tensors,
    all_gather_group=all_gather_group,
)
```

对应代码见
[`worker.py`](../vllm-ascend/vllm_ascend/worker/worker.py)。

`isend_tensor_dict` 的行为是：

1. 发送 tensor metadata；
2. 对 `hidden_states`、`residual` 等 tensor 调用异步 `isend`；
3. 返回通信 handle，不在当前代码行等待完成。

FlashComm 开启时 `all_gather_group=None`，所以 PP send 直接发送当前 TP rank
拥有的 local tensor。下一次 Worker 执行开始时才等待上一轮 send：

```python
if self._pp_send_work:
    for handle in self._pp_send_work:
        handle.wait()
```

这允许 PP send 与后续调度/CPU 工作重叠。
