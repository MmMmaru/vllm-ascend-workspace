# Sequence-parallel communication in parallel linear layers

This document describes the sequence-parallel (SP) implementation at commit
`72cd5424d` and the boundary of the parallel-linear refactor.

## Existing implementations

vLLM has two independent features called sequence parallelism:

- The compilation pass in
  `vllm/compilation/passes/fusion/sequence_parallelism.py` rewrites
  all-reduce and RMSNorm patterns to reduce-scatter, local normalization, and
  all-gather. It is model-independent and requires full-graph compilation.
- MoE sequence parallelism is selected by
  `ParallelConfig.use_sequence_parallel_moe`. Models shard their token
  dimension so expert work can use the tensor-parallel ranks as sequence
  ranks. Most of this path was implemented explicitly in model code.

The model-side implementations fall into the following groups.

| Implementation | Model families | Model-side communication |
| --- | --- | --- |
| Legacy MoE blocks | AXK1, GPT-OSS, GraniteMoE, InternS1 Pro, Llama 4, MiMo V2, Nemotron-H, OpenPangu, Qwen3-MoE | `sequence_parallel_chunk` before MoE and all-gather after MoE |
| Attention-to-MoE bridge | DeepSeek V2, Qwen3-Next, Qwen3.5 (through Qwen3-Next), DeepSeek V3.2 | all-gather before attention and reduce-scatter after its row-parallel output |
| Transformers backend | Generic Transformers MoE fuser | `sequence_parallel_chunk` and all-gather in the fuser |
| New model implementation | Kimi K3 | all-gather/reduce-scatter around attention and around sharded dense MLPs; additional gathers at model and MTP boundaries |
| New model implementation | DeepSeek V4 | all-gather/reduce-scatter around attention; additional gathers in model, dSPARK, and MTP boundaries |

The inventory excludes collectives that do not represent token-dimension SP,
including vocabulary/logits gathers, vision Q/K gathers, pipeline transfers,
and Inkling's hidden-dimension reduce-scatter/all-gather pair.

Boundary gathers are not parallel-linear operations. Final hidden states,
auxiliary hidden states, MTP inputs/outputs, and pipeline boundaries may still
need explicit model orchestration after the linear migration.

## Unified entry point

SP collectives live in `vllm.distributed.communication_op`:

- `sequence_parallel_all_gather` gathers dimension 0 and optionally removes
  padding before computation.
- `sequence_parallel_reduce_scatter` pads dimension 0 to the TP world size,
  sums partial results, and scatters token shards.
- Both operations first use a device communicator's custom SP collective when
  available, then fall back to the regular TP collective.

`vllm.models.common.ops.sequence_parallel` remains as a compatibility facade
for existing Kimi K3 and DeepSeek V4 call sites.

Parallel linear layers expose one `sequence_parallel` switch:

```text
local token shard
  -> ColumnParallelLinear.prepare_input() -> all-gather
  -> column-parallel computation
  -> row-parallel computation
  -> RowParallelLinear.reduce_output() -> reduce-scatter
  -> local token shard
```

The default is `False`, so existing layers retain all current TP behavior.
When enabled, a column-parallel layer gathers its input token shards before
the quantization method runs. A row-parallel layer reduce-scatters its partial
output instead of all-reducing it. `MergedColumnParallelLinear` and
`QKVParallelLinear` expose the same constructor option.

`sequence_parallel_unpadded_size` can be passed to a column-parallel forward
call when attention metadata requires padding rows to be removed before the
GEMM. Reduce-scatter always restores divisibility by padding dimension 0.

LoRA column and row wrappers call the base layer's `prepare_input` and
`reduce_output` methods. This keeps the communication policy in the parallel
linear layer for both base and LoRA execution.

## Model migration boundary

This change only establishes the common implementation. No model enables the
new linear option yet. A later migration should, per model:

1. Enable SP on the first column-parallel projection that consumes a token
   shard.
2. Enable SP on the row-parallel projection whose partial result returns to a
   token shard.
3. Remove the matching model-side all-gather, padding, reduce-scatter, and
   `reduce_results=False` workaround.
4. Retain and separately review model-boundary gathers that are not part of a
   column/row linear pair.
