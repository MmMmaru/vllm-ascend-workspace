```
NPUWorker.execute_model
  ↓
NPUModelRunner.execute_model
  ↓
准备 scheduler/batch/attention metadata
  ↓
_pad_for_sequence_parallelism
  ↓
_preprocess
  ↓
set_ascend_forward_context
  ↓
_model_forward
  ↓
Qwen3MoeForCausalLM.forward
  ↓
Qwen3MoeModel.forward
  ↓
Transformer layer
  ↓
Ascend Linear/RMSNorm/MoE/Attention
  ↓
torch.ops.vllm.xxx
  ↓
通过vllm direct 详见
```

通信算子注册：[vllm-ascend/vllm_ascend/ops/register_custom_ops.py](..vllm-ascend/vllm_ascend/ops/register_custom_ops.py#L220)。
模型算子注册：[utils.py](..vllm-ascend/vllm_ascend/utils.py#L638)

```