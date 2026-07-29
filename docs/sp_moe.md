
## SP FX Graph
**moe pass**
### with DP = 2, EP on before apply
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %all_gather : [num_users=1] = call_function[target=torch.ops.vllm.all_gather.default](args = (%alias, 0, 2, tp:0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %slice_1 : [num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%all_gather, 0, 0, %arg1_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %maybe_chunk_residual_1 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%slice_1, %getitem_15), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %npu_add_rms_norm_bias_1 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%slice_1, %maybe_chunk_residual_1, %arg13_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_18 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_1, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_20 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_1, 2), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %unquantized_gemm_3 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%getitem_18, %arg14_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %qkv_rmsnorm_rope_default_46 : [num_users=3] = call_function[target=torch.ops.vllm.qkv_rmsnorm_rope.default](args = (%unquantized_gemm_3, %arg9_1, %arg7_1, %arg15_1, %arg16_1, 2048, 256, 128, 1e-06), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_1052 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_1053 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_1054 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 2), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %empty_1 : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%arg1_1, 2048],), kwargs = {dtype: torch.bfloat16, device: npu:2, pin_memory: False})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %view_17 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1052, [-1, 16, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %view_19 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1053, [-1, 2, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %view_20 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1054, [-1, 2, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %auto_functionalized_v2_2 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.unified_attention_with_output.default,), kwargs = {query: %view_17, key: %view_19, value: %view_20, layer_name: model.layers.1.self_attn.attn, output_scale: None, kv_cache_dummy_dep: None, _output_base_index: 0, _output_size: (%arg1_1, 16, 128), _output_stride: (2048, 128, 1), _output_storage_offset: 0, _output_block_scale_base_index: None, _all_bases: [%empty_1]})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_31 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_2, 1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %detach_5 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%getitem_31,), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %view_23 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%detach_5, [-1, 16, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %view_24 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%view_23, [-1, 2048]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %unquantized_gemm_4 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%view_24, %arg17_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %all_reduce_1 : [num_users=2] = call_function[target=torch.ops.vllm.all_reduce.default](args = (%unquantized_gemm_4, tp:0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %maybe_chunk_residual_2 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%all_reduce_1, %getitem_20), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %npu_add_rms_norm_bias_2 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%all_reduce_1, %maybe_chunk_residual_2, %arg18_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_32 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_2, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_34 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_2, 2), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %sequence_parallel_chunk_impl_1 : [num_users=2] = call_function[target=torch.ops.vllm.sequence_parallel_chunk_impl.default](args = (%getitem_32,), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %unquantized_gemm_5 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%sequence_parallel_chunk_impl_1, %arg19_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %auto_functionalized_v2_3 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.moe_forward.default,), kwargs = {router_logits: %unquantized_gemm_5, shared_experts_input: None, input_ids: None, layer_name: from_forward_context, hidden_dim_unpadded: 0, _hidden_states_base_index: 0, _all_bases: [%sequence_parallel_chunk_impl_1]})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %getitem_35 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_3, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %alias_1 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%getitem_35,), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:47 [sequence_parallelism.py:350]     %all_gather_1 : [num_users=1] = call_function[target=torch.ops.vllm.all_gather.default](args = (%alias_1, 0, 2, tp:0), kwargs = {})


### with DP = 2, EP on after apply
layer = 48
SP pass replace 48 
MoE pass replace 96
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %maybe_chunk_residual_default_95 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%alias, %getitem_1153), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %npu_add_rms_norm_bias_default_95 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%alias, %maybe_chunk_residual_default_95, %arg13_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_1247 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_95, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_1248 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_95, 2), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %all_gather_default_95 : [num_users=1] = call_function[target=torch.ops.vllm.all_gather.default](args = (%getitem_1247, 0, 2, tp:0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %unquantized_gemm_3 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%all_gather_default_95, %arg14_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %qkv_rmsnorm_rope_default_46 : [num_users=3] = call_function[target=torch.ops.vllm.qkv_rmsnorm_rope.default](args = (%unquantized_gemm_3, %arg9_1, %arg7_1, %arg15_1, %arg16_1, 2048, 256, 128, 1e-06), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_1052 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_1053 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_1054 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 2), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %empty_1 : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%arg1_1, 2048],), kwargs = {dtype: torch.bfloat16, device: npu:2, pin_memory: False})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %view_17 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1052, [-1, 16, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %view_19 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1053, [-1, 2, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %view_20 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1054, [-1, 2, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %auto_functionalized_v2_2 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.unified_attention_with_output.default,), kwargs = {query: %view_17, key: %view_19, value: %view_20, layer_name: model.layers.1.self_attn.attn, output_scale: None, kv_cache_dummy_dep: None, _output_base_index: 0, _output_size: (%arg1_1, 16, 128), _output_stride: (2048, 128, 1), _output_storage_offset: 0, _output_block_scale_base_index: None, _all_bases: [%empty_1]})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_31 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_2, 1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %detach_5 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%getitem_31,), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %view_23 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%detach_5, [-1, 16, 128]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %view_24 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%view_23, [-1, 2048]), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %unquantized_gemm_4 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%view_24, %arg17_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %reduce_scatter_default_46 : [num_users=2] = call_function[target=torch.ops.vllm.reduce_scatter.default](args = (%unquantized_gemm_4, 0, 2, tp:0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %maybe_chunk_residual_default_46 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%reduce_scatter_default_46, %getitem_1248), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %npu_add_rms_norm_bias_default_46 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%reduce_scatter_default_46, %maybe_chunk_residual_default_46, %arg18_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_1150 : [num_users=2] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_46, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_1151 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_46, 2), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %unquantized_gemm_5 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%getitem_1150, %arg19_1), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %auto_functionalized_v2_3 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.moe_forward.default,), kwargs = {router_logits: %unquantized_gemm_5, shared_experts_input: None, input_ids: None, layer_name: from_forward_context, hidden_dim_unpadded: 0, _hidden_states_base_index: 0, _all_bases: [%getitem_1150]})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %getitem_35 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_3, 0), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %alias_1 : [num_users=2] = call_function[target=torch.ops.aten.alias.default](args = (%getitem_35,), kwargs = {})
(Worker_DP1_TP0_EP2 pid=409162) DEBUG 07-29 10:58:54 [sequence_parallelism_moe.py:194]     %maybe_chunk_residual_default_94 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%alias_1, %getitem_1151), kwargs = {})

### DP = 1, TP = 2, EP on
SP pass replace 96

(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %maybe_all_reduce_tensor_model_parallel : [num_users=1] = call_function[target=torch.ops.vllm.maybe_all_reduce_tensor_model_parallel.default](args = (%getitem_16,), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %alias : [num_users=2] = call_function[target=torch.ops.aten.alias.default](args = (%maybe_all_reduce_tensor_model_parallel,), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %maybe_chunk_residual_1 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%alias, %getitem_15), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %npu_add_rms_norm_bias_1 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%alias, %maybe_chunk_residual_1, %arg13_1), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_18 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_1, 0), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_20 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_1, 2), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %unquantized_gemm_3 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%getitem_18, %arg14_1), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %qkv_rmsnorm_rope_default_46 : [num_users=3] = call_function[target=torch.ops.vllm.qkv_rmsnorm_rope.default](args = (%unquantized_gemm_3, %arg9_1, %arg7_1, %arg15_1, %arg16_1, 2048, 256, 128, 1e-06), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_1052 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 0), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_1053 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 1), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_1054 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 2), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %empty_1 : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%arg1_1, 2048],), kwargs = {dtype: torch.bfloat16, device: npu:1, pin_memory: False})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %view_17 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1052, [-1, 16, 128]), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %view_19 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1053, [-1, 2, 128]), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %view_20 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1054, [-1, 2, 128]), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %auto_functionalized_v2_2 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.unified_attention_with_output.default,), kwargs = {query: %view_17, key: %view_19, value: %view_20, layer_name: model.layers.1.self_attn.attn, output_scale: None, kv_cache_dummy_dep: None, _output_base_index: 0, _output_size: (%arg1_1, 16, 128), _output_stride: (2048, 128, 1), _output_storage_offset: 0, _output_block_scale_base_index: None, _all_bases: [%empty_1]})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_31 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_2, 1), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %detach_5 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%getitem_31,), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %view_23 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%detach_5, [-1, 16, 128]), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %view_24 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%view_23, [-1, 2048]), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %unquantized_gemm_4 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%view_24, %arg17_1), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %all_reduce_1 : [num_users=2] = call_function[target=torch.ops.vllm.all_reduce.default](args = (%unquantized_gemm_4, tp:0), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %maybe_chunk_residual_2 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%all_reduce_1, %getitem_20), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %npu_add_rms_norm_bias_2 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%all_reduce_1, %maybe_chunk_residual_2, %arg18_1), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_32 : [num_users=2] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_2, 0), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_34 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_2, 2), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %unquantized_gemm_5 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%getitem_32, %arg19_1), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %auto_functionalized_v2_3 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.moe_forward.default,), kwargs = {router_logits: %unquantized_gemm_5, shared_experts_input: None, input_ids: None, layer_name: from_forward_context, hidden_dim_unpadded: 0, _hidden_states_base_index: 0, _hidden_states_alias: True, _all_bases: [%getitem_32]})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %getitem_35 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_3, 0), kwargs = {})
(Worker_TP1_EP1 pid=408396) DEBUG 07-29 10:54:22 [sequence_parallelism.py:350]     %maybe_all_reduce_tensor_model_parallel_1 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_all_reduce_tensor_model_parallel.default](args = (%getitem_35,), kwargs = {})

#### after
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %reduce_scatter_default_94 : [num_users=2] = call_function[target=torch.ops.vllm.reduce_scatter.default](args = (%getitem_16, 0, 2, tp:0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %maybe_chunk_residual_default_94 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%reduce_scatter_default_94, %getitem_1248), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %npu_add_rms_norm_bias_default_94 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%reduce_scatter_default_94, %maybe_chunk_residual_default_94, %arg13_1), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_1245 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_94, 0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_1246 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_94, 2), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %all_gather_default_94 : [num_users=1] = call_function[target=torch.ops.vllm.all_gather.default](args = (%getitem_1245, 0, 2, tp:0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %unquantized_gemm_3 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%all_gather_default_94, %arg14_1), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %qkv_rmsnorm_rope_default_46 : [num_users=3] = call_function[target=torch.ops.vllm.qkv_rmsnorm_rope.default](args = (%unquantized_gemm_3, %arg9_1, %arg7_1, %arg15_1, %arg16_1, 2048, 256, 128, 1e-06), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_1052 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_1053 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 1), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_1054 : [num_users=1] = call_function[target=operator.getitem](args = (%qkv_rmsnorm_rope_default_46, 2), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %empty_1 : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([%arg1_1, 2048],), kwargs = {dtype: torch.bfloat16, device: npu:0, pin_memory: False})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %view_17 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1052, [-1, 16, 128]), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %view_19 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1053, [-1, 2, 128]), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %view_20 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%getitem_1054, [-1, 2, 128]), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %auto_functionalized_v2_2 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.unified_attention_with_output.default,), kwargs = {query: %view_17, key: %view_19, value: %view_20, layer_name: model.layers.1.self_attn.attn, output_scale: None, kv_cache_dummy_dep: None, _output_base_index: 0, _output_size: (%arg1_1, 16, 128), _output_stride: (2048, 128, 1), _output_storage_offset: 0, _output_block_scale_base_index: None, _all_bases: [%empty_1]})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_31 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_2, 1), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %detach_5 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%getitem_31,), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %view_23 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%detach_5, [-1, 16, 128]), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %view_24 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%view_23, [-1, 2048]), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %unquantized_gemm_4 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%view_24, %arg17_1), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %reduce_scatter_default_93 : [num_users=2] = call_function[target=torch.ops.vllm.reduce_scatter.default](args = (%unquantized_gemm_4, 0, 2, tp:0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %maybe_chunk_residual_default_93 : [num_users=1] = call_function[target=torch.ops.vllm.maybe_chunk_residual.default](args = (%reduce_scatter_default_93, %getitem_1246), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %npu_add_rms_norm_bias_default_93 : [num_users=2] = call_function[target=torch.ops._C_ascend.npu_add_rms_norm_bias.default](args = (%reduce_scatter_default_93, %maybe_chunk_residual_default_93, %arg18_1), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_1243 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_93, 0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_1244 : [num_users=1] = call_function[target=operator.getitem](args = (%npu_add_rms_norm_bias_default_93, 2), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %all_gather_default_93 : [num_users=2] = call_function[target=torch.ops.vllm.all_gather.default](args = (%getitem_1243, 0, 2, tp:0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %unquantized_gemm_5 : [num_users=1] = call_function[target=torch.ops.vllm.unquantized_gemm.default](args = (%all_gather_default_93, %arg19_1), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %auto_functionalized_v2_3 : [num_users=1] = call_function[target=torch.ops.higher_order.auto_functionalized_v2](args = (vllm.moe_forward.default,), kwargs = {router_logits: %unquantized_gemm_5, shared_experts_input: None, input_ids: None, layer_name: from_forward_context, hidden_dim_unpadded: 0, _hidden_states_base_index: 0, _hidden_states_alias: True, _all_bases: [%all_gather_default_93]})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %getitem_35 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_v2_3, 0), kwargs = {})
(Worker_TP0_EP0 pid=408395) DEBUG 07-29 10:54:24 [sequence_parallelism.py:362]     %reduce_scatter_default_92 : [num_users=2] = call_function[target=torch.ops.vllm.reduce_scatter.default](args = (%getitem_35, 0, 2, tp:0), kwargs = {})


### 不同情况下fused MOE的输入/输出都是什么？
fused MOE外部其实就三种模式
重点关注[prepare_finalize.py](../vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py#L:241)
，关注算子的输入输出。
use_sequence_parallel_moe 要求 DP>1 && TP>1 && enableEP （vllm main分支应该DP>1要求没了）
1、SP+EP Moe模式
fused MOE接受local token-shard tensor，(所以MOE pass里要把前面的allgather和分片换掉)，里面实现了all2all通信，输出得到依然是token分片tensor。这个时候就不要再做TP allreduce了。（这个时候就要MOE pass：1、去除多余的MOE之前的allgather 2、替换MOE之后的allgather到rms norm之后）
2、TP模式
外部实现allgather，allreduce。fused MOE接受完整tensor。这个时候SP pass生效两次（attn之前，MOE之后）
3、没有SP的EP
接受完整tensor(内部不发生all2all通信，需要在外部allgather & allreduce 作为expert的dispatch和combine)
### 上游的实现
上游通过allgather之后slice还是做成分片的输入。

#### 分情况讨论
1、DP=2, TP=2, enableEP=True -> EP=4, TP=1
这种情况下EP=4， fused MOE接受local token-shard tensor，(所以MOE pass里要把前面的allgather和分片换掉)，里面实现了all2all通信，输出得到依然是token分片tensor。这个时候就不要再做TP allreduce了。

2、DP=1, TP=2, enableEP -> TP=1, EP=2， 此时MoE是non SP的
TP模式下：sp pass替换两次一个block，fused MOE之后会有一个TP allreduce。

3、enableEP=false同理。TP模式

4、EP size这个变量是干嘛的？和enable EP区别
没了，用户侧只支持enable ep。内部通过EP size=DP * TP计算。

#### 遗留问题
不同的内部通信方式是否会导致输入输出不同？需要测试。