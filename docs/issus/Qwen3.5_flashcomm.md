开启flashcomm报错

(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] WorkerProc hit an exception.
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Traceback (most recent call last):
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 992, in worker_busy_loop
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = func(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/worker/worker_base.py", line 351, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.worker.execute_model(scheduler_output)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py", line 632, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return func(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2361, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self._model_forward(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2930, in _model_forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = run_model()
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_5.py", line 671, in forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self.language_model.model(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/compilation/decorators.py", line 507, in __call__
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.forward(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 585, in forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states, residual = layer(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                               ^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 120, in forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self.linear_attn(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] WorkerProc hit an exception.
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 126, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Traceback (most recent call last):
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops.vllm.qwen_gdn_attention_core(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 992, in worker_busy_loop
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = func(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/worker/worker_base.py", line 351, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", line 1731, in qwen_gdn_attention_core
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.worker.execute_model(scheduler_output)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self._forward_core(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 275, in _forward_core
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py", line 632, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops._C_ascend.npu_causal_conv1d_custom(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return func(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] RuntimeError: _C_ascend::npu_causal_conv1d_custom() Expected a value of type 'List[int]' for argument 'query_start_loc_opt' but instead found type 'Tensor'.
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Position: 5
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2361, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Value: tensor([0, 3, 3], device='npu:0', dtype=torch.int32)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self._model_forward(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Declaration: _C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, int[] query_start_loc_opt, int[] cache_indices_opt, int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, int activation_mode, int pad_slot_id, int run_mode) -> Tensor output
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Cast error details: Unable to cast Python instance of type <class 'torch.Tensor'> to C++ type '?' (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2930, in _model_forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Traceback (most recent call last):
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = run_model()
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 992, in worker_busy_loop
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = func(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/worker/worker_base.py", line 351, in execute_model
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.worker.execute_model(scheduler_output)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py", line 632, in execute_model
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_5.py", line 671, in forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self.language_model.model(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return func(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/compilation/decorators.py", line 507, in __call__
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.forward(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2361, in execute_model
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self._model_forward(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 585, in forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states, residual = layer(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2930, in _model_forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                               ^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = run_model()
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 120, in forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self.linear_attn(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_5.py", line 671, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self.language_model.model(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/compilation/decorators.py", line 507, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.forward(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 585, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 126, in forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states, residual = layer(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops.vllm.qwen_gdn_attention_core(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                               ^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", line 1731, in qwen_gdn_attention_core
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self._forward_core(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 275, in _forward_core
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops._C_ascend.npu_causal_conv1d_custom(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 120, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self.linear_attn(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] RuntimeError: _C_ascend::npu_causal_conv1d_custom() Expected a value of type 'List[int]' for argument 'query_start_loc_opt' but instead found type 'Tensor'.
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Position: 5
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Value: tensor([0, 3, 3], device='npu:1', dtype=torch.int32)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Declaration: _C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, int[] query_start_loc_opt, int[] cache_indices_opt, int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, int activation_mode, int pad_slot_id, int run_mode) -> Tensor output
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Cast error details: Unable to cast Python instance of type <class 'torch.Tensor'> to C++ type '?' (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 126, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Traceback (most recent call last):
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops.vllm.qwen_gdn_attention_core(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 992, in worker_busy_loop
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = func(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/worker/worker_base.py", line 351, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", line 1731, in qwen_gdn_attention_core
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.worker.execute_model(scheduler_output)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self._forward_core(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 275, in _forward_core
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py", line 632, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops._C_ascend.npu_causal_conv1d_custom(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return func(*args, **kwargs)
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] RuntimeError: _C_ascend::npu_causal_conv1d_custom() Expected a value of type 'List[int]' for argument 'query_start_loc_opt' but instead found type 'Tensor'.
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Position: 5
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2361, in execute_model
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Value: tensor([0, 3, 3], device='npu:0', dtype=torch.int32)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self._model_forward(
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Declaration: _C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, int[] query_start_loc_opt, int[] cache_indices_opt, int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, int activation_mode, int pad_slot_id, int run_mode) -> Tensor output
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Cast error details: Unable to cast Python instance of type <class 'torch.Tensor'> to C++ type '?' (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2930, in _model_forward
(Worker_TP0_EP0 pid=6735) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] 
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = run_model()
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_5.py", line 671, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states = self.language_model.model(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/compilation/decorators.py", line 507, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self.forward(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 585, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     hidden_states, residual = layer(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]                               ^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 120, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self.linear_attn(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._call_impl(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return forward_call(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 126, in forward
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops.vllm.qwen_gdn_attention_core(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", line 1731, in qwen_gdn_attention_core
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     self._forward_core(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/home/x50063850/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py", line 275, in _forward_core
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     torch.ops._C_ascend.npu_causal_conv1d_custom(
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]     return self._op(*args, **kwargs)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] RuntimeError: _C_ascend::npu_causal_conv1d_custom() Expected a value of type 'List[int]' for argument 'query_start_loc_opt' but instead found type 'Tensor'.
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Position: 5
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Value: tensor([0, 3, 3], device='npu:1', dtype=torch.int32)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Declaration: _C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, int[] query_start_loc_opt, int[] cache_indices_opt, int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, int activation_mode, int pad_slot_id, int run_mode) -> Tensor output
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] Cast error details: Unable to cast Python instance of type <class 'torch.Tensor'> to C++ type '?' (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)
(Worker_TP1_EP1 pid=6736) ERROR 07-20 14:20:39 [multiproc_executor.py:1000] 
(EngineCore pid=6709) ERROR 07-20 14:20:39 [dump_input.py:72] Dumping input data for V1 LLM engine (v0.24.0) with config: model='/home/weights/Qwen/Qwen3.5-35B-A3B', speculative_config=None, tokenizer='/home/weights/Qwen/Qwen3.5-35B-A3B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=262144, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=npu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_verbose=False), seed=0, served_model_name=qwen, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'vllm_ascend.compilation.compiler_interface.AscendCompiler', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto'), 
(EngineCore pid=6709) ERROR 07-20 14:20:39 [dump_input.py:79] Dumping scheduler output for model execution: SchedulerOutput(scheduled_new_reqs=[NewRequestData(req_id=cmpl-ab5d3216103500f5-0-bc2e5ddc,prompt_token_ids_len=3,prefill_token_ids_len=None,mm_features=[],sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[248044], bad_words=[], thinking_token_budget=None, include_stop_str_in_output=False, ignore_eos=False, max_tokens=10, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, structured_outputs=None, extra_args=None),block_ids=([1], [2], [3], [4]),num_computed_tokens=0,lora_request=None,prompt_embeds_shape=None)], scheduled_cached_reqs=CachedRequestData(req_ids=[],resumed_req_ids=set(),new_token_ids_lens=[],all_token_ids_lens={},new_block_ids=[],num_computed_tokens=[],num_output_tokens=[]), num_scheduled_tokens={cmpl-ab5d3216103500f5-0-bc2e5ddc: 3}, total_num_scheduled_tokens=3, scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={}, num_common_prefix_blocks=[0, 0, 0, 0], finished_req_ids=[], free_encoder_mm_hashes=[], preempted_req_ids=[], has_structured_output_requests=false, pending_structured_output_tokens=false, num_invalid_spec_tokens=null, kv_connector_metadata=null, ec_connector_metadata=null, new_block_ids_to_zero=[1], num_spec_tokens_to_schedule=0)
(EngineCore pid=6709) ERROR 07-20 14:20:39 [dump_input.py:81] Dumping scheduler stats: SchedulerStats(num_running_reqs=1, num_waiting_reqs=0, num_skipped_waiting_reqs=0, step_counter=0, current_wave=0, kv_cache_usage=0.004020100502512558, prefix_cache_stats=PrefixCacheStats(reset=False, requests=0, queries=0, hits=0, preempted_requests=0, preempted_queries=0, preempted_hits=0), connector_prefix_cache_stats=None, kv_cache_eviction_events=[], spec_decoding_stats=None, kv_connector_stats=None, waiting_lora_adapters={}, running_lora_adapters={}, cudagraph_stats=None, perf_stats=None)
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233] EngineCore encountered a fatal error.
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233] Traceback (most recent call last):
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/engine/core.py", line 1224, in run_engine_core
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     engine_core.run_busy_loop()
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/engine/core.py", line 1265, in run_busy_loop
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     self._process_engine_step()
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/engine/core.py", line 1304, in _process_engine_step
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     outputs, model_executed = self.step_fn()
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]                               ^^^^^^^^^^^^^^
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/engine/core.py", line 497, in step
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     model_output = future.result()
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]                    ^^^^^^^^^^^^^^^
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 91, in result
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     return super().result()
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]            ^^^^^^^^^^^^^^^^
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/usr/local/python3.12.13/lib/python3.12/concurrent/futures/_base.py", line 449, in result
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     return self.__get_result()
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/usr/local/python3.12.13/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     raise self._exception
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 95, in _wait_for_response
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     response = self.aggregate(self.get_response())
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]                               ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 391, in get_response
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233]     raise RuntimeError(
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233] RuntimeError: Worker failed with error '_C_ascend::npu_causal_conv1d_custom() Expected a value of type 'List[int]' for argument 'query_start_loc_opt' but instead found type 'Tensor'.
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233] Position: 5
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233] Value: tensor([0, 3, 3], device='npu:0', dtype=torch.int32)
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233] Declaration: _C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, int[] query_start_loc_opt, int[] cache_indices_opt, int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, int activation_mode, int pad_slot_id, int run_mode) -> Tensor output
(EngineCore pid=6709) ERROR 07-20 14:20:39 [core.py:1233] Cast error details: Unable to cast Python instance of type <class 'torch.Tensor'> to C++ type '?' (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)', please check the stack trace above for the root cause
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704] AsyncLLM output_handler failed.
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704] Traceback (most recent call last):
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/engine/async_llm.py", line 660, in output_handler
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704]     outputs = await engine_core.get_output_async()
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704]   File "/home/x50063850/vllm-workspace/vllm/vllm/v1/engine/core_client.py", line 1061, in get_output_async
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704]     raise self._format_exception(outputs) from None
(APIServer pid=6658) ERROR 07-20 14:20:39 [async_llm.py:704] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.