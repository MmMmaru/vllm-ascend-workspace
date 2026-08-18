## patch文件

### 模型相关

- `worker/patch_deepseek_v2.py`
    - DeepSeek V2/V3、Kimi K2/K2.6、GLM-5.1/5.2 使用的 `DeepseekV2MLAAttention`、`DeepseekV2Model`。
    - GLM-5.2 的 shared indexer 跳过 `Indexer` 初始化；`DeepseekV2Model.forward` 补充 PP、TP 下的 hidden state / aux hidden state 处理。
- `worker/patch_process_weights_after_loading.py`
    - 使用 Ascend `DSAAttention` 的模型（DeepSeek V3.2、DeepSeek V4、GLM DSA 等）。
    - 将 `DSAAttention.process_weights_after_loading()` 接入 vLLM 的权重加载后处理流程。
- `worker/patch_kimi_k25.py`
    - Kimi K2.5：`kimi_k25_vit.Learnable2DInterpPosEmbDivided_fixed.forward`。
    - 将不支持的 NPU 插值路径改为 Ascend 兼容实现，并处理 RoPE 形状插值。
- `platform/patch_kv_cache_coordinator.py`
    - DeepSeek V4：按 DSA、SWA、EAGLE 的 KV cache 分组处理 prefix cache 命中。
    - 混合 Mamba 模型：修正 FullAttention / Mamba KV cache 的命中长度计算。
- `worker/patch_minimax_m2.py`、`worker/patch_minimax_m2_linear_attn.py`
    - MiniMax-M2：`MiniMaxM2MoE.forward`、`MiniMaxM2Attention.forward`、`MiniMaxM2Model.load_weights`，以及线性注意力 RMSNorm 的初始化、权重加载和 q/k norm。
    - 处理 NPU 上的 FP8 权重反量化、MoE TP 通信和融合 attention。
- `platform/patch_minimax_m2_config.py`
    - MiniMax-M2：模型配置的 FP8 校验和 ACL graph 的 HCCL 配置。
- `worker/patch_qwen3_5.py`
    - Qwen3.5、Qwen3-Next：GDN、`Qwen3NextAttention.forward`、Qwen3.5 MTP，以及 Ascend 融合 attention 路径。
- `worker/patch_qwen3_dflash.py`、`worker/patch_v2/patch_dflash_speculator.py`
    - Qwen3 DFlash：`DFlashQwen3Model.precompute_and_store_context_kv` 和 v2 DFlash 的 CUDA graph manager。
    - 前者替换为 NPU 可用的 KV 预计算/写入流程，后者切换到 ACL graph manager。
- `worker/patch_qwen3vl.py`
    - Qwen3-VL、Qwen3-VL-MoE：`Qwen3VLForConditionalGeneration` 的 DeepStack embedding、Qwen3-VL-MoE 的 PP layer boundary，以及 Qwen3/Qwen3-MoE attention 的融合 qkv、RMSNorm、M-RoPE。
- `worker/patch_idex_310.py`
    - 310P 上的 Qwen3-VL、Qwen3.5 / Qwen3-Next：GDN chunk index、GDN attention backend/state，以及 Qwen3-VL 的视觉 RoPE。
- `worker/patch_step3p5.py`
    - Step3.5、Step3.7：`Step3p5Attention.forward`，接入 `split_qkv_rmsnorm_rope`。

### MTP / Eagle / DSpark

- `platform/patch_pp_mtp.py`
    - 所有启用 PP + MTP/Eagle 的目标模型：DeepSeek、Qwen3-Next、Qwen3.5、GLM、MiMo、Pangu、ERNIE、Nemotron-H、Exaone、LongCat、Step3.5/3.7 等。
    - 修改 `ModelRunnerOutput`、EngineCore、Scheduler 的 draft token 回传、PP in-flight fence 和本地 drafter 的 PP 校验。
- `platform/patch_speculative_config.py`
    - DeepSeek V4 DSpark、Qwen3 DSpark：为 DSpark draft config 补齐 `ptd_token_id`。
- `worker/patch_eagle3_init.py`
    - Eagle3 的 LLaMA/Qwen 目标模型、DeepSeek V2/V3 目标模型，以及 Kimi K2/K2.6 等 DeepSeek 架构目标模型。
    - 修正 PP 下 Eagle3 draft model 的全局 layer 数和 checkpoint layer prefix。
- `worker/patch_eagle3_pp_aux.py`
    - Eagle3 + PP：DeepSeek V2/V3、Kimi K2/K2.6，以及基于 `EagleModelMixin` 的 MiniMax、LLaMA、Qwen 等模型。
    - 通过 `IntermediateTensors` 在 PP stage 之间传递 aux hidden states。
- `worker/patch_v2/patch_eagle_speculator.py`
    - v2 Eagle speculative decoding：将 upstream CUDA graph manager 替换为 Ascend ACL graph manager。
- `vllm_ascend/models/deepseek_v4.py`、`deepseek_v4_mtp.py`、`deepseek_v4_dspark.py`、`qwen3_dspark.py`
    - DeepSeek V4、DeepSeek V4 DSpark、Qwen3 DSpark 是 Ascend 自己注册的模型实现，不属于 patch 文件；其运行时仍会叠加上面的通用 MoE、MTP/Eagle、权重加载和 KV cache patch。

## patch模型

ds v4 model自己实现
decoder layer里无allgather逻辑，放到 model forward / PP 边界处理

- DeepSeek V4：`worker/patch_process_weights_after_loading.py`、`platform/patch_kv_cache_coordinator.py`；模型主体在 `vllm_ascend/models/deepseek_v4.py` 自己实现。
- DeepSeek V4 MTP / DSpark：`platform/patch_speculative_config.py`、`platform/patch_pp_mtp.py`；DSpark draft 模型主体在 `vllm_ascend/models/deepseek_v4_dspark.py` 自己实现。

ds v2 model自己实现

- DeepSeek V2/V3：`worker/patch_deepseek_v2.py`；MLA 初始化和 `DeepseekV2Model.forward` 走 patch，模型其余部分使用 upstream。
- DeepSeek V3.2：`worker/patch_process_weights_after_loading.py`、`worker/patch_v2/patch_attn_utils.py`；分别处理 DSA 权重后处理和 SFA indexer cache backend。
- GLM DSA：`worker/patch_process_weights_after_loading.py`；将 DSA 权重后处理接入模型加载流程。

Kimi K2.5

- `worker/patch_kimi_k25.py`：VIT 位置插值。
- `worker/patch_eagle3_init.py`、`worker/patch_eagle3_pp_aux.py`：启用 Eagle3 + PP 时的 layer index 和 aux hidden states。
- `platform/patch_kv_cache_coordinator.py`、`platform/patch_mamba_config.py`、`platform/patch_mamba_manager.py`：使用混合 attention/Mamba、prefix cache 或 KV transfer 时生效。

MiniMax-M2

- `platform/patch_minimax_m2_config.py`：FP8 校验和 ACL graph 的 HCCL 配置。
- `worker/patch_minimax_m2.py`、`worker/patch_minimax_m2_linear_attn.py`：MoE、attention、FP8 load、线性 attention RMSNorm。
- `worker/patch_eagle3_pp_aux.py`：Eagle3 + PP aux hidden states。

Qwen3.5 / Qwen3-Next

- `worker/patch_qwen3_5.py`：GDN、attention、MTP。
- `platform/patch_mamba_config.py`、`platform/patch_mamba_manager.py`、`worker/patch_mamba_utils.py`：Mamba KV cache、state copy 和 KV transfer。
- 310P 额外使用 `worker/patch_idex_310.py`。

Qwen3-VL / Qwen3-VL-MoE

- `worker/patch_qwen3vl.py`：DeepStack embedding、PP layer boundary、融合 attention。
- `platform/patch_vision.py`：视觉模型公共的 `FusedInputNorm.forward` 兼容处理。
- 310P 额外使用 `worker/patch_idex_310.py`：视觉 RoPE 和 GDN 相关实现。

Qwen3 DFlash

- `worker/patch_qwen3_dflash.py`：DFlash context KV 预计算和写入。
- v2 runner 额外使用 `worker/patch_v2/patch_dflash_speculator.py`：ACL graph manager。

Step3.5 / Step3.7

- `worker/patch_step3p5.py`：attention 融合算子。
- `platform/patch_pp_mtp.py`：使用 MTP 或 PP + MTP 时的 scheduler / output 兼容处理。

通用 patch（不绑定单一模型）

- MoE：`platform/patch_fused_moe.py`、`worker/patch_fused_moe.py`，将 `FusedMoEFactory` 指向 `AscendMoERunner`。
- Mamba：`platform/patch_mamba_config.py`、`platform/patch_mamba_manager.py`、`worker/patch_mamba_utils.py`，对所有使用 vLLM Hybrid Mamba cache 的模型生效。
- Triton / 采样：`worker/patch_triton.py`、`worker/patch_v2/patch_triton.py`、`worker/patch_rejection_sampler.py`，替换 NPU 不兼容或性能不足的 Triton、采样和惩罚 kernel。
- v1/v2 worker：`worker/patch_cudagraph.py`、`worker/patch_bind_kv_cache.py`、`worker/patch_distributed.py`、`worker/patch_v2/patch_attn_utils.py`、`worker/patch_v2/patch_block_table.py`、`worker/patch_v2/patch_input_batch.py`、`worker/patch_v2/patch_model_state.py`、`worker/patch_v2/patch_uva.py`，提供 Ascend 的 graph、通信、KV cache、输入 batch、attention metadata 和 UVA 兼容层。
- 平台/运行时：`platform/patch_use_v2_model_runner.py`、`platform/patch_torch_accelerator.py`、`platform/patch_structured_output.py`、`platform/patch_dp_device_ids.py`、`platform/patch_vision.py` 等，按 NPU、运行模式或请求特性生效，不应归到某个单一模型。
