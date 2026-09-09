## 目标1
测试PR里的代码改动中，dsa_cp，sfa_cp中的pad是否必要？

## 步骤
1、查看改动的文件确定修改范围
2、在空闲的卡上，（目前17.122空闲）进行测试，看看pad逻辑删除是否有影响。

## 结论（2026-08-19，17.122 卡 0-3 实测）

**pad 逻辑是必要的，不能删除。**

- 修改范围：`vllm_ascend/attention/context_parallel/dsa_cp.py`（`_split_full_hidden_states_for_cp`）与 `sfa_cp.py`（`_prepare_native_hidden_states`）中 `actual_tokens < num_tokens_pad` 时的 `F.pad`，`num_tokens_pad = round_up(num_input_tokens, tp_size)`。外部（cos/sin、slot_mapping）已按 `num_tokens_pad` pad 好，hidden_states 本身没有外部 pad。
- 测试：`tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py`，probe（`_cp_pad_probe`）全程打点。
- 带 pad：case1（dsv3_2_sfa_dcp + MTP）PASSED，probe 显示 pad 分支每个 worker 真实触发 3 次（如 expected=14 actual=13）。
- 删 pad：case1 FAILED —— prefill 13 tokens 未 pad 到 14，下游 `vllm_ascend/ops/triton/rope.py:397` `assert cos.shape[0] == num_tokens` 触发 AssertionError，worker 崩溃。
- dsa_cp 的 pad 与 sfa_cp 结构完全相同（同样的 round_up 对齐 + 切片校验），case2 中 DSA_SPLIT 已确认被调用（dummy/decode 阶段均 exact）；其 pad 场景与 SFA 同类，删除后会在 shape 校验处抛 RuntimeError。
- 备注：case2（deepseek_v4 w4a8）在 17.122 上 aclgraph capture 阶段报 `aclnnHcPre` 环境错误（带 pad 也崩），与 pad 无关；不加 `OMP_NUM_THREADS=1` 时 EngineCore 还会遇到 PyTorch "Invalid thread pool" 崩溃（同为环境问题）。

## 目标2
看一下目前vllm-ascend/vllm_ascend/compilation/passes/norm_quant_fusion_pass.py 这个pass是否正常工作

## 验收
打印出debug 模式下的pass是否生效。

## 结论（2026-08-20，17.110 卡 0-1 实测）

**pass 注册与调用正常，但对当前真实模型一个 pattern 都没匹配上（未生效），根因是 pattern 覆盖缺口。**

- 场景：`vllm serve` W8A8_DYNAMIC 模型（`Qwen3.5-35B-A3B-w8a8-org` TP2 + `Qwen3.5-4B-w8a8` TP2），`VLLM_LOGGING_LEVEL=DEBUG`。脚本：`scripts/verify_norm_quant_fusion.sh`。
- 对照组（`--additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false}}'`）：日志无任何 `Replaced .* patterns`（pass 未注册），chat 正常。
- 开启组（默认）：pass 正常注册并被调用，DEBUG 日志每条 split graph 打印 `Replaced 0 patterns`（norm_quant_fusion_pass.py:707），35B 模型 82 次调用（41 split graph × 2 worker）**全部 Replaced 0**，chat 正常。
- 根因（FX dump 实锤，`--compilation-config '{"debug_dump_path":...}'` + `scripts/count_norm_quant_sites.py` 统计）：图中 104 处 norm 全部是 `_C_ascend.npu_add_rms_norm_bias(bias=None)`（custom op 路径，`ops/layernorm.py:98`），0 处 `torch.ops.npu.npu_add_rms_norm`；其中 **102 处 norm→npu_dynamic_quant 相邻（本应被融合）**。但已注册 pattern 里：无 bias 的动态量化 pattern 只认 `torch.ops.npu.npu_add_rms_norm`；`_C_ascend` 路径只有 `AddRMSNormDynamicQuantPatternWithBias`，要求 bias 为真实 tensor，匹配不到 bias=None。→ custom op 开启 + W8A8 动态量化 + 无 bias（当前主路径）无 pattern 覆盖。
- 修复（已实施并验证，`norm_quant_fusion_pass.py`）：`AddRMSNormDynamicQuantPattern` / `AddRMSNormDynamicQuantSPPattern` 的 pattern 改为按 `enable_custom_op()` 匹配真实算子——custom op 开（当前主路径）用 `_C_ascend.npu_add_rms_norm_bias(bias=None)`，关则保持 `torch.ops.npu.npu_add_rms_norm`；replacement 不变。
- 修复后验证（17.122，4B-w8a8 TP2）：**48 次 `Replaced 1` + 16 次 `Replaced 2`**（共 80 处融合，修复前 82 次调用全 0），chat 正常；合成回归（`.temp/synthetic_norm_quant_test.py`，仓库外副本，本地 Qwen3.5-0.8B 构造 ModelConfig 适配离线环境）±SP × eps{1e-5,1e-6} 共 4 case 全部 PASS（matched_count=1）。
- 已知遗留：仓库内 `test_norm_quant_fusion.py` 的 `ModelConfig(dtype=dtype)` 依赖 HF 默认模型 Qwen/Qwen3-0.6B，离线环境跑不了（本次未改）；custom-op 关闭路径（VLLM_BATCH_INVARIANT=1 / A5 / 扩展缺失）的 else 分支未实测；17.110 上的 35B 日志与 dump 在已停止的容器 `xrs_vllm_main` 里（/tmp/serve_fuse_on.log、/tmp/nqf_dump），17.122 上有 4B 的等价产物（/tmp/serve_4b_fix.log、/tmp/nqf_dump_fix）。
- 备注：A5 专属 MX pattern 在 910C（A3）上不注册属预期；17.122/17.111 测试期间被宿主级未登记进程占满（每卡仅剩 1~8GB），实验在 17.110 完成。

## 目标3
不开共享专家DP + SP路径下是否会有问题？

## 验收
DP2，TP2，enable EP，共享专家DP关闭，测试qwen3.5-35b-a3b回答是否正常

### DP，w/o SP
vllm-ascend/tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py
### DP，SP
vllm-ascend/tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py
## 代码阅读
目前只有三种
SP + DP
正常路径，不需要做通信
SEDP + non SP
切分+allgather
non SEDP + non SP
TP模式
开启SP一定开启SEDP