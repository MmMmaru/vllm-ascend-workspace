### 09-09 09:40
9.143 上验证 SP-MoE 开关 patch 真正生效：同步 workspace（vllm 全量+vllm-ascend 增量）并以 COMPILE_CUSTOM_KERNELS=0 重装可编辑包
UT `test_patch_parallel_config.py` 5 passed；Qwen3-30B-A3B（DP1/TP4/EP/flashcomm）serve 日志打出 `Sequence-parallel MoE is enabled` 且试请求输出正常。验证完服务已停，卡已释放。
注意：远端 PYTHONPATH 必须用 `:$PYTHONPATH` 追加（直接赋值会丢 CANN 路径报 `No module named 'acl'`）；remote 下发含 `$` 的命令要在本地转义。

### 08-28 13:42

- 新增 `scripts/prepare_lm_eval_gsm8k.py` 和全量入口
  `scripts/lm_eval.sh`：只读转换本地 GSM8K Parquet 的 1319 条测试样本为
  JSONL，自动生成 `generate_until` task，默认不限制样本数。
- 保持 `/mnt/share/datasets/gsm8k` 不变；结果写入 `.temp/lm_eval/`，日志
  写入 `.log/`，使用 `tokenizer_backend=none` 访问本地 completions 服务。
- 已在 17.111 验证完整转换和 `LIMIT=1` 评测通过，strict/flexible
  `exact_match=1.0`；随后全量评测 job `j-20260828-134235-01` 完成
  1319/1319 条，耗时约 39 分钟，strict `exact_match=0.8643`、flexible
  `exact_match=0.5345`，服务已停止。

### 08-27

- 在 17.111（A3/CANN 9.1，4–7 卡）为 `docs/plans/refactor_sp.md` 的 SP 线性层重构完成远端环境准备：重建后的容器重新同步 workspace，`modelscope==1.35.3` 安装成功，`vllm-ascend` 自定义扩展以 `COMPILE_CUSTOM_KERNELS=1` 编译成功；使用当前 vLLM 源码时需显式设置 `VLLM_VERSION=0.27.2`，否则旧 PCP 导入路径不存在。
- `tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py` 收集通过（4 tests）；Qwen3.5-35B-A3B precision E2E（job `j-20260827-165318-01`）在模型构造阶段失败：vLLM 的 `QwenGatedDeltaNetAttention` 向 `MergedColumnParallelLinear` 传入 `sequence_parallel`，而当前 `vllm-ascend` 的 `AscendMergedColumnParallelLinear` 不接受该参数（`vllm_ascend/ops/linear.py:222`）。
- 为验证通用 harness 又运行 Qwen3-30B-A3B DP2/TP2/EP precision E2E（job `j-20260827-170022-01`），同样未进入精度断言：`AscendQKVParallelLinear` 未初始化 vLLM 新线性层要求的 `sequence_parallel` 属性，前向时触发 `AttributeError`。因此本轮没有可报告的 max/mean logprob delta，结论是被 `vllm-ascend` 自定义线性层 API 配套缺失阻塞，而非精度失败；两个任务已停止，4–7 卡已释放，源码未改动。

### 08-26

- DSA-CP A3 修复已完成并在 81.162 卡 6–9 验证：根因是 A3
  错误进入仅适用于 A5 的 full-weight `o_proj` 路径，量化 kernel 在临时
  gathered weight 布局下失败/产生异常输出；现在 A3 保留原 TP/all-to-all
  路径，A5 继续使用优化路径。
- 修正 DSA-CP 的完整 hidden-state/Q token shard/KV cache 布局、HC 多维
  状态按 token 维分片、metadata shared 参数和 ACL graph hook；A3 legacy
  DSA-CP 的 graph replay 曾在首个请求后卡住 HCCL，现明确降级 eager，避免
  默认 graph 配置死锁，A5 graph 支持不变。
- 新增 4096-token 长 prefill 的 DSA-CP 开关 E2E guard，并补充 HC 分片和
  A3 graph 支持单测。81.162 远端定向 UT 最终 `48 passed`，实际 E2E guard
  `2 passed`；同一
  DSpark-w4a8 权重下 CP-off 长序列输入吞吐约 `6124/6140/6138 tok/s`，
  CP-on eager 约 `7046/6922/6919 tok/s`，平均提升约 `13.5%`；默认 graph
  配置经 A3 eager fallback 也完成 3 次约 `7073/7072/7050 tok/s`。
- 两种开关均完成 4096-token prompt + 1 token decode，prompt token 保留
  正确且无此前报错。正式 semantic golden 尚未在目标
  `gdydems/DeepSeek-V4-Flash-w4a8-mtp` 权重上复核：81.162 可用的是本地
  DSpark 量化权重，输出不能作为该 golden 的等价替代；共享目录另一份
  39-shard 权重在装载到 15/39 shard 时 worker 退出，官方 raw FP8 权重则
  在 A3 触发 `customize_dtype is not supported by the current soc version`。
- DSA-CP 在 ModelRunner V2 的首个真实请求此前分别卡在 metadata 缺少
  `common_ratio_to_sas_metadata`、ACL graph 缺少 `update_graph_params`，已
  补齐对应构建参数和空 graph hook。
- 远端 81.162（卡 6–9）形状探针定位到非规整 token 数下
  `sequence_parallel_chunk` 对三维 `(tokens, hc_mult, hidden)` 使用二维
  padding，导致 `(5, 4, 4096)` 变成 `(1, 7, 4096)`、触发 `npu_hc_pre_v2`
  的 HC 维度错误；模型侧改为先展平 per-token 维度、只沿 token 维分片再
  恢复形状。
- 完整状态布局下 CP-on eager 已能完成请求，但与 CP-off token 仍不一致；
  临时加入可按层选择的数值探针，继续定位 attention 前后状态差异。
- 完整 `tests/ut` 收集被容器缺少 `examples.disaggregated_prefill_v1` 和
  `pytest_mock` 阻断；排除这两个环境错误后，回归在既有 A2 attention
  precision 路径出现失败并长时间无进展，已停止，未将其归因于本次 DSA-CP
  修改。

### 08-25 13:00

实验记录：MTP=2 + flashcomm1 报错是否为 92616f3（删 FlashComm v1）引入的回归验证。
81.162 卡0,1，vllm-ascend 切到 92616f3^（92ea871b2），工作区 vllm（含检查），Qwen3.5-35B-A3B TP2/EP、`enable_flashcomm1=true`、qwen3_5_mtp `num_speculative_tokens=2`、FULL_DECODE_ONLY；引擎初始化即抛 `Can't determine cudagraph shapes ... multiple of 3 ... and 2 ...`（job j-20260825-114629-01）。
结论：父提交同样报错，非 92616f3 回归；根因是 `update_pass_config` 在 flashcomm1 开启时把 `pass_config.enable_sp` 临时置 True，上游检查自 2025-11（64e39d667c/8f4f77a727）起就拦 MTP=2+TP2 组合，此前 CI 只跑过 MTP3（query len=4 可整除）所以从未暴露。现场已恢复（本地/远程均回 657df3efa）。

### 08-21

- 回退 DeepSeek V4 DSA-CP 的外部分片和输出 all-gather：模型传入完整 hidden states，DSA-CP 内部切 Q shard，o_proj 后直接保留 local output。
- 修改 `vllm_ascend/attention/context_parallel/dsa_cp.py`；按用户要求未运行测试，仅完成 `git diff --check`。

### 08-18 02:00

- 按 `docs/plans/pr-multicard-a3-e2e-risk-tests.md` 在 17.111 完成 PR #13946（`lijiaqi/delete-flashcomm`，本地 HEAD `c6a0a28e2`，与 CI 绿的头 `41461cf4a` 零 diff）E2E 风险测试。P0：阶段一 `test_deepseek_v4.py`+`test_sequence_parallel_linear.py` 5 passed（含 golden token-id 与 SP precision，job j-20260817-205751-01）；阶段二 `context_parallel/test_accuracy.py` 2 passed（DSA-CP/SFA-DCP golden，j-20260817-212305-01）；DSpark 两组因指定模型 `UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test`（w4a8+n_predict=1）本地/网络均不可得，按模型缺失例外记录（CI 同代码全绿替代覆盖）。P1 已过 6 项：`test_qwen3_30b_a3b` eplb、`test_qwen3_moe_eplb` w8a8、`test_qwen3_5` mtp3、`test_deepseek_v3_2_w8a8_pruning` 非 PD 用例、`test_shared_expert_dp`（DSV2_LITE_MODEL=/mnt/share/weights/DeepSeek-V2-Lite-Chat）。`test_graph_mode.py` 两个 ACLGRAPH case 缺 `vllm-ascend/DeepSeek-V2-Lite-W8A8` 模型未跑。
- 8 个测试文件的模型路径改为 env 读取（默认值保持上游 ID，符合 AGENTS.md 约定），未 commit。
- 关键排障：remote CLI 的 `--cards` 只做占用登记，**不注入 `ASCEND_RT_VISIBLE_DEVICES`**，漏传会导致多个 job 挤到 chip 0-3；`RemotePDServer`（`tests/e2e/conftest.py:549-562`）无视外层 env，永远从 device 0 起分配（PD 测试固定占 chip 0-3）；多 vLLM 实例并存需 `HCCL_HOST_SOCKET_PORT_RANGE`/`HCCL_NPU_SOCKET_PORT_RANGE` 显式端口段，否则 `hcclCommInitRootInfoConfig error code 7`。
- 环境事故与恢复：17.111 容器 PID 1 是 `tail -f` 不回收僵尸，加上误判为驱动泄漏后执行了 `remote down/up`，workspace 为容器本地存储被清空；已通过 `remote sync` 恢复代码、`COMPILE_CUSTOM_KERNELS=1` 重编译（约 25min）、`COMPILE_CUSTOM_KERNELS=0` 恢复 editable 安装，并补装 pytest-asyncio/modelscope。 workspace 备份在 NFS `/mnt/share/vaws-backup-20260818.tar.gz`（403MB，gzip 校验通过）。
- 真实根因（订正）：显存"泄漏"实为**另一租户的 DP16 vLLM 任务**（`VLLM::Worker_DP` 每 chip 约 50GB，npu-smi 进程列表可见），从 23:20 左右起占满全部 16 个逻辑卡；该占用导致 PD disagg 与 mrv2_eplb 两个 P1 用例无法获得显存。容器重建与代码环境已就绪，待机器空出后即可重跑这两个用例。

### 08-13 17:10

- 完成 model&setup 4（Kimi-K2.5 w4a8，DP2/TP8/EP）：17.110 仅 8 物理卡（512GB）放不下，改在 17.111（8 NPU × 2 chip = 16 逻辑卡，每逻辑卡 64GB）执行。对话正常（Paris / `12+30=42`，无乱码）；benchmark 50/50 成功，output 466.39 tok/s、TTFT mean 4155.66ms、TPOT mean 52.91ms，结果已回填 `benchmark-delete-flashcomm.md`，JSON 为 `.vaws-local/benchmark/80.5.17.111/runs/2026-08-13T09-07-05Z_80.5.17.111_31992_a0938071.json`。
- 17.111 环境修复：容器加代理（`80.253.137.110:7897`，`no_proxy` 含本机 IP）和 ATB 库路径；parity 安装被 `triton-ascend==3.2.2` pin 阻断（源上最高 3.2.0，容器内 3.2.1），改为 parity materialize + 手工 editable 安装 + 从旧树复制 `vllm_ascend_C`/`libvllm_ascend_kernels.so`/`_cann_ops_custom` 到 `/vllm-workspace`；parity 快照无 git tag 导致 `vllm.__version__` 失真，服务需显式 `VLLM_VERSION=0.26.0`。
- skill 修复：serving `select_devices` 支持 2 chip/物理卡的逻辑 id 校验（logical=physical*2+chip）；benchmark 新增 `--health-timeout` 透传、bench 客户端输出强制 UTF-8（修 Windows GBK reader 崩溃）并落盘 `.vaws-local/benchmark/<machine>/client_output_*.log`。
- 排障记录：两轮 bench 50/50 全挂的根因是容器代理经 `trust_env=True` 泄漏进 `vllm bench serve` 客户端，请求被错误代理；已把本机 IP 加入 `no_proxy` 后恢复。

### 08-13 10:38

- 修正文档 `Qwen3-235B-A22B.md` 中被限流的 `docs.vllm.com.cn` 链接，统一改为可用的官方 vLLM CLI 参数页。
- `ruff-check`、`ruff-format`、目标 Markdownlint 和 `git diff --check` 均通过。

### 08-13 10:24

- 修复 PR #13946 的 `dsa_v1.py` Ruff F841：删除 DSA v1 统一 forward 中已无调用方的五个临时变量；保留 `actual_tokens`。
- 对齐 PR head 的 main 合并内容，并完成全仓 `ruff-check`、`ruff-format`、`markdownlint` 与 `git diff --check` 验证。

### 08-12 14:06

- 扩展 `.agents/scripts/simple_code_sync.py`：源路径支持文件和目录，目录自动使用 `scp -r` 递归传输。
- 已将 `.temp/simple_code_sync_smoke_dir` 传输到 17.111，并在远端执行 `run.sh`；输出与本地文件 SHA256 校验均通过。

### 08-11 21:25

- 在空闲的 17.111（`80.5.17.111`，A3、CANN 9.0.1）完成当前工作区的 vllm-ascend A3 编译和 editable 安装：211 个 custom-op 构建目标完成，`cann-ops-transformer-custom_linux-aarch64.run` 打包成功，`vllm_ascend_C` 构建成功；导入校验、A3 设备类型校验和 ATB 注册均通过。构建日志位于远端 `.compile-log/a3-ninja-package-final-20260811.log`、`.compile-log/vllm-ascend-install-a3-final-20260811.log`。
- 完成目标二第一项：`/mnt/share/weights/Qwen3-30B-A3B-W8A8`，DP1/TP2/EP、设备 `0,1`，服务健康检查和模型列表检查通过；OpenAI chat 请求返回 HTTP 200，关闭思考后回答 `Paris`。服务日志为 `/vllm-workspace/.vaws-runtime/serving/20260811_131302/stdout.log`，响应保存于远端 `.goal2/qwen3-30b-chat-final.json`。
- 完成目标二第二项：原 `/mnt/share/weights/Qwen3-VL-2B-Instruct` 只有 `.git` 且缺少 `config.json`，因此按 TODO 的替代条件使用完整的 `/mnt/share/weights/Qwen3-VL-4B-Instruct`，DP1/TP2、设备 `0,1`；服务健康检查通过，chat 请求返回 HTTP 200，`12+30` 返回 `42`。服务日志为 `/vllm-workspace/.vaws-runtime/serving/20260811_131724/stdout.log`，响应保存于远端 `.goal2/qwen3-vl-4b-chat.json`。
- 两个服务均已停止，目标卡 `0,1` 未保留本次服务进程。收尾时发现另有既有 Qwen3.5-35B-A3B 服务 PID `646303` 使用设备 `4,5,6,7`、端口 `38081`，未擅自终止。编译期间为绕过 CANN 9.0.1 内置 op-info 缺少 `limited` 的不兼容基础算子，远端 `csrc/build_aclnn.sh` 曾做临时构建调整；验证结束后已恢复原文件（SHA256 `f80f19bd80cdc9966663b4f1b2cf12f07e8d5017266f86cfbe9b32de5a264289`）。

### 08-10 18:45

- 在 90.90.97.4 最后四张逻辑卡 `12,13,14,15` 上完成 main 分支离线 DP E2E：Qwen3-30B-A3B、vLLM `35efdf6b3` + `vllm-ascend-main` `9f3aa1e7`、DP2/TP2/EP、FlashComm=1、CANN 9.1、eager。作业 `job-20260810T104208Z-d1da19d8` 状态 `succeeded`、exit code 0；DP rank 0/1 均完成 200 条 prompt 并输出生成文本。为适配容器没有 ModelScope，将官方示例通过未跟踪临时 launcher 指向本地模型，并将采样改为 greedy；这是测试启动适配，不是仓库源码改动。
- E2E 日志没有上一轮随机采样触发的 `hfusion.cumsum`/`EngineDeadError`；结束后 `npu-smi` 显示 NPU 6/7（逻辑卡12–15）均为 `No running processes found`。

### 08-10 18:30

- 按当前资源窗口使用 90.90.97.4 最后四张逻辑卡 `12,13,14,15`，以 vLLM `35efdf6b3` + `vllm-ascend-main` `9f3aa1e7`、CANN 9.1、Qwen3-30B-A3B `/mnt/weights/Qwen3-30B-A3B` 完成 DP2/TP2/EP、FlashComm=1 的 4K/2K、并发1 正式在线 benchmark。服务使用 `FULL_DECODE_ONLY` 图模式，2 次请求中 1 次 warmup；计入统计的第2次为 `5.8678 output tok/s`、TTFT `399.7 ms`、TPOT `170.3 ms`、E2E `349020.4 ms`，warmup 为 `5.7055 output tok/s`。完整结果为 `.vaws-local/benchmark/90.90.97.4/runs/2026-08-10T10-19-42Z_90.90.97.4_38940_d0cfd00d.json`。
- 同一 main + FlashComm=1 四卡服务通过正常 OpenAI Chat 请求，固定回复为 `FLASHCOMM_OK`，system fingerprint 为 `vllm-0.1.dev1+g9374f773a-tp2-dp2-ep-6887a6a9`。
- 使用 `lm_eval==0.4.12`、本地 GSM8K JSONL task `goal2_gsm8k` 和 Qwen3-30B-A3B tokenizer 完成一次文本评测执行 smoke（`limit=1`）：strict/flexible exact-match 均为 `0`；结果和样本已保存到 `.temp/goal2-lm-eval-qwen3-main-chat/`。该结果只证明评测链路可执行，不代表完整数据集准确率。
- benchmark、正常对话和 lm-eval 完成后已停止服务；`npu-smi` 确认 NPU 6/7（逻辑卡12–15）无本次残留进程。

### 08-10 15:45

- Goal2 main + FlashComm online path is now valid for Qwen3-30B-A3B on CANN 9.1: matched vLLM `35efdf6b3` + vllm-ascend-main `9f3aa1e7`, DP2/TP2/EP, devices `12,13,14,15`, eager mode, 4K input/2K output, concurrency 1. `vllm bench serve` completed `1/1` request with `failed=0`, `377.366s`, and `5.4271 output tok/s`; result JSON is under `.vaws-local/benchmark/90.90.97.4/runs/`.
- After setting `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2]}` to satisfy the TP2 capture-size constraint, the same Qwen3 case also passed full-graph startup and completed `1/1` request with `failed=0`, `368.484s`, `5.5579 output tok/s`, `TTFT=437.6ms`, `TPOT=179.8ms`; result JSON is `.vaws-local/benchmark/90.90.97.4/runs/2026-08-10T08-10-14Z_90.90.97.4_35704_79a07f02.json`.
- A matched full-graph FlashComm=0 baseline also completed `1/1` request with `failed=0`, `344.454s`, `5.9456 output tok/s`, `TTFT=1285.0ms`, `TPOT=167.6ms`; result JSON is `.vaws-local/benchmark/90.90.97.4/runs/2026-08-10T08-21-54Z_90.90.97.4_18340_7cb1ab78.json`. On this single-request sample, FlashComm=1 is `-6.52%` output tok/s versus off; the sample is not a stable multi-run estimate and should be rerun with warmups before using it as a performance conclusion.
- The same service passed a normal OpenAI chat request with `chat_template_kwargs.enable_thinking=false` and returned `FLASHCOMM_OK`; system fingerprint records `tp2-dp2-ep`.
- `lm_eval` is installed in the container (`0.4.12`). A local JSONL GSM8K task ran through `local-completions` with the remote Qwen3 tokenizer, `limit=1`, and saved result/sample artifacts under `.temp/goal2-lm-eval-qwen3-tokenizer/`; this is an execution smoke result, not a full accuracy score.
- The exact DP2/TP2 normal-conversation e2e test remains skipped with `reason="broken, fix me"`; this is an existing test skip, not passing functional evidence.
- The exact `Qwen/Qwen3.5-35B-A3B` target is being downloaded locally through the ModelScope skill manager into `D:\temp\goal2-qwen35-exact` (official target is 66.99 GiB; current proxy speed is approximately 0.2 MiB/s). The remote container cannot reach ModelScope directly, so no remote weight replacement has been attempted. The existing remote directory is incomplete and uses a different 15-shard layout.
- The available Qwen3-VL-2B fallback was attempted for multimodal startup on devices `10,11`: with batch-invariant disabled it reached model profiling but failed because CANN 9.1 lacks `aclnnAddRmsNormBias`; with batch-invariant enabled it instead hit the known 3-D input assertion in `linear_batch_invariant`. This is fallback-model evidence only, not the absent Qwen3-VL-30B target.
- The AGENT_TODO DeepSeek target is BF16, and the complete remote weights are available at `/mnt/weights/dsv4_bf16/DeepSeek-V4-Flash-bf16`; its DP2/TP4/EP test still needs eight logical cards, so it was not substituted into the four-card run. The requested Qwen3-VL-30B-A3B-Instruct directory is still absent remotely; only the Qwen3-VL-2B fallback is present and was not used as target evidence.
- Qwen3 full-graph startup is now verified in the 4K/2K benchmark with `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2]}`; `--enforce-eager` was used only for the separate normal-chat smoke.

### 08-07 20:00

- 完成目标2的 SP 消融：容器 `90.90.97.4`、CANN 9.1、Qwen3-30B-A3B（`/mnt/weights/Qwen3-30B-A3B`），vLLM 子模块为 `v0.26.0`，DP2/TP2/EP，输入 16384，50 个有效请求。SP off 为 `0.8634 requests/s`、`14147.23 tokens/s`；SP on 为 `1.0755 requests/s`、`17621.86 tokens/s`，吞吐提升约 `24.56%`。
- 完成目标2的 FlashComm PP2 关闭组：DP2/TP2/PP2/EP，`VLLM_BATCH_INVARIANT=1`，关闭旧 norm/rope/muls fusion 以适配 CANN 9.1，输入 16384、50 个有效请求；结果 `60.6846 s`、`0.8239 requests/s`、`13500.12 tokens/s`。
- FlashComm 开启组确认读取 `enable_flashcomm1=True`，但在旧 `vllm-ascend` main 基线（`9f3aa1e7`）的 RoPE warmup 中触发 `positions.shape[0] != num_tokens` 断言，未生成吞吐 JSON。该问题发生在 warmup、不是性能结果；因此未用运行时改算子逻辑来伪造开启组数据，目标2的 FlashComm on/off 数值对比仍被环境/版本兼容性阻塞。

### 07-28 02:45

- 完善 `.agents/skills/handoff_context/SKILL.md`：补充 frontmatter、触发场景（了解=读取 CONTEXT.md，理解=读源码构建/增量更新）、工作流程与硬约束（≤800 行、可点击链接、事实标注）。
- 未改动 CONTEXT_template.md。

### 07-23 02:10

- 在 `.agents/skills/concise-code-explanation/` 新增简洁代码解释 skill：将“解释代码/为什么会发生”类回答限制为 2-3 段，并强制覆盖 Why、Input、Logic、Output、Flow，附带数据流架构图和实际源码路径/行号定位。
- 通过 `skill-creator` 的 `quick_validate.py`，并完成 skill 内容字段与元数据检查。

### 07-23 02:14

- 按要求将 `concise-code-explanation` 的数据流图格式改为仅使用 ASCII，禁止 Mermaid、PlantUML 和其他图形 DSL。
- 重新通过 `quick_validate.py`、ASCII-only 内容检查和目标文件 diff 检查。

### 07-21 14:45

- 完成 SP、FlashComm1、GraphFusionPassManager 和 Ascend ACL graph capture/replay 的代码调研。
- 在 CPU 容器执行 `tests/ut/compilation/test_flashcomm.py`，结果为 `5 passed`。
- 在 NPU 容器运行 Qwen3-30B-A3B TP2/EP FlashComm E2E，观察到 PIECEWISE bucket `num_tokens=2` 的 ACL graph capture 和 replay，首次运行 `1 passed`。
- 增加 FlashCommPass DEBUG 级别的 before/after FX graph 打印；真实图显示原 pattern `unquantized_gemm(x, weight, None)` 未命中实际的两参数 Dynamo 节点。
- 将 FlashComm column/row pattern 和单测无 bias 图修正为两参数形态；修正后的 NPU E2E 因 Docker socket permission denied 尚未重跑。
- 新增报告：`docs/flashcomm_sp_cuda_graph_report.md`。

### 07-21 16:25

- 重新完成 SP NPU 验证：Qwen3-30B-A3B TP2/EP 在 `VLLM_COMPILE + FULL_DECODE_ONLY` 下通过，结果 `1 passed`；两张卡的 `SequenceParallelismPass` 均替换 48 个 pattern。
- 修复 SP 真实图与模板不一致：补齐 `maybe_chunk_residual`、适配 Dynamo 省略的 RMSNorm 默认参数，并增强 before/after FX graph 与替换计数日志。
- 修复 SP 开启后的 shape/tiling 错误：同步 `maybe_pad_and_reduce` fake shape 与真实 TP/EP 分支；local residual 在 full-sequence 输入前自动 all-gather。
- 验证日志：`.log/sp_e2e_real10.log`；单测 `tests/ut/compilation/test_flashcomm.py` 为 `5 passed`。

### 07-22 06:57

- 删除 `linear.py` 中 FlashComm1 的 eager 通信 fallback；保留普通 `super().forward()` 作为未启用编译 pass 时的原始线性层路径。
- 将 FlashCommPass 的列并行 replacement 改为显式 `all_gather -> unquantized_gemm`，行并行 replacement 改为显式 `npu_mm_reduce_scatter_base`，移除 row fallback pattern。
- 增加 row replacement 的 FX 算子断言，并补充日志中的 `mm_reduce_scatter` 计数。
- row replacement 统一使用 TP group 的 `world_size`，避免 pattern replacement 阶段重新读取未初始化的全局并行状态。
- Docker 复跑当前受 `/var/run/docker.sock` 权限阻断；最近一次容器 E2E 已进入模型初始化，但被容器 CANN 缺少 `aclnnAddRmsNormBias` 阻断，尚未进入 FlashCommPass。

### 07-22 16:20

- 在 `xrs_vllm_main` 中加载目标 worktree 源码，并复用容器已构建的 `vllm_ascend_C` 与 custom-transformer OPP。
- Qwen3-30B-A3B TP2/EP FlashComm1 FX 编译 E2E 通过：`1 passed, 1 warning`，日志为 `.log/flashcomm_debug16.log`。
- DEBUG 强制编译日志 `.log/flashcomm_debug18.log` 确认 `FlashCommPass after apply replacement`，并多次记录 `replaced 2 patterns (all_gather=1, mm_reduce_scatter=1, reduce_scatter=0, all_reduce=0)`。
- 当前 E2E 使用单 token 实际生成，避免容器 CANN TND attention 在 TP 长 prefill 上的 `queryT`/`actualSequenceLengthQ` 分片约束；编译 pass 仍覆盖 `(1, 8192)` compile range。
- 目标 FlashComm 单测与线性层回归测试通过：`19 passed, 14 warnings`。

### 07-22 19:45

- 按 `EXPS.md` 重新启动 FlashComm TP4 吞吐实验；修正实验配置为运行时 `enable_flashcomm1=true`、编译期 `pass_config.enable_sp=false`，并补充 MoE 必需的 `--enable-expert-parallel`。
- `.log/exp_flashcomm_20260722_v3.log` 已确认 `FlashCommPass after apply replacement` 和 `npu_mm_reduce_scatter_base` 目标进入真实 FX 图；最终因容器 CANN 缺少 `aclnnAddRmsNormBias`，未生成 JSON。
- 使用干净 `main` 临时 worktree `37f44d577` 运行 SP 关闭基线；同样在模型初始化阶段被 `aclnnAddRmsNormBias` 缺失阻断，未生成 JSON。
- 修复两个 worktree 的 Qwen3 MoE E2E 用例模型变量：新增 `SP_TEST_MODEL` 环境覆盖，避免当前分支测试运行时 `NameError` 和固定本地权重路径。
- Docker 内相关 UT：`fix/sp-by-pass` compilation/linear 为 `39 passed`，`flashcomm-by-pass` 为 `45 passed`；新增 E2E collection 两个 worktree 均通过。
- 全量 UT 首次收集缺少已声明依赖 `pytest-mock`，补齐后重新执行；约 5% 用例通过后连续 7 分钟无日志增长，已中止，日志为 `.log/ut_full_fix_sp_20260722_retry.log`。

### 07-23 10:08

- 统一 `bench_sp_tpot.sh` 和 `bench_flashcomm_tpot.sh` 的 Ascend 运行环境：显式加入 CANN、Ascend Toolkit、`ASCEND_CUSTOM_OPP_PATH`、custom-transformer `libcust_opapi.so`、`vllm_ascend_C` 和 `LD_LIBRARY_PATH`。
- 新增 `scripts/ascend_bench_bootstrap/sitecustomize.py`，使 FlashComm 源码 worktree 可以加载已编译的 Ascend 原生扩展；两个实验改为独立日志和 JSON 输出。
- 重新验证 SP 基线：100/100 请求完成，无 `aclnnAddRmsNormBias` 错误，吞吐 `0.60 requests/s`、`9827.21 tokens/s`；完整结果为 `.log/bench_sp_true_16384.log` 和 `.log/bench_sp_true_16384.json`。
- 首次按统一库路径执行 FlashComm TP4/EP 完整实验：`FlashCommPass` 已替换 96 个 pattern，确认自定义库问题已排除；但默认 `max_num_batched_tokens=8192` 的 profile 在 `copy_between_host_and_device_opapi` 触发 507035/MTE 越界，未生成 JSON。将 FlashComm 实验的最大 batch token 调整为 1024 后重跑。

### 07-23 10:52

- FlashComm TP4/EP worktree 在空闲四卡 `0,1,4,5` 上重跑：on 仍在 `profile_run -> _dummy_sampler_run` 触发 `copy_between_host_and_device_opapi` 的 DDR/MTE 越界 `507035`；四个 rank 均确认 `FlashCommPass replaced 96 patterns`，未生成 JSON。日志为 `.log/exp_flashcomm_on_1024.log`。
- FlashComm off 控制完整完成 `100/100`，JSON 为 `.log/exp_flashcomm_off_1024.json`；吞吐 `5.33 requests/s, 5474.23 total tokens/s, 21.30 output tokens/s`。
- main `37f44d577` 的 SP off 完整完成 `100/100`，JSON 为 `.log/exp_sp_main_off_16384.json`；吞吐 `1.47 requests/s, 24170.38 total tokens/s, 5.90 output tokens/s`。main SP on 在 `SequenceParallelismPass` applicable 的 `(1024,8192)` 图中复现 `shape [2048,8,128] is invalid for input of size 8388608`，日志为 `.log/exp_sp_main_on_16384.log`，未生成 JSON。
- main SP on 失败与已跑通的 `fix/sp-by-pass` 基线差异对应：main 未包含 `f3562272f fix: align sequence parallel graph shapes`，其 SP reduce/all-reduce pattern 未对 local residual 执行 `maybe_chunk_residual`；现有 `fix/sp-by-pass` SP on 结果仍为 `.log/bench_sp_true_16384.json`。
- 两个 benchmark 脚本新增 `ENABLE_SP`/`ENABLE_FLASHCOMM`、独立 cache 和 `VISIBLE_DEVICES` 参数；脚本语法、bootstrap Python 编译检查通过。

### 07-23 20:00

- 修复 `SequenceParallelismPass` 漏匹配 MoE 输出链路的问题：实际 FX 图为 `maybe_all_reduce -> aten.alias -> maybe_chunk_residual -> AddRMSNormBias`，新增 alias-preserving pattern，并覆盖 middle、last、Qwen3-VL 三种 RMSNorm 形态。
- Docker TP2/EP2 Qwen3-30B-A3B E2E 通过：`SequenceParallelismPass Replaced 96 patterns`，after-graph 显示 `reduce_scatter -> maybe_chunk_residual -> AddRMSNormBias -> all_gather`，功能结果 `1 passed`；日志为 `.log/e2e_sp_maybe_ar_local5.log`。
- 更新 `docs/sp_debug.md`，精确说明 485 次 all-gather 的来源、`maybe_chunk_residual` 隐式 gather 调用位置及 profiling/after-graph 判定方法。
- 补充旧日志判读说明：`before apply replacement graph()` 和 `Pattern N:` 中仍会看到 `maybe_all_reduce -> alias -> maybe_chunk_residual`；验收必须检查 `Replaced 96 patterns` 之后的 after-graph。
- 验证补充：聚焦 UT `8 passed`，功能 E2E `1 passed`；全量 `tests/ut` 已在 Docker 中启动，但在既有 `attention/a2/test_attention_v1_precision.py` 精度用例阶段超时，未将该环境级阻塞归因于本次 SP 修改。
### 07-28 20:20

- 定位 DP2/TP2/EP4 + SP 乱码：SP+EP 的 MoE finalize 已返回 token-sharded 结果，但 Ascend runner 仍无条件执行 TP all-reduce，随后 Qwen3 MoE 再 TP all-gather，导致不同 token 段被相加。
- 修复 `AscendMoERunner._maybe_reduce_final_output`：`moe_config.is_sequence_parallel` 时跳过最终 TP all-reduce；同时补齐 DP/TP 变长分片在 EP all-gather 后 unpad、EP reduce-scatter 前 pad 的处理。
- Docker focused UT `27 passed`；使用真实 `bash scripts/serve_sp.sh`、DP2/TP2/EP 开启/SP 开启服务验证，短请求 19 tokens 和长请求 1420 tokens 均恢复连贯输出；SP off 对照也通过。
- 详细定位、profiling、after-graph 和 E2E 验证方法写入 `docs/sp_debug.md`。

### 07-28

- 新增 `docs/container_compile_flow.md`，记录换仓后在 `xrs_vllm_main` 中对齐 catlass、重建 CANN custom-op/C++ 扩展、验证实际加载路径、运行 `bench_sp_tpot.sh` 和执行两卡 SP E2E smoke 的完整流程与失败边界。
- 为 `SequenceParallelismPass` 增加 SP 替换/TP 回退 warning：可执行且命中 pattern 时报告替换数量，未命中或 compile range 低于阈值时报告回退 TP。
- 新增 `tests/ut/compilation/test_sequence_parallelism.py`，覆盖命中替换、零命中回退和 token 范围不满足时回退三种状态；Docker 定向测试结果为 `3 passed`。
- 增加两卡 Qwen3-VL SP fallback 功能 E2E，并通过本地权重验证 TP fallback 输出与无 SP 基线一致；日志同时观察到 `(1, 2047)` 回退 TP 和 `(2048, 8192)` 替换 53 个 pattern。结果为 `1 passed`，日志为 `.log/e2e_sp_warning_fallback_0728.log`。
- Docker 运行 `tests/ut/compilation`，结果为 `29 passed, 14 warnings`；新增源码和测试通过 `py_compile`。

### 07-29 04:04

新增 `docs/sp_moe_pass.html`，用 before/after 流程图讲解 SP pass 与 MoE pass 的 FX 图替换。
按三种 fused MoE 外部模式分节：SP+EP MoE（DP2/TP2/EP4）、TP 模式 EP on（DP1/TP2/EP2）、TP 模式 EP off；标注具体 shape 与关键算子，省略 getitem/view。
内联 JS 已通过 node 语法与运行校验。随后追加幻灯片播放模式（方向键/空格翻页、进度条、#页码 定位、文档模式切换）；后按更新后的 sp_moe.md 修正情况 1 before 图（o_proj 后 all_reduce、MoE 前 sequence_parallel_chunk），重排为 5 页：设置→模式映射、三个模式各一页（图+要点）、总结。

### 08-10

- 新增 `scripts/offline_profile.py`：固定 Qwen3-30B-A3B、DP2/TP2、随机离线负载和 Torch NPU profiler，运行后自动执行 `analyse()` 并校验 4 个 rank 的 `kernel_details.csv` 与 `trace_view.json`；输出目录统一追加 `D-H-M` 时间后缀，本机仅完成静态校验，未启动 NPU 推理。

- 测试一当前源码快照已 materialize 到 `90.90.97.4`；通过 `HCCL_NPU_SOCKET_PORT_RANGE=auto` 与 `HCCL_HOST_SOCKET_PORT_RANGE=auto` 完成 Qwen3-4B DP1/TP2/SP 4K/2K 基础烟测，输出吞吐 `14.35 tok/s`、TTFT `7483 ms`、TPOT `66.05 ms`，证据为 `.log/test1_infra_qwen3_4b_bench_sp_dp1_hccl_clean.log`。
- 正式目标仍未产出：Qwen3-30B 路径为未量化 BF16 约 57G，Qwen3.5-35B 路径仅有 15 个分片中的 1 个约 446M，DeepSeek 可用路径为 `w8a8` 而 TODO 要求 `w4a8`。
- 目标 DP2/TP2/EP 与 DP2/TP4/EP 分别需要 4/8 个逻辑 NPU；尝试的 DP1/TP2/EP+SP 降配启动也因既有任务并发占用、worker 初始化失败而无结果，未停止他人进程，正式 `vllm bench` 与 multimodal/text `lm-eval` 待资源和权重补齐，日志为 `.log/test1_qwen3_30b_dp1_tp2_smoke.log`。
- 远端已安装并验证 `lm-eval==0.4.12`；烟测结果 JSON 为 `.vaws-local/benchmark/90.90.97.4/runs/2026-08-10T06-29-13Z_90.90.97.4_41656_b134186e.json`。
- 测试3准备阶段已确认 `vllm-ascend-main@9f3aa1e70` 的远端源码 overlay 可导入，`VLLM_ASCEND_ENABLE_FLASHCOMM1=0` 生效；目标权重在 90.90.97.4 可见，正式测试尚未开始。
- 90.90.97.4 的正确依赖容器持续被其他服务占用，观测到 0–4 的 OmniDiff/TP worker 以及 6–7 的 Python worker，无法满足 Qwen 的 4 卡或 DeepSeek 的 8 卡配置；未停止他人任务。
- 为寻找空闲资源保守修复了 80.5.17.110 和 90.90.97.44 的已登记退出容器。80.5.17.110 的 main overlay 启动先后受 vLLM API 不匹配、旧 Transformers 缺少 `HunYuanVLProcessor` 阻断；受控 parity 安装又因镜像缺少 `setuptools_rust` 在 editable metadata 阶段失败，未使用裸 pip。90.90.97.44 的 Transformers 为 5.5.4，同样不满足 main 的 5.14.1 要求。因此测试3仍待正确依赖与足够 NPU 资源。

### 08-11

- 修复 Windows remote-code-parity：增加 SSH 连接保活和有界超时，流式 SSH 超时主动终止；统一本地/远端输出为 UTF-8。
- 修复 `gc_runtime_cache.py` 使用 Windows `Path` 拼接容器 POSIX 路径的问题；更新 parity 行为文档。
- 验证 parity 脚本编译、SSH 超时 smoke、POSIX 路径 smoke 和 snapshot plan 均通过。
### 08-12

- Restored the deleted vllm-ascend runtime on 80.5.17.111: rebuilt the Ascend custom-op package and verified the generated package plus `vllm_ascend_C` import with the NPU platform.
- Fixed the uneven DP/SP EP reduce-scatter contract by cropping each rank's padded result to its local token count; updated fake-shape coverage and Qwen3.5 regression assertions.
- Remote current-source verification: `tests/ut/ops` + `tests/ut/compilation` passed with `233 passed, 16 skipped`; changed files compile and `git diff --check` passes. Full UT is still blocked by an existing vLLM API mismatch in `test_npu_ipc_engine.py` (`IPCTrainerInitInfo` is absent from the current vLLM submodule), while a subsequent run reached 167 passed before an attention test-order failure.
- Qwen3.5 live output, target profiling, and DeepSeek-V4-Flash-w4a8 benchmark remain pending: `npu-smi` still reports 16 cross-container `VLLM::Worker_DP` processes using all eight NPUs, and they are not registered in this workspace's serving state.
### 08-12 03:15

- Rebuilt the Ascend custom-op package on `80.5.17.111` with the validated serial `-j1` flow after the automatic editable build failed inside `build_aclnn.sh`. The package completed successfully and was installed into `vllm_ascend/_cann_ops_custom`.
- Verified the rebuilt package (`cann-ops-transformer-custom_linux-aarch64.run`, SHA256 `392cab2c...3b6a3b8`) and `libcust_opapi.so` (SHA256 `9d833a4f...7268a06a`). Re-registered `vllm-ascend` as an editable install with `COMPILE_CUSTOM_KERNELS=0` to preserve the verified compiled extension.
- Runtime import now resolves `vllm` and `vllm_ascend` from `/vllm-workspace`, imports `vllm_ascend_C`, and reports platform `npu`. The `vllm-ascend` directory was present both locally and remotely; it was not missing during this continuation.
- Parity 后复核远端当前快照：定向回归单测 `test_register_custom_ops.py`、`test_fused_moe.py`、`test_linear.py`、`test_moe_mlp.py` 通过（`70 passed`），Qwen3.5 四卡 E2E 文件收集到 2 个测试用例；编译包 SHA256 `392cab2c...3b6a3b8`、`libcust_opapi.so` SHA256 `9d833a4f...7268a06a` 保持不变。
- 目标一的真实 Qwen3.5 DP2/TP2/EP、profiling 和 DeepSeek-V4-Flash-w4a8 benchmark 尚未执行：17.111 当前仅卡 `5,6,7` 空闲，卡 `0,1,2,3,4` 各占用约 61.8GB HBM，探针判定为其他容器；没有可安全停止的本工作区进程。

### 08-12

- 注册 `141.61.52.183`：A5/`ascend950dt`，复用 `xrs_vllm_main`，容器 SSH 端口 `46000`，镜像为显式 A5 镜像；已写入 `.vaws-local/machine-inventory.json`。
- 完成主机密钥登录、容器 SSH、A5 运行库补齐和 `torch_npu` 导入验证；NPU tensor smoke 因其他容器 `k3_wyx_0807` 占用 8 张卡未通过，未停止其他服务，机器保持待修复状态。

### 08-12

- 更新 `.agents/skills/grill/SKILL.md`：将逐题提问改为一次列出最多 10 个高影响问题，再集中讨论；补充环境事实查询、批量追问和行动确认规则。
- Skill 元数据校验与 `git diff --check` 通过。

### 08-12 20:48

- 在 `vllm-ascend` 执行 `ruff format`、`ruff check` 和 `markdownlint`，并保留自动格式化结果。
- 修复 `platform.py` worker 选择逻辑的缩进语法错误及 `model_runner_v1.py` 超长注释；三项检查均通过。

### 08-12 20:57

- 新增 `scripts/lint_ruff_markdown.sh`，固定在 `vllm-ascend` 中运行 `ruff-format`、`ruff-check` 和 `markdownlint`。
- 脚本对自动修复型 hook 自动复跑一次，其他 pre-commit hook 不会被调用。

### 08-12 21:07

- 删除 `vllm_ascend/ops/fused_moe/fused_moe.py` 中无效的 `TODO1` 注释；代码差异检查通过。
- `pre-commit` 尝试运行后因钩子环境初始化超时，提交使用 `-s` 并跳过重复钩子。

### 08-17 17:10

- 将 `.agents/skills/e2e_test/README.md` 整理为标准 `.agents/skills/e2e_test/SKILL.md`。
- 补充 TEST_MODEL、remote CLI、远程 pytest、日志查看和提交前清理流程。
- 通过 skill 元数据校验、markdownlint 和文件级 `git diff --check`。

### 08-17 17:45

- 对齐 remote-plugin 的 Ascend `docker run` 参数：privileged、host 网络、500G shm、driver/npu-smi、数据目录、时区和代理挂载。
- 更新 remote-plugin 容器创建单测与 fake inspect 数据；206 个单测通过，Python 编译检查和 `git diff --check` 通过。
- 本地未安装 ruff，未执行 ruff 检查。

### 08-17 17:53

- 增加 `machines.json.machine_type` 分支，显式识别 A2/A3/A5/310P/Ascend 类型并兼容回退 `tags.chip`。
- Ascend 容器 `--workdir` 改为配置中的 `container.workspace_root`，与后续 workspace 初始化路径一致。
- 更新 PRD 和分支测试；208 个 remote-plugin 单测通过，机器配置 JSON、Python 编译和差异检查通过。

### 09-02 13:28

- 统一 DeepSeek V4 SP attention 外层布局：普通 attention reduce-scatter，DSA-CP full output 按 token 切片。
- 补齐 MTP/DSpark SP shard/gather，并增加 TP 边界与 DSA-CP on/off 精度用例。
- 按要求未运行功能测试；`ruff-format`、`ruff-check` 和 `git diff --check` 通过，无 Markdown 改动。

### 09-02 12:40

- 排查 9.103 `remote sync` 慢（实测 3m44s）：根因是容器 DNS（nameserver 90.91.64.173）不可达，git 写 reflog（materialize 的 reset/checkout/update-ref -m）解析 ident 主机名每次等满 10s，materialize 阶段独占 201s。
- 修复：9.103 容器 `/etc/hosts` 追加 `127.0.0.1 worker-97-4` 与 `::1 worker-97-4`（容器重建后会被重置，需注意）。
- 验证：修复前 update-ref -m/reset/checkout 均 10s；修复后全部 0s，`remote sync 9.103` 总耗时 3m44s → 4s。正常机器基线：17.122 增量 8s、9.143 全量首同步 20.6s。

### 09-02 18:20

- 上游 Qwen3.5 SP 将 positions 的真实 token 数传入 ColumnParallelLinear 做 AG 后 unpad，并由 RowParallelLinear 在 RS 前补齐；MoE 开关仅允许原有 DP/TP/EP 合法拓扑，dense SP 继续使用编译 pass token threshold。
- 81.182 DP2/TP2/EP 对话 SP on/off 输出一致，prompt logprob 最大/平均差 0.0797/0.0137；64×(1024 in/128 out)、并发 16 benchmark：SP off/on 为 148.48/142.28 output tok/s。
- 远端定向测试 7 passed；配置/SP 回归 24 passed、1 deselected（完整 conftest 缺 `tblib` 且机器无法访问 PyPI）。

### 09-03 11:32

- 对齐 DeepSeek V4 DSA-CP 的 `need_gather_q_kv` 路径：SP 本地 Q、KV hidden gather、外部 reduce-scatter，移除 attention 末端 AllGather。
- 按要求未运行测试；3 个改动文件通过 `ruff check`、`ruff format --check` 和 `git diff --check`。

### 09-03 12:53

- 新增下游公共 SP 算子并对齐 DeepSeek V4 model：普通 attention 外部 AG/RS，DSA-CP 保持 local-token 输入输出。
- DSA-CP prefill 使用 full O 权重且跳过 AllToAll；decode 使用 AllToAll、TP O-Proj partial 和内部 ReduceScatter。
- 按要求未运行功能测试；6 个改动文件通过 ruff、格式、包发现和差异静态检查。

### 09-03 13:40
9.103按scripts/vllm_run.sh重起DS-V4-Flash-w8a8-mtp服务（DP2TP4，flashcomm1+dsa_cp，job j-20260903-131147-01，卡0-7）
对话验证正常（中文一句话自答，finish stop）；bench_serve.sh 50/50成功：output 160.88 tok/s，TTFT均值3096ms，TPOT均值43.75ms。旧lm-eval服务已死（进程被Kill）并清理。

### 09-03 14:45
pre-#13946(92ea871b2)同配置性能对照：对话正常；bench 50/50成功，output 163.73 tok/s、TTFT均值2982ms、TPOT均值42.93ms，相对现HEAD(160.88/3096/43.75)快约2%。本地已恢复fix-dsacp@9e859f4d9+stash pop。
远端9.103仍为pre-13946代码：全量sync被新建feat-pass-for-sp/未初始化子模块卡住fail-closed，待处理。

### 09-03 15:50
FusedMMCollectivePass v1远端验证（goal自主推进中）：141.61.81.14空闲16卡；修正其machines.json workspace_root误配site-packages->/vllm-ascend-workspace后全量sync成功；配置面UT首轮暴露两处问题并已修（TP group mock、其它pass构造需ModelConfig故测试只开目标开关）；torch_npu探针：npu_mm_reduce_scatter_base有、npu_mm_all_gather_base无。当前141.61.81.14的22端口SSH连续超时，UT重跑与FX探针待网络恢复后继续。

### 09-03 16:15
fix-dsacp干净HEAD重测(全量sync后)：对话正常；bench 50/50，output 155.26 tok/s、TTFT均值3201ms/p50 3060ms、TPOT均值45.11ms。较第一次HEAD测试(160.88/3096/43.75)略慢，机器抖动约3%。服务留存9.103卡0-7供用户自测。

### 09-03 17:30
FusedMMCollectivePass转17.119继续（141.61.81.14 SSH持续超时）：远端为parity快照树，以cp-r+sync --paths搭验证树，避开0-3卡在线服务，用8-15卡。
配置面UT 2 passed；探针确认npu_mm_all_gather_base无、npu_mm_reduce_scatter_base有（world_size限2/4/8，需hcom）、上游AsyncTPPass依赖CUDA symm_mem故NPU需自有pass；V2-Lite TP2开/关对照job j-20260903-171107-01运行中。

### 09-03 17:35
17.119探针环境坑定位：`remote run`命令里`:$PYTHONPATH`自引用不可靠，worker报`No module named 'acl'`；
改用绝对路径`export PYTHONPATH=<feat树>:/usr/local/Ascend/cann-9.1.0/python/site-packages`后`acl.rt`与feat树导入正常。
V2-Lite TP2开/关对照重跑job j-20260903-172800-01（卡8-9）。
另记：v1 pattern-1 `mm(AG)->AG(mm)`仅对复制权重（如MLA `ReplicatedLinear` a_proj）数学成立，
柱切分MLP遇到AG输入会被改错；待开/关输出对照实证后再定是否加作用域守卫。

### 09-03 19:00
现HEAD(b1a428e10)在9.103(A3)上起服后首个prefill即崩：dsa_cp.py:1469 RuntimeError，DSA_O_PROJ_TP仅A5 profile支持，A3硬拒绝。200prompts对比阻塞，待用户定对比基线。
pre-13946 200prompts已测：冷TTFT均值2857/p50 2586/吞吐175.66，热TTFT均值2495/p50 2559/吞吐173.09。

### 09-03 19:05
FusedMMCollectivePass v1 在 17.119 V2-Lite TP2 完成编译版开/关对照：开 fuse 后 pass 正常触发但 `replaced 0`，FX dump 显示真实图只有 `all_reduce`+`unquantized_gemm`+`fused_moe`、无 `mm(AG)`/`quantize(AG)` 站点；70-token 长 prompt 开/关输出逐字一致；配置面 UT 重跑 `2 passed`，ruff 全 clean。结论：v1 在此拓扑是安全 no-op，不加作用域守卫（pattern-1 行线性代数上恒保 rank 内数值；实弹需含 `mm(AG)` 的拓扑如下一步 Qwen3.5 SP/MLA）。
排障沉淀：HCCL 多实例每 phase 需独立数字端口段（`auto` 会 hang 在 HCCL init、默认报 error 7）；17.119 验证树曾新旧混用（manager 新/`hardware_profile.py`+`ascend_config.py` 旧致 `AttributeError`），已 sha256 对齐；`cudagraph NONE` 下 torch.compile 不跑、既有 eager 探针全部无效，须 `FULL_DECODE_ONLY`；job 状态滞留 running 系 tracker 假象，以远端进程/done 文件为准；17.119 的 `vllm` 来自第三方快照（`/mnt/share/.../zhaochuang001/vllm`），非 workspace 树。

### 09-03 19:20
按要求：删fix-dsacp stale worktree(prune)、本地切回fix-dsacp@a76098e65、全量sync、9.103起服。同样的A3门禁崩溃(dsa_cp.py:1469)复现，对话首请求即500。现HEAD在A3上无法跑dsa_cp性能，对比待分支修复。

### 09-03 22:00
A3加DSA_O_PROJ_TP(c8f6b05f0，已-s提交，2文件)后dsa_cp在A3跑通。200prompts冷热对比：pre-13946冷2857/热2495(TTFT均值)，现HEAD冷2848/热2313；热机吞吐173.09→178.00。删flashcomm无回退，热机TTFT反优7%。model.py用户改动未动。

### 09-07 15:00
17.119 GLM-5.2-W4A8C8(DP2+TP4,8010)上ais_bench：AIME2024 accuracy 13.33(4/30,推理约55分钟)，结果已pull到`.log/aisbench-aime2024-0907/`；GSM8K全量仅跑13条即按要求stop；ais-bench SKILL.md按实战重写（NO_PROXY、tee日志、pull链路、坑点）。

### 09-07 15:15
ais-bench SKILL.md按house格式重整（frontmatter+适用范围/核心边界/标准配置/结果核验/换数据集/相关参考，对齐lm-eval-precision-benchmark），已-s提交；markdownlint除MD013行长外clean。

### 09-07 15:25
用户取消GSM8K-100计划：重启服务j-20260907-150947-01在权重100%读完、起服前kill，卡0-7已释放，评测目标pause；已下发的gsm8k_100配置模板留本地.temp/ais_bench/备用。

### 09-07 15:45
补充 `vllm-ascend/docs/source/user_guide/feature_guide/sequence_parallelism.md` 的 SP MoE 原理说明，覆盖 token 切分、EP 通信和不均衡 token 的 padding 处理。
独立 markdownlint 与 `git diff --check` 通过；当前环境未安装 pre-commit，未运行功能测试（文档变更）。

### 09-08 SP 上游代码实现
在 sp-refactor-new 实现 SP 统一开关、DP=1 校验、共享专家输出修复及 dense >1000 动态切换。
补充配置/linear/E2E 用例；保留 RS pad；动态 dense SP 首版关闭 CUDA graph。
按用户要求未运行测试或 lint；详细状态见 docs/PRs/refactor_sp_upstream.md。

### 09-08 PR 草稿
在 docs/PRs/refactor_sp_upstream.md 追加上游 PR 标题与正文草稿。
说明 SP 行为、兼容性、dense CUDA graph 限制、测试与模型评估待办及 AI 参与。
仅编写本地文档，未运行验证、检索重复 PR 或发布到 GitHub。

### 09-08 8K TTFT 性能对照完成
179 H20 4–7 卡，Qwen3.5-35B-A3B BF16，DP1/TP4/EP V1 eager；补编缺失 embedding 算子并通过参考对照。
8192 输入 / 1 输出、并发 1，每组 2×32 请求全成功；SP off/on 平均 TTFT 267.40/254.02 ms（-5.00%，存在轮间波动）。
日志已拉回 .log/spbench-results；服务已停止；记录见 docs/PRs/refactor_sp_upstream_benchmark.md。

### 09-08 SP-on 对话检查
179 H20，DP1/TP4/EP V1 eager：10 次 chat 请求均 200；9 次关闭 thinking 的请求正常结束，无乱码。
算术、长文本提取和 3 路并发正常；多轮活动推荐约束遵循不严，默认模板在 512 token 推理阶段截断。
请求响应已拉回 .log/spbench-results/chat-sp-on-49ddb9f27e；服务已停止；未做 SP-off 精度对照。

### 09-08 Dense SP CUDA graph 支持
删除 dense SP 禁图分支；1000 阈值分隔捕获桶，SP/TP 独立编译及 AOT 缓存共享权重，重置时清理。
H20 3 项回归通过；Qwen3.5-4B FULL(decode-only)/PIECEWISE 捕获与交替阈值输入跑通，生成 token 与基线一致。
PIECEWISE 所选 token logprob 最大差 0.0148；记录与日志见 docs/PRs/refactor_sp_upstream_benchmark.md。

### 09-08 16:55 9.138 vllm-ascend 源码树补齐编译产物（自产自销）
9.138 上 4 次 pip 编译均未生效（装到了非服务 python 侧，源码树仍缺 cp311 产物导致 PYTHONPATH 方式 import 失败）。
把本机 site-packages（09-05 d8c9e3a67 预编译）的 vllm_ascend_C.so/libvllm_ascend_kernels.so/_build_info.py/_version.py 与 68M _cann_ops_custom 补进 /vllm-ascend-workspace/vllm-ascend/vllm_ascend/（cp -a / cp -an 只补缺，git 保持干净）。
源码树 PYTHONPATH import 验证 OK（platform 激活、device_type A3）。17.119 的 09-07 产物因工作区抖动+正跑服务未采用。

### 09-08 21:15
- 同步实测 workflow 到 skill：`ais-bench` 新增 MODEL-DATA-001 核验、gsm8k 数据格式核验与修复流程、双客户端互斥、17.111 gsm8k200 accuracy 97.50 记录；重写 `vllm-ascend-benchmark`（bench serve 标准流程 + 实测记录）；新增 `scripts/bench_perf.sh`。
- 17.111 性能（DSpark-w4a8-int4 DP2+TP4+EP，random 4096-in/1024-out，并发 8）：100/100 成功，输出吞吐 136.46 tok/s，TTFT 均值 1397ms、TPOT 均值 55.55ms；产物 `.log/bench-dspark-0908/`。
- markdownlint 通过（benchmark 干净；ais-bench 仅剩与原文一致的 MD013 行长）。
