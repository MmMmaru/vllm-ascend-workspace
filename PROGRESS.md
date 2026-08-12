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
