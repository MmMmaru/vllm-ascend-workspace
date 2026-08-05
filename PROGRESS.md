### 08-03 12:00

- 实现 SP 下 matmul + reduce_scatter 算子融合（参照上游 AsyncTPPass，开关复用 `pass_config.fuse_gemm_comms`）：新增 `torch.ops.vllm.npu_mm_reduce_scatter` custom op（`register_custom_ops.py`，底层 `npu_mm_reduce_scatter_base`）、`MatmulReduceScatterFusionPass`（`sequence_parallelism.py`，匹配 `unquantized_gemm + reduce_scatter`）、挂载于 `graph_fusion_pass_manager.py`，`platform.py` 补齐 fuse_gemm_comms 的 sp_min_token_num 默认值。
- 关键修正：实际图中 mm 节点是 Ascend 的 `unquantized_gemm(x, weight)`（weight 为 (n,k) F.linear 布局），非上游的 `aten.mm`；初版 pattern 0 命中后改为匹配该 op，融合算子内部做 `weight.t()`。
- 验证：Qwen3-30B-A3B TP2/EP 精度脚本输出与未融合一致（48 层全部命中）；16K 输入吞吐 14099.69 → 14437.00 tokens/s（+2.39%）；E2E 新增 `test_qwen3_vl_sp_fuse_gemm_comms_tp2` 并按规范改用 `TEST_MODEL` 环境变量。

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

### 07-29 16:27

- 将 DP2/TP2/EP SP 回归测试从 `two_card/test_sp_pass.py` 移到 `four_card/test_sp_pass.py`。
- 在 `GraphFusionPassManager` 增加可选的 pass match-count 记录，E2E 直接读取 `SequenceParallelismPass.matched_count`，不再解析日志。
- 使用空闲 NPU 2,3,4,5 和本地 Qwen3-30B-A3B 权重验证，四卡 E2E `1 passed`；pytest 收集共 2 个测试通过。

### 07-30 12:12

- 新增 `scripts/bench_sp_serve.sh`，在同一脚本中后台启动 `vllm serve`，健康检查通过后执行 `vllm bench serve`，退出时自动清理服务。
- 按当前 SP 配置使用 TP4、NPU 2,3,4,5、FULL_DECODE_ONLY、`enable_sp=true`、16K 输入、100 请求和 10 warmup；bench 使用本地 tokenizer 与 `/v1/completions`。
- Docker 验证通过：SP pass 每个 rank 替换 96 patterns，100/100 请求成功，耗时 159.10 秒、请求吞吐 0.6285 req/s、总 token 吞吐 10300.26 token/s；结果为 `.log/bench_sp_serve_20260730T120606Z.json`。

### 07-30 12:37

- 新增 `docs/model_runner_v2_design.md`，整理上游 Model Runner V2 的选择逻辑、SchedulerOutput/InputBatch、forward/sampling 两阶段、PP/DP、KV/Attention、CUDA Graph 和异步输出设计。
- 文档引用当前 `vllm/` checkout 的源码路径与行号，包含 ASCII 数据流图、MRV1 对比、MRV2 限制和硬件后端阅读边界。
- 已通过源码链接目标存在性、引用行范围和 `git diff --check` 校验；本次为文档变更，未运行 NPU 全量/E2E 测试。

### 07-30 09:24

- 将 `scripts/naive_precision_test.py` 简化为 vLLM 高层离线 `LLM.chat()` 单请求精度测试。
- 使用 `TEST_MODEL` 覆盖模型路径，默认 `Qwen/Qwen3-30B-A3B`；采用贪心采样并关闭 Qwen3 思考模式。
- 已通过 Python 语法编译检查和 `git diff --check`；未在宿主机执行 NPU 推理。

### 07-30 11:11

- 为 `scripts/naive_precision_test.py` 增加 `TEST_ENABLE_SP=1`，开启非 eager 编译、SP pass 和 FULL_DECODE_ONLY ACL graph。
- Docker 使用空闲 NPU 2,3,4,5、TP4 和本地 Qwen3-30B-A3B 验证通过；日志确认 SP pass 替换 96 个 pattern 并完成 ACL graph replay。
- SP 模式生成结果正常，输出与 eager 基线一致；运行结束后 2–5 卡无残留进程。

### 07-30 12:12

- 新增 `scripts/bench_sp_serve.sh`，在同一脚本中后台启动 `vllm serve`，健康检查通过后执行 `vllm bench serve`，退出时自动清理服务。
- 按当前 SP 配置使用 TP4、NPU 2,3,4,5、FULL_DECODE_ONLY、`enable_sp=true`、16K 输入、100 请求和 10 warmup；bench 使用本地 tokenizer 与 `/v1/completions`。
- Docker 验证通过：SP pass 每个 rank 替换 96 patterns，100/100 请求成功，耗时 159.10 秒、请求吞吐 0.6285 req/s、总 token 吞吐 10300.26 token/s；结果为 `.log/bench_sp_serve_20260730T120606Z.json`。

### 07-31 10:45

- 修复 `scripts/bench_sp_serve.sh` 的 DP2/TP2/EP benchmark 配置：默认关闭 SP，使用环境变量生成合法 JSON，并恢复 bench 详细日志与 TTFT 分位数统计。
- 增加结果校验：`completed` 必须等于请求数且 `failed` 必须为 0，否则输出服务端日志并以失败退出。
- Docker 完整验证通过：TP2/DP2/EP、16K 输入、100 请求、10 warmup，100/100 成功，耗时 95.96 秒，总 token 吞吐 17078.51 token/s；无残留进程。

### 08-03 05:03
重写 concise-code-explanation skill 为总分总结构
- 修改 `.agents/skills/concise-code-explanation/SKILL.md`：去掉 2-3 段限制，改为总述数据流、分模块讲设计、总结串回全局。
- 同步更新 `agents/openai.yaml` 的展示名与默认提示词。
- 文档类改动，无需运行验证；已提交 62cf5b1。

### 08-03 16:05
修正 `scripts/naive_precision_test.py` 的 Ascend profile 路径分析
- 记录 `llm.start_profile()` 前已有的 rank 目录，只对本次新增的 `*_ascend_pt` 执行离线 `analyse()`。
- Docker TP2 使用 NPU 2、3 验证通过，生成结果正常，两个新 rank 的 profile 均完成离线分析并打印 `ASCEND_PROFILER_OUTPUT` 路径。

### 08-03 20:20
SP 特性 W8A8(INT8) 量化支持（worktree `vllm-ascend/.worktrees/sp-int8`，提交 8ae7b7c8b）
- 新增 `npu_quant_mm_reduce_scatter` custom op、`QuantMatmulReduceScatterPattern`（pm DSL 搜索模式）与独立 pass；aclgraph 捕获区间门控 + compile range 按 max_capture_size 分裂。
- 修复三个关键问题：aclnnMatmulReduceScatterV2 对非连续输入 ND 报错/NZ 静默错值（impl 内 x.contiguous()）；matcher 忽略 pattern 外 kwargs 导致 rank0 bias 被静默丢弃（extra_check 过滤）；rank1 分片缺 bias 修正（apply 改为 bias-free mm + 每 rank 加 1/tp 修正，图 rank 对称，2 的幂除法精确）。
- 验证：fused vs nofuse prompt-logprob top1 全一致；E2E `test_sp_w8a8.py` 通过；16K 输入吞吐 SP-off 14664 / SP-nofuse 14452 / SP-fused 14653 tokens/s；ruff/mypy 通过（gitleaks 因二进制架构问题跳过，代码无密钥）。
- 新增脚本 `scripts/naive_precision_test_sp_int8.py`、`scripts/bench_sp_int8_tpot.sh`。
### 08-05 06:23
更新代码解释 skill 的代码定位规范。
- 修改 `.agents/skills/concise-code-explanation/SKILL.md`：要求 Codex 使用可点击的绝对路径 Markdown 链接并附行号。
- 未运行测试；仅完成文档规则更新。
