---
name: solve-remote-env
description: 在 vLLM Ascend 远程 NPU 机器上部署、编译、启动服务或运行评测时，依据当前机器事实处理环境配置与已验证坑点。实时记录坑点+解决方案
---

# solve-remote-env

这个 skill 用于解决远程 NPU 环境问题，并沉淀可复用的配置和故障处理。它不替代
`remote-plugin` 的机器管理、同步、Job 和日志能力；凡是远程操作，都通过
`.agents/skills/remote-plugin/remote`（或安装后的 `remote`）执行，不裸用 SSH 或 Docker。

## 信息来源和优先级

使用目标机器前按以下顺序确认信息：

1. `.remote/machines.json`：机器登记、`mode`、宿主机、容器、镜像、`workspace_root`、硬件标签。
2. `.remote/state/docs/<alias>.facts.json`：最近一次 `remote verify` 生成的 OS、磁盘、NPU、Python、
   pip 源和代理事实。缺失、过期或与现场不符时，先 `remote verify <alias>`。
3. `.remote/state/docs/overall.md` 和 `docs/instructions/`：人类维护的补充说明和当前推荐流程。
4. `CONTEXT.md`、`scripts/`、历史实验文档：只把带日期、机器和验证结果的内容当作历史经验，不能覆盖当前 facts。

当前工作区的机器事实不是实时租约；`verify_status: degraded`、磁盘满、卡数不匹配或 SSH
不可达都必须在选机时显式处理。代理密码、SSH 密码、token 和其他凭据不得写入本文件、命令、Job
描述、日志或 Git 跟踪文件。`docs/instructions/proxy_setup.sh` 中存在带认证信息的历史代理配置，
阅读时只取网段和端口，实际凭据从用户环境变量或安全输入取得。

旧快照偶尔按宿主机名保存 facts（例如 `141.61.41.195.facts.json`），不一定和 alias 文件名相同；
按 alias 找不到时检查 `state/docs/` 中的实际文件，或重新 verify，不要把“找不到文件”当成“没有事实”。

## 远程操作标准流程

仓库当前没有根目录 `./remote` 文件；从仓库根目录直接使用：

```bash
REMOTE="${REMOTE:-./.agents/skills/remote-plugin/remote}"

"${REMOTE}" machines
"${REMOTE}" machines --probe
"${REMOTE}" status <alias> --probe
"${REMOTE}" verify <alias>       # facts 缺失、过期或异常时执行
"${REMOTE}" up <alias>           # 容器模式拉起/复用；SSH 模式只初始化工作区
"${REMOTE}" sync <alias>        # 只同步代码，不安装、不编译
```

实际干活前先看 `machines` 的 running jobs 和卡级占用。`alias` 使用
`.remote/machines.json` 中的机器别名，不要用容器名代替。容器模式是“SSH 到宿主机，再由
remote-plugin 执行 `docker exec` 进入容器”；容器通常不需要 sshd，`ssh_port: 46000` 是登记元数据，
不能据此改用容器 SSH。

同步后必须显式执行安装、编译或测试，例如：

```bash
"${REMOTE}" run <alias> \
  --cmd 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; <command>' \
  --timeout 3600
```

短命令优先使用阻塞的前台 `run`。服务或其他长任务才使用后台 Job，并声明任务和卡：

```bash
"${REMOTE}" run <alias> \
  --background --task "vLLM 服务/测试用途" --cards 0,1,2,3 \
  --cmd 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; <command>' \
  --timeout 10800
"${REMOTE}" logs <job-id> --follow
"${REMOTE}" jobs --machine <alias>
```

后台任务按仓库约定以 5 分钟粒度检查 `remote jobs`/`remote logs`；不要高频轮询。结束后确认
卡占用释放。`--logs none` 的前台任务没有 Job 记录；后台默认保留合并的 `full.log`。Job 显示
`running` 但进程已退出时，先核对实际进程、端口和日志，再只对对应 Job 执行
`remote stop <job-id>`。

CLI 进度走 stderr，最终结果是 stdout 单行 JSON；判断成功时看 `status`、`exit_code` 和 `preview`，
不要根据混合输出中的某一行猜结果。

`remote up` 是幂等操作：无镜像时拉取，有镜像无容器时创建，健康容器直接复用。已有容器的镜像
或设备挂载发生漂移时通常只告警，不会自动重建；只有容器未运行或无法 `docker exec` 才进入
`needs_repair`。容器模式的预期配置是 host network、较大的共享内存、Ascend 设备/driver/dcmi/
`npu-smi` 挂载，以及 `/home`、`/mnt`、`/workspace`、`/tmp` 等数据目录挂载；应让 machine
登记和 `remote up` 负责这些配置，不要另起一个同名容器。`remote down` 会停止并移除受管容器，
除非用户明确要求，不要执行。

## 当前登记的远程环境

以下是 2026-08-28 工作区的登记快照；使用前仍以 `machines.json` 和 `verify` 为准。当前登记的
容器/SSH 工作区根路径均为 `/vllm-ascend-workspace`，但历史服务记录中仍有 `/vllm-workspace`，
两者不能混用。

<!-- markdownlint-disable MD013 -->
| alias | 登记硬件和模式 | 容器/镜像要点 | 已知选择风险 |
| --- | --- | --- | --- |
| `17.111` | A3，`ascend-910c`，16 卡，container | `xrs_vllm_main`，2026-08-11 A3 镜像 | 已验证过大模型和 Kimi K2.5；facts 曾报告卡数不匹配、pip 源不可达，先 verify 和查占用 |
| `17.122` | A3，`ascend-910c`，16 卡，container | `xrs_vllm_main`，2026-08-11 A3 镜像 | facts 磁盘余量 0 GB/100%，不要在未清理前编译；历史 CP e2e 使用过此机 |
| `17.110` | A3，`ascend-910c`，16 卡，container | `xrs_vllm_main`，2026-07-30 A3 镜像 | 历史记录称只有 8 物理卡/512 GB，放不下 K2.5；登记卡数与 facts 曾不一致，必须现场核对 |
| `81.162` | A3，`ascend-910c`，16 卡，container | `x50063850-kimi-prof`，2026-08-20 A3 镜像 | facts 磁盘约 2 GB、99%，不适合作为编译首选 |
| `9.103` / `9.143` | A3，16 卡，container | `xrs_vllm_main`；镜像分别为 2026-08-05 A3 和 `qwen3.8-a3` | facts 显示磁盘接近/达到 100%，网络/DNS 也可能不可用；登记说明当前只支持单机 |
| `17.119` | A3，16 卡，container | `xrs_vllm_main`，`kv-non-contiguous-b093-20260825` | 最近 verify 为 SSH 超时/不可达，不要把镜像登记当作机器可用 |
| `41.195` | A5，`ascend950dt`，8 卡，container | `xrs_vllm_main`，2026-08-17 A5 openEuler 镜像 | facts 只有约 6 GB 空闲；A5 需要遵守下方设备、镜像和 URMA 特殊规则 |
| `141.62` / `141.66` | A5，8 卡，直接 SSH | 无 container 配置 | 当前登记备注为 SSH 认证失败，尚未查询镜像，不要直接选用 |
<!-- markdownlint-enable MD013 -->

模型路径要先在目标机检查是否存在。当前文档记录过的共享路径包括：

- 90 网段的 `/mnt/a800_weights`：Qwen3.6-35B-A3B、Qwen3-30B-A3B、DeepSeek-V4-Flash-w4a8、
  GLM-5.2-w4a8、MiniMax-M3。
- 80 网段 17.111/相关机器的 `/mnt/weight`：Qwen3-30B-A3B、Qwen3.5-35B-A3B、
  DeepSeek-V4-Flash-w4a8；17.122 还记录过 DeepSeek V3/V4 系列。
- 多机可见的 `/mnt/share/weights`、`/mnt/weights`；Kimi K2.5 历史路径为
  `/mnt/share/weights/kimi-k2.5-w4a8_modelscope`。

### A3 卡映射

17.111 的历史验证是 8 物理 NPU × 2 chip = 16 逻辑卡，逻辑编号关系为
`logical = physical * 2 + chip`，每个逻辑卡约 64 GB。`npu-smi` 在不同镜像/布局下可能报告
重复或额外条目，不能只用 `npu_count` 判断可用卡；要同时看 `tags.cards`、`cards_match`、实际
HBM 和任务占用。`ASCEND_RT_VISIBLE_DEVICES` 必须与测试实际使用的逻辑卡、`remote run --cards`
和 TP/DP/EP 配置一致。

### A5 容器差异

- `machine_type` 使用 `A5`，SoC 使用 `ascend950*`（当前是 `ascend950dt`）；使用登记的明确镜像，
  不要自动拼接 `-a5` 后缀。
- 宿主机没有 `/dev/devmm_svm` 时不要强行挂载；保留 `/dev/davinci_manager`、`/dev/hisi_hdc`
  等必要设备。
- 容器缺 URMA 运行库时，需要从宿主机补齐到容器 `/usr/local/lib`，并加入运行时库搜索路径；
  先确认库和 ABI，再做补齐。

## 远端工作区和基础环境

不要把工作区路径写死到历史路径。进入目标后以 machine profile 的 `workspace_root` 为准：

```bash
WORKSPACE_DIR="${WORKSPACE_DIR:-/vllm-ascend-workspace}"
VLLM_DIR="${WORKSPACE_DIR}/vllm"
ASCEND_DIR="${WORKSPACE_DIR}/vllm-ascend"

# remote run 不保证自动加载 shell profile；先加载 CANN，再追加源码路径。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 按目标机器、任务和占用情况替换，不要照抄默认卡号。
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${ASCEND_DIR}:${VLLM_DIR}:${PYTHONPATH:-}"
export VLLM_VERSION=0.27.1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400
export HCCL_BUFFSIZE=1024
export VLLM_LOGGING_LEVEL=DEBUG
```

`VLLM_VERSION` 必须和当前 vLLM/vllm-ascend 代码兼容。当前测试和脚本以 `0.27.1` 为基准，
但源码快照可能没有可用的 `_version`，或 `vllm.__version__` 是 dev 版本；这时必须显式设置，
不能依赖安装器猜版本。benchmark 稳定运行时可以将日志级别改为 `INFO`。遇到
`ParallelOpenMP.cpp:64` 的 `set_num_threads`/`pool INTERNAL ASSERT FAILED`，再增加：

```bash
export OMP_NUM_THREADS=1
```

部分 CP/调试任务还使用过 `VLLM_USE_V2_MODEL_RUNNER=1` 或
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；这类变量只按对应测试命令需要启用，不要
把实验开关当作所有模型的默认环境。

### CANN 和 ATB

`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 必须发生在设置源码 `PYTHONPATH` 之前，
且 `PYTHONPATH` 要追加 `${PYTHONPATH:-}`，不能直接覆盖 CANN 的 Python 路径。否则常见症状是
`ModuleNotFoundError: No module named 'acl'`。

17.111 的历史服务还要求环境文件包含 ATB 路径：`ATB_HOME_PATH` 正确指向镜像内 ATB 根目录，
`LD_LIBRARY_PATH` 包含 `nnal/atb/latest/atb/cxx_abi_1/lib`；缺少它会在 worker 启动时报
`libatb.so` 找不到。镜像路径不同就先检查实际目录，不要猜路径或复制其他镜像的库。

## 网络和 pip 配置

安装、下载模型或安装 lm-eval 前，先看目标 machine facts 的 `has_proxy`、`proxy_env`、
`pip_index_url`、`pip_index_reachable`、`dns_ok`。仓库 docs 的常用 pip 源是：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

但 facts 当前可能记录 Huawei 源或源不可达；源和代理事实不确定、报错与 facts 不一致时，停止并
询问可用代理/镜像，不要盲目重试或擅自换源。

按网段选择 `docs/instructions/proxy_setup.sh` 中的代理配置，认证 URL 只通过安全环境变量传入：

```bash
PROXY_URL="${PROXY_URL:?请从安全渠道提供当前代理 URL}"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export NO_PROXY="127.0.0.1,localhost,<目标机器 IP>,<服务地址>"
export no_proxy="$NO_PROXY"
```

已记录的无凭据代理端点是 80 网段 `80.253.137.110:7897`、141 网段
`141.3.183.246:7897`，90 网段使用 `90.255.24.103:6688` 的认证代理；端点也可能变化，
以目标机器当前事实和用户确认结果为准。

`no_proxy` 必须包含服务实际地址和本机回环地址。历史上 `vllm bench serve`/HTTP 客户端的
`trust_env=True` 把发往 `127.0.0.1` 的请求送进代理，导致 50/50 请求全失败且服务端没有 access
日志。排查本机服务时同时设置大小写变量，并用：

```bash
curl --fail --noproxy '*' http://127.0.0.1:8010/health
curl --fail --noproxy '*' http://127.0.0.1:8010/v1/models
```

## 同步、安装和编译

`remote sync` 只做代码同步，不会自动安装依赖、生成 `_build_info.py` 或编译自定义算子。
结果中的 `bundle_transfer.modes`：

- `skip`：远端已有相同 snapshot；
- `delta`：基于上一 snapshot 的增量 bundle；
- `full`：首次同步、parity 缺失或远端漂移时的 fail-closed 回退。

同步后普通 vllm-ascend 开发流程只编译插件、用 `PYTHONPATH` 使用 workspace 内 vLLM：

```bash
cd "${ASCEND_DIR}"
rm -rf csrc/build
export COMPILE_CUSTOM_KERNELS=0
pip install --no-cache-dir --no-deps --no-build-isolation -ve .
```

`rm -rf csrc/build` 只允许指向已确认的目标工作区；磁盘接近满时先查空间和缓存，不要用宽泛
路径清理。若首次命令失败，按原因处理：先确认 CANN/代理，再确认 pip 源；只有在依赖事实明确
且用户允许时，才按 `container_compile_flow.md` 尝试普通 `pip install -ve .`。不要把网络错误
伪装成编译错误，也不要无限重试。

### 自定义算子和生成物坑

- 出现 `aclrmsnorm not found` 等算子缺失时，尝试在匹配镜像中使用
  `COMPILE_CUSTOM_KERNELS=1`；仍失败通常是 CANN/镜像 ABI 不匹配，应换兼容镜像并记录版本。
- sync 不传生成物。若出现 `ImportError: cannot import name '_build_info'`，在目标 `vllm-ascend`
  目录重新执行 editable install 生成它。
- `COMPILE_CUSTOM_KERNELS=0` 的重装可能卸载镜像原有的 `vllm_ascend_C*.so`。随后 Dynamo
  在 `enable_custom_op()` 内联导入时会报 `torch._dynamo.exc.Unsupported: Import failure`。
  安装前先确认磁盘、现有扩展和是否有同镜像同 commit 的构建产物。
- 17.122 曾因容器盘 100% 满无法运行 opbuild。最后手段是从已验证的同镜像同 commit 工作区
  取得 `vllm_ascend_C*.so`、`libvllm_ascend_kernels.so`（含 `lib/` 下副本）和完整
  `vllm_ascend/_cann_ops_custom/` vendor 包；缺 vendor 包可能报
  `aclnnAddRmsNormBias ... not in libopapi.so`。复制前必须核对 Python/架构/CANN/commit ABI，
  否则应换镜像或询问，不要随意混用机器产物。
- 2026-09-08 `9.138` 无编译产物修复（已验证）：该机存在双工作区，pip editable 指向旧
  `/vllm-workspace`（镜像构建产物齐全），而 `remote sync` 只写新 `/vllm-ascend-workspace`
  （无 `_build_info.py`/`.so`）。修复 = sync 后在新树重装（`COMPILE_CUSTOM_KERNELS=1`）。
  新树 `csrc/third_party` 为空且容器无外网（`getaddrinfo failed for gitcode.com`）时，
  opbuild 会按 `makeself` → `json(include.zip)` 顺序失败；从同机旧树复制
  `third_party/pkg/*`（abseil/protobuf/include.zip，供 cmake `file://` 缓存）、
  `third_party/makeself/` 全目录（命中 `makeself-header.sh` 存在性守卫跳过下载）、
  `third_party/json/`（命中 `nlohmann/json.hpp` 守卫）后增量重编通过。
  注意构建命令不要套 `| tail`，否则远端合并日志在结束前无输出、失败根因也被截断。
- 旧 parity 流程遇到 requirements pin（例如某版本 `triton-ascend` 源中不存在）时，历史降级
  路径是 materialize 源码 → 对 vllm 和 vllm-ascend 做 `COMPILE_CUSTOM_KERNELS=0` 的 no-deps
  editable install → 复用兼容编译产物 → 使用 `serve_start --skip-parity`。这不是当前
  `remote sync` 的默认流程，只有明确处于 parity 工作流时使用。

## 服务、在线 benchmark 和 GSM8K 评测

服务参数必须同时记录模型路径、`ASCEND_RT_VISIBLE_DEVICES`、TP/DP/EP、端口、日志路径和
完整启动参数。`scripts/vllm_run.sh` 是当前实验脚本，不是通用入口：它硬编码了工作区、DeepSeek
模型、TP4、profiler、speculative decoding 和 DSA/SP 开关，使用前要逐项检查。

服务启动后先做健康检查，再做请求或 benchmark：

```bash
curl --fail --noproxy '*' http://127.0.0.1:8010/health
curl --fail --noproxy '*' http://127.0.0.1:8010/v1/models
```

在线 benchmark 的服务/客户端契约通常是：服务 `--served-model-name qwen`，客户端
`--backend openai`、`--base-url http://127.0.0.1:8010`、`--endpoint /v1/completions`。
不要把权重目录当成 model name，也不要把 completions 和 chat completions endpoint 混用。
`vllm bench serve` 若需加载远程 tokenizer/config，按模型增加 `--trust-remote-code`；大模型
加载慢时，健康检查超时不能沿用 300 秒，Kimi K2.5 历史验证使用过 `--health-timeout 1800`。
benchmark 先 warmup，比较版本时固定硬件、模型、环境、输入输出长度、并发和 warmup，并保存
原始 JSON，不只比较一个均值。

精度评测使用 `/mnt/share/datasets/gsm8k` 的本地数据和 `/v1/completions` 服务，不要使用会从
Hugging Face 下载数据的内置 `gsm8k` task。当前 `scripts/lm_eval.sh`/配套 skill 的稳定约定是：

- `model=qwen` 是服务暴露名；`tokenizer_backend=none` 避免当前容器的 tokenizer/config 识别问题；
- `batch_size=1`，`NUM_CONCURRENT=8` 是已验证起点，异常时降低并发；
- `LIMIT=1` 只做链路 smoke，完整运行必须清空 `LIMIT`，日志确认 `limit: None`；
- `/mnt/share/datasets/gsm8k/test.parquet` 当前预期转换为 1319 条，结果和样本放 `.temp/`，日志放 `.log/`；
- 客户端同样要设置大小写 `NO_PROXY/no_proxy`，并在全量前依次验证健康接口、单请求、`LIMIT=1`、
  一条 sample，再恢复全量。

如果当前工作树的 `scripts/lm_eval.sh` 仍保留尾部旧版命令，注意其中 `--log_samples` 后缺少续行
反斜杠，会在首个评测结束后再次调用 lm-eval，甚至把 `--output_path` 当成 shell 命令；先检查
脚本尾部，避免把脚本语法问题误判为远程服务问题。

## 症状速查

<!-- markdownlint-disable MD013 -->
| 症状 | 先检查 | 处理 |
| --- | --- | --- |
| `No module named 'acl'` | 是否 source CANN、是否覆盖了 PYTHONPATH | 先 source `set_env.sh`，再追加源码路径 |
| `Invalid vllm version dev`、取不到 `FusedMoEFactory` | workspace 是否无 `_version`/git tag | 显式设置与代码匹配的 `VLLM_VERSION`，当前基准为 `0.27.1` |
| `cannot import name '_build_info'` | 是否刚做过 `remote sync` | 在目标 `vllm-ascend` 重新 editable install |
| `libatb.so` 找不到 | `ATB_HOME_PATH` 和 `LD_LIBRARY_PATH` | 补齐镜像对应的 ATB `cxx_abi_1/lib` 路径 |
| `ParallelOpenMP.cpp:64` | EngineCore 启动线程数 | 增加 `OMP_NUM_THREADS=1` |
| `aclrmsnorm not found` | 自定义算子是否编译、CANN/镜像是否匹配 | 在兼容镜像尝试 `COMPILE_CUSTOM_KERNELS=1` |
| `Free memory ... less than desired GPU memory utilization` 或 e2e 等待显存超时 | `npu-smi`、残留进程、共享机占用 | 先排除环境/占用，再判断代码；不要只调显存比例 |
| 本机请求无响应、服务端无 access 日志 | `NO_PROXY/no_proxy`、代理泄漏 | 回环地址加入 no-proxy，curl 使用 `--noproxy '*'` |
| `tar: vllm/.git: Cannot open: File exists` | 子模块 `.git` 文件/目录形态 | 抽查远端目标文件和 commit；确认已解压后无需盲目重同步 |
| Job 仍是 `running` 或 `full.log` 不更新 | 实际进程、端口、远端临时日志 | 按 5 分钟粒度核对；确认退出后只清理对应 Job |
<!-- markdownlint-enable MD013 -->

全量 benchmark/e2e 失败时，不要直接重复跑完整任务。先查客户端配置，再同时看服务 stdout
和 stderr，手工发一条请求，最后以最小并发复现；服务未健康前不要消耗长时间模型加载。

本地 Mac/PC 不运行依赖 `torch_npu` 的测试；本地只做文档/静态检查，NPU 编译、服务、UT 和
e2e 直接在目标远端容器验证。临时文件放 `.temp/`，日志放 `.log/`；远程产物需要带回时用
`remote pull`，相对路径以目标 `workspace_root` 为基准，并保存到带 alias/时间戳的独立目录。

## 记录新坑点

新增经验必须带日期、alias、镜像/代码版本、症状、根因、修复和验证结果，并注明是“已验证”、
“历史记录”还是“未实测”。建议格式：

```markdown
### YYYY-MM-DD `<alias>` / `<场景>`（已验证|历史|未实测）

- 症状：
- 根因：
- 修复：
- 验证：命令、结果、日志/产物位置
```

不要把一次机器故障升级成所有机器的默认配置；不要把代理凭据、密码或 token 作为“解决方法”
写入记录。
