---
name: lm-eval-precision-benchmark
description: 使用本地真实数据集和已启动的 vLLM completions 服务执行 lm-eval 精度评测，涵盖数据转换、task 配置、全量运行、结果核验和常见故障排查；不用于在线吞吐 benchmark。
---

# lm-eval Precision Benchmark

## 适用范围

当用户需要通过本地 vLLM OpenAI-compatible 服务跑 lm-eval 精度评测时使用本 skill，尤其是：

- 本地 Parquet/JSONL 数据集，不希望访问 Hugging Face 下载数据；
- 服务使用 `/v1/completions` 接口，模型通过 `--served-model-name` 暴露；
- 需要把数据转换成 lm-eval 自定义 task，并保存完整结果和逐样本输出；
- 需要区分 smoke test、完整数据集和结果解析问题。

在线请求吞吐、TTFT、TPOT、ITL 等性能测试使用 `vllm-ascend-benchmark`，不要把两种测试混用。

## 核心边界

- 评测前先确认服务已启动且健康；不要在 lm-eval 客户端里隐式启动或停止服务。
- 原始数据集只读。转换后的 JSONL、YAML、结果和日志分别放在 `.temp/`、`.log/`，不要修改 `/mnt/share/datasets`。
- 远程操作只能通过仓库内的 remote-plugin CLI，禁止裸 SSH。远程 NPU 服务需要登记卡占用；评测客户端本身通常不占 NPU 卡。
- 优先让评测客户端以前台阻塞方式运行。确需后台运行时，必须声明任务；按仓库规范每 5 分钟检查一次状态，不能高频手动轮询。
- 不把代理密码、token 或其他凭据写入 skill、task、脚本或日志。安装依赖前先依据目标机器 facts 和现有网络配置处理代理/源。

## 标准配置

### 服务契约

以当前 DeepSeek-V4-Flash DSpark 配置为例，服务脚本的契约是：

```text
served model: qwen
endpoint:     http://127.0.0.1:8010/v1/completions
dataset:      /mnt/share/datasets/gsm8k/test.parquet
task:         local_gsm8k
```

`model_args` 中的 `model` 必须填服务暴露名 `qwen`，不是权重目录名；`local-completions` 要使用
`/v1/completions`，不要改成 `/v1/chat/completions`。

服务就绪后，至少检查：

```bash
curl --fail --noproxy '*' http://127.0.0.1:8010/health
curl --fail --noproxy '*' http://127.0.0.1:8010/v1/models
```

若直接从远程容器启动服务，先加载 CANN 环境；`remote run` 不保证自动加载 shell profile：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

当前服务脚本使用 `VLLM_VERSION=0.27.1`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400` 和 `HCCL_BUFFSIZE=1024`。使用源码 overlay 时，
`PYTHONPATH` 必须追加 vLLM/vllm-ascend 路径而不能覆盖 CANN 相关路径；版本快照缺少 `_version` 时要显式设置
`VLLM_VERSION`。

### 本地数据转换

当前 `/mnt/share/datasets/gsm8k` 是真实 GSM8K 题目，但保存为评测管道生成的 Parquet，不是 lm-eval
内置 `gsm8k` task 的 Hugging Face 原始格式。已验证的列结构为：

```text
prompt:       list[{role: string, content: string}]
reward_model: {style: ..., ground_truth: ...}
extra_info:   {...}
```

仓库中的 `scripts/prepare_lm_eval_gsm8k.py` 严格读取 `prompt[0].content` 作为题面、
`reward_model.ground_truth` 作为答案，输出 `text/target` JSONL；字段缺失时直接报错，不使用静默 fallback。

完整 test 数据的已知行数是 1319。推荐用脚本生成 task 文件：

```bash
TASK_DIR=.temp/lm_eval/local_gsm8k_test
python scripts/prepare_lm_eval_gsm8k.py \
  --input /mnt/share/datasets/gsm8k/test.parquet \
  --output "${TASK_DIR}/test.jsonl" \
  --task-yaml "${TASK_DIR}/local_gsm8k.yaml" \
  --task-name local_gsm8k \
  --split test
```

生成的 YAML 应使用 `dataset_path: json`、`data_files.test`、`test_split: test`、
`output_type: generate_until`、`doc_to_text: "{{text}}"` 和 `doc_to_target: "{{target}}"`。
不要只替换 `--include_path` 或 `--tasks` 而不检查 YAML 是否真的指向新的数据文件。

### lm-eval 参数

适用于当前服务的最小完整命令如下；完整评测不要传 `--limit`：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

MODEL_ARGS="model=qwen,base_url=http://127.0.0.1:8010/v1/completions,\
tokenizer_backend=none,max_gen_toks=512,max_retries=1,timeout=300,\
num_concurrent=8"

lm_eval \
  --model local-completions \
  --model_args "${MODEL_ARGS}" \
  --include_path .temp/lm_eval/local_gsm8k_test \
  --tasks local_gsm8k \
  --batch_size 1 \
  --log_samples \
  --output_path .temp/lm_eval/local_gsm8k_test/results \
  2>&1 | tee .log/lm_eval_local_gsm8k_test_$(date +%Y%m%d_%H%M%S).log
```

已封装的入口是 `scripts/lm_eval.sh`，默认完成 Parquet 转换、task 生成、全量评测、样本保存和日志保存：

```bash
# 服务健康后运行完整 test 集
NUM_CONCURRENT=8 bash scripts/lm_eval.sh

# 只验证链路和 task，不代表完整准确率
LIMIT=1 bash scripts/lm_eval.sh
```

其中 `LIMIT` 为空才是完整数据集；`NUM_CONCURRENT=8` 是在当前 `--max-num-seqs 32` 服务上验证过的起点，
服务压力或超时异常时再降低并发。`batch_size=1` 是当前 completions 评测的稳定配置。

`tokenizer_backend=none` 是关键配置：它让客户端不加载本地 tokenizer，只把转换后的字符串发给服务。
当前 DeepSeek 配置与容器中的 Transformers 版本组合会触发 tokenizer/config 识别问题，因此不要额外添加
`tokenizer=/mnt/weight/...`，除非已经确认目标模型的 tokenizer 可以在该环境独立加载。

### 远程执行方式

服务必须常驻，因此服务本身可以作为声明了卡占用的后台任务；评测客户端优先使用阻塞的前台命令：

```bash
REMOTE=./.agents/skills/remote-plugin/remote

"${REMOTE}" machines
"${REMOTE}" run 17.111 \
  --background \
  --task "vLLM lm-eval 精度评测服务" \
  --cards 12,13,14,15 \
  --cmd "bash scripts/vllm_run.sh" \
  --timeout 3600

# 服务健康后，等待该命令返回；完整 GSM8K 可能需要较长时间
"${REMOTE}" run 17.111 \
  --cmd "NUM_CONCURRENT=8 bash scripts/lm_eval.sh" \
  --timeout 10800
```

如果客户端也必须后台运行：

```bash
"${REMOTE}" run 17.111 \
  --background \
  --task "lm-eval GSM8K test 全量" \
  --cmd "NUM_CONCURRENT=8 bash scripts/lm_eval.sh" \
  --timeout 10800
```

只在 5 分钟粒度检查 `remote jobs`/`remote logs`。Job 记录可能在进程退出后仍显示 `running`；遇到这种情况，
用目标 Job 的实际进程、端口和日志核验，再用 `remote stop <job-id>` 清理滞留记录，不能据此停止其他任务。

结果需要带回本地时，使用 `remote pull`，相对路径以远端 workspace 为基准，并保存到单独目录：

```bash
"${REMOTE}" pull 17.111 \
  .log/lm_eval_local_gsm8k_test_<timestamp>.log \
  .temp/lm_eval/local_gsm8k_test \
  --dest .temp/remote-17.111/lm_eval_<timestamp>
```

## 结果核验

完成后同时核对以下证据，不要只看终端最后一行：

1. 转换输出行数等于目标 split 行数；当前 test 应为 1319。
2. 日志出现 `Generating test split: 1319 examples`、请求进度 `1319/1319`，且 `limit: None`。
3. `results/qwen/results_*.json` 与 `samples_local_gsm8k_*.jsonl` 均生成，样本文件可用于定位解析错误。
4. 日志没有 `ERROR`、`Traceback` 或请求失败；服务端日志也没有对应异常。
5. 记录模型、服务参数、设备、task YAML、原始结果 JSON 和日志路径，保证结果可复现。

本配置在 17.111 上实际完成过 1319/1319 条，约耗时 39 分钟，参考输出为：

```text
strict-match exact_match:     0.8643 ± 0.0094
flexible-extract exact_match: 0.5345 ± 0.0137
```

这是当前模型/服务/采样配置的运行记录，不是所有模型的验收阈值。

## 换数据集时的判断

更换数据集通常不只是替换路径和 task 名：

- 若新数据仍是同样的 `prompt[0].content` + `reward_model.ground_truth` schema，可以复用转换脚本，修改
  `--input`、`--output`、`--task-yaml`、`--task-name` 和 `--split`。
- 若 schema 不同，先检查 Parquet/JSONL 的真实字段，再修改或新增明确的数据适配脚本；不要在 GSM8K
  helper 里添加不透明的多格式 fallback。
- 新 task 的 `dataset_path`、`data_files.<split>`、`<split>_split`、`doc_to_text`、
  `doc_to_target` 和 filter 必须互相匹配；`--include_path` 只是让 lm-eval 找到 YAML，
  不会自动理解数据字段。
- 多选题、分类、对话或多模态数据不能直接套用 GSM8K 的 `generate_until` 与数字 regex，需要单独确定
  prompt、target、generation 和 metric。

## 相关参考

- 具体故障症状、原因和修复顺序：阅读
  [references/troubleshooting.md](references/troubleshooting.md)。
- 在线吞吐 benchmark：使用 `.agents/skills/vllm-ascend-benchmark/SKILL.md`。
- 服务健康检查和启停：使用 `.agents/skills/vllm-ascend-serving/SKILL.md`。
- 远程机器、占用、同步、阻塞执行和产物拉回：使用
  `.agents/skills/remote-plugin/SKILL.md`。
