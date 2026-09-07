---
name: ais-bench
description: 在远程 NPU 机器上用 ais_bench 对已启动的 vLLM chat 服务做精度评测，涵盖服务确认、模型配置下发、后台执行、结果核验与产物拉回；不用于在线吞吐 benchmark，不用于 completions 接口的 lm-eval。
---

# ais-bench 精度评测

## 适用范围

当用户需要在远程 NPU 容器上，用 ais_bench 对已启动的 vLLM OpenAI-compatible 服务
（`/v1/chat/completions`）跑精度评测时使用本 skill，尤其是：

- 模型通过 `--served-model-name` 暴露，评测走 chat 接口（AIME/GSM8K/MATH/GPQA/CEval/MMLU 等）；
- 需要先验证最小数据集（AIME2024）链路，再跑全量大数据集；
- 需要把远端 `outputs/` 产物拉回本地 `.log/` 存档，保证结果可复现。

在线请求吞吐、TTFT、TPOT、ITL 等性能测试使用 `vllm-ascend-benchmark`；
`/v1/completions` 接口的本地 lm-eval 评测使用 `lm-eval-precision-benchmark`，
不要把三种测试混用。

## 核心边界

- 评测前先确认服务已启动且健康；不要在 ais_bench 客户端里隐式启动或停止服务。
- 一切远程操作走仓库内的 remote-plugin CLI，禁止裸 SSH；被测服务所在卡段不能再声明给评测任务，
  评测客户端本身通常不占 NPU 卡。
- 容器内一般配了 http_proxy，直连 `127.0.0.1` 必须绕过代理，否则 curl 返回 `000`、
  ais_bench 请求失败；凡是健康检查与评测命令，一律加 `--noproxy '*'` 或先关代理。
- 评测客户端优先以前台阻塞方式运行。确需后台运行时，必须带 `--background --task` 声明任务；
  按仓库规范每 5 分钟检查一次状态，不能高频手动轮询。
- 模型配置文件不要在远端手写 vim：本地维护模板，经 `remote run` 覆盖到远端 site-packages。
- 不把代理密码、token 或其他凭据写入 skill、task、脚本或日志。

## 标准配置

### 服务契约

以当前 17.119 上的 GLM-5.2-W4A8C8 配置为例，评测前必须记录：

```text
served model: qwen
endpoint:     http://127.0.0.1:8010/v1/chat/completions
weight:       /mnt/weight/GLM-5.2-W4A8C8-A3-0731
parallel:     DP2 + TP4 + EP, cards 0-7, max_model_len 16384
```

服务就绪后，至少检查（注意 `--noproxy`）：

```bash
curl --fail --noproxy '*' http://127.0.0.1:8010/health
curl --fail --noproxy '*' http://127.0.0.1:8010/v1/models
```

### 模型配置

先查 ais_bench 安装路径（看 `Location` 字段），模型配置目录为
`<site-packages>/ais_bench/benchmark/configs/models/vllm_api/`：

```bash
pip show ais_bench_benchmark
```

示例路径：`/usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench`。

本地模板（例：`.temp/ais_bench/vllm_api_general_chat.py`），用 `remote run` 覆盖到远端：

```bash
REMOTE=./.agents/skills/remote-plugin/remote

"${REMOTE}" run <alias> \
  --cmd "cp /vllm-ascend-workspace/.temp/ais_bench/vllm_api_general_chat.py \
    /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py"
```

模板内容（非流式，推理模型去思考过程；comments in English）：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",            # must be "service" for API testing
        type=VLLMCustomAPIChat,    # non-streaming; use VLLMCustomAPIChatStream for streaming
        abbr="vllm-api-general-chat",  # output dir name under predictions/results
        path="",
        model="qwen",              # must match --served-model-name of vllm serve
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="127.0.0.1",
        host_port=8010,
        url="",
        max_out_len=4096,
        batch_size=8,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0,         # accuracy test: greedy decoding
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),  # strip reasoning for reasoning models
    )
]
```

关键字段：`model` 必须与服务的 `--served-model-name` 一致；`abbr` 决定输出子目录名；
推理模型保留 `extract_non_reasoning_content`，否则答案抽取失败拉低分数。

数据集缺失时从权重机拷贝（9740 数据集源
`/mnt/l00517252/ais_bench/aisbench_auto_tools_prefix-master-master/datasets`）：

```bash
cp -r /mnt/l00517252/ais_bench/aisbench_auto_tools_prefix-master-master/datasets/* \
      /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/datasets/
```

### 执行命令

aime 题量最小，先跑它验证链路，再跑其他数据集：

```bash
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --dump-eval-details --max-num-workers 8
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt --merge-ds --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets mmlu_gen_0_shot_cot_chat_prompt --merge-ds --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets livecodebench_code_generate_lite_gen_0_shot_chat --merge-ds --dump-eval-details
```

### 远程执行方式

服务常驻后台（声明卡占用）；评测客户端优先前台阻塞，超时按数据集放大。
大数据集必须后台运行并落盘（`remote logs` 看不到 `| tee` 重定向后的 stdout，
所以同时 `tee` 到 `/tmp/aisbench-<dataset>.log`）：

```bash
REMOTE=./.agents/skills/remote-plugin/remote

"${REMOTE}" machines

"${REMOTE}" run 17.119 \
  --background \
  --task "aisbench-aime最小集" \
  --cmd "export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost; /usr/local/python3.11.10/bin/ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --dump-eval-details 2>&1 | tee /tmp/aisbench-aime.log" \
  --timeout 7200
```

只在 5 分钟粒度检查 `remote jobs`/`remote logs`。Job 记录可能在进程退出后仍显示
`running`（`| tee` 管道未退出）；遇到这种情况，用 summary 落盘核验，再用
`remote stop <job-id>` 清理滞留记录，不能据此停止其他任务。

停止任务：

```bash
"${REMOTE}" stop <job-id>
```

## 结果核验

完成后同时核对以下证据，不要只看终端最后一行：

1. 远端输出根目录为启动时 cwd 下的 `outputs/default/<YYYYMMDD_HHMMSS>/`，结构完整：
   `configs/`（生效配置快照）、`logs/infer/` + `logs/eval/`、
   `predictions/<abbr>/<dataset>.jsonl`（逐题预测）、
   `results/<abbr>/<dataset>.json`（`{"accuracy": ..., "details": {...}}`）、
   `summary/summary_<ts>.{txt,csv,md}`（汇总表，accuracy 即最终分数）。
1. 预测行数等于数据集题数（如 AIME2024 为 30）。
1. 日志出现 `Inference tasks completed` + `Evaluation tasks completed`，且 summary 三件套已落盘。
1. 日志没有 `ERROR`、`Traceback` 或请求失败；服务端日志对应时段也全是 `200 OK`。
1. 用 `remote pull` 把 summary + results + predictions + `/tmp` 日志拉回本地
   `.log/aisbench-<dataset>-<MMDD>/`，相对路径以远端 workspace 为基准：

```bash
"${REMOTE}" pull <alias> \
  outputs/default/<ts>/summary outputs/default/<ts>/results \
  outputs/default/<ts>/predictions /tmp/aisbench-<dataset>.log \
  --dest .log/aisbench-<dataset>-<MMDD>
```

1. 记录模型、服务参数（DP/TP/EP、卡段、端口、`max_model_len`）、模型配置模板、
   数据集名、原始结果 JSON 和日志路径，保证结果可复现。

本次在 17.119 上的实际运行记录（GLM-5.2-W4A8C8，DP2+TP4，
`max_out_len=4096, batch_size=8`）：AIME2024 推理约 55 分钟，产物
`outputs/default/20260907_134206/`，已拉回 `.log/aisbench-aime2024-0907/`：

```text
aime2024   85d0f2     accuracy  gen     13.33
```

这是当前模型/服务/采样配置的运行记录，不是所有模型的验收阈值。

## 换数据集时的判断

更换数据集通常不只是替换 `--datasets`：

- `--merge-ds` 只用于 ceval/mmlu/livecodebench 这类多子集数据集；aime/math500/gpqa/gsm8k 不要加。
- 大数据集（gsm8k 1319 题、mmlu、ceval）加 `--max-num-workers 8` 提并发；先确认服务
  `--max-num-seqs` 扛得住，超时异常时再降低。
- `max_out_len` / `batch_size` 按题型调：长推理题（aime）用 4096 + 小 batch；
  照搬大 batch 会导致超长输出截断或服务排队超时。
- 后台任务的 `--timeout` 按题量放大（AIME 30 题都要约 1 小时）；`tee` 日志文件名
  与 `--task` 中的数据集名保持一致，否则多轮次产物互相覆盖。
- 中途停掉的轮次会在远端留下 `defunct` 的 ais_bench 残留进程，可忽略；但重跑前先
  `remote stop` 旧 job，避免两个客户端同时打同一个服务端口。

后台跑在 curses 下会刷 `Can't set cursor ... running in background mode` 和
`Failed to read status file ... Attempting to clear and continuing`，均为无害告警。

## 相关参考

- 远程机器、占用、同步、阻塞执行和产物拉回：使用
  `.agents/skills/remote-plugin/SKILL.md`。
- 服务健康检查和启停：使用 `.agents/skills/vllm-ascend-serving/SKILL.md`。
- 在线吞吐 benchmark：使用 `.agents/skills/vllm-ascend-benchmark/SKILL.md`。
- `/v1/completions` 接口的 lm-eval 评测：使用
  `.agents/skills/lm-eval-precision-benchmark/SKILL.md`。
