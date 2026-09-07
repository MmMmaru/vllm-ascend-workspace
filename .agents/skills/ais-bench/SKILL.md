# ais-bench 精度评测工作流

本 skill 负责在远程 NPU 机器上经 `remote` CLI 跑 ais_bench 精度测试：
查占用 → 确认服务 → 下发模型配置 → 后台执行 → 拉回结果。
一切远程操作走 `remote` CLI，不裸用 ssh（见 remote-plugin skill）。

## 0. 前置检查

1. `remote machines` 确认目标机无卡冲突；被测服务所在卡段（`ASCEND_RT_VISIBLE_DEVICES`）不能再声明给评测任务。
2. 确认被测服务存活。注意容器内一般配了 http_proxy，直连 `127.0.0.1` 必须先关代理，否则 curl 返回 `000`：

```sh
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
curl -s -o /dev/null -w 'health:%{http_code}\n' --max-time 10 http://127.0.0.1:<port>/health
curl -s --max-time 15 http://127.0.0.1:<port>/v1/models
```

记录模型路径、`served-model-name`、端口、DP/TP/EP 与卡段，后续写报告用。

## 一、精度测试

### 1. 查找 ais_bench 安装路径

```sh
pip show ais_bench_benchmark  # 看 Location 字段
```

示例：`/usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench`。
模型配置目录为 `<site-packages>/ais_bench/benchmark/configs/models/vllm_api/`。

### 2. 下发模型配置

本地维护配置模板（例：`.temp/ais_bench/vllm_api_general_chat.py`），经 `remote run` 覆盖到远端，
不要在远端手写 vim：

```sh
cp /vllm-ascend-workspace/.temp/ais_bench/vllm_api_general_chat.py \
   /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```

模板（非流式，推理模型去思考过程；字段注释为英文）：

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

数据集缺失时从权重机拷贝（9740 数据集源 `/mnt/l00517252/ais_bench/aisbench_auto_tools_prefix-master-master/datasets`）：

```sh
cp -r /mnt/l00517252/ais_bench/aisbench_auto_tools_prefix-master-master/datasets/* \
      /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/datasets/
```

### 3. 后台执行测试

必须带 `--background --task` 声明任务；命令内先关代理，`tee` 落盘到 `/tmp/aisbench-<dataset>.log`
（`remote logs` 看不到 tee 重定向后的 stdout）：

```sh
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
/usr/local/python3.11.10/bin/ais_bench --models vllm_api_general_chat \
  --datasets aime2024_gen_0_shot_chat_prompt --dump-eval-details 2>&1 | tee /tmp/aisbench-aime.log
```

数据集命令（aime 题量最小，先跑它验证链路）：

```sh
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --dump-eval-details --max-num-workers 8
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt --merge-ds --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets mmlu_gen_0_shot_cot_chat_prompt --merge-ds --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets livecodebench_code_generate_lite_gen_0_shot_chat --merge-ds --dump-eval-details
```

耗时参考（GLM-5.2-W4A8C8，DP2+TP4，`max_out_len=4096, batch_size=8`）：
AIME2024 共 30 题，推理约 55 分钟；GSM8K 1319 题只会更久，按 5 分钟粒度轮询，
不要 busy-poll。

### 4. 看结果与拉回本地

远端输出根目录为启动时的 cwd 下的 `outputs/default/<YYYYMMDD_HHMMSS>/`：

```text
outputs/default/20260907_134206/
  configs/      本次实际生效的配置快照
  logs/infer/   推理日志，logs/eval/ 评测日志
  predictions/<abbr>/<dataset>.jsonl   逐题预测（含 --dump-eval-details 详情）
  results/<abbr>/<dataset>.json        {"accuracy": ..., "details": {...}, "type": ...}
  summary/summary_<ts>.{txt,csv,md}    汇总表，accuracy 即最终分数
```

先读 `summary_*.md` 拿分数，再用 `remote pull` 拉回本地 `.log/aisbench-<dataset>-<MMDD>/`
（summary + results + predictions + `/tmp/aisbench-<dataset>.log`）：

```sh
python3 .agents/skills/remote-plugin/remote pull <alias> \
  outputs/default/<ts>/summary outputs/default/<ts>/results \
  outputs/default/<ts>/predictions /tmp/aisbench-<dataset>.log \
  --dest .log/aisbench-<dataset>-<MMDD>
```

### 5. 停止任务

```sh
python3 .agents/skills/remote-plugin/remote stop <job-id>
```

## 二、已知坑点

* 容器 http_proxy 会劫持 `127.0.0.1`：凡是 curl 健康检查、ais_bench 本体命令，
  一律先 `export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost`。
* 后台跑在 curses 下会刷 `Can't set cursor ... running in background mode` 和
  `Failed to read status file ... Attempting to clear and continuing`，均为无害告警。
* 评测跑完后 `remote jobs` 可能仍显示 `running`（`| tee` 管道未退出），以
  `summary/summary_*.md` 落盘为准，确认后手动 `remote stop`。
* 评测期间不要重启被测服务；中途停掉的轮次会在远端留下 `defunct` 的 ais_bench 残留进程，可忽略。

## 三、性能测试

`ais_bench --help` 查看性能参数；也可参考华为内部 wiki
[基于aisbench性能测试工具](https://wiki.huawei.com/domains/143914/wiki/137614/WIKI202511048896614)。
