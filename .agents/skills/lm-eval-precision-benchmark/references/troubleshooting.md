# lm-eval 精度评测排障

本文件只记录与本 skill 直接相关、已经在本地 Parquet + vLLM completions 流程中验证过的故障。
先确认服务、数据和 task 配置，再处理客户端；不要把客户端报错直接归因于模型代码。

## 症状与处理

### 找不到 task

- 症状：`Task local_gsm8k not found`。
- 原因：未传 `--include_path`，或 YAML 文件名与 task 名不一致。
- 处理：确认 `--include_path` 指向包含 `<task>.yaml` 的目录，且 `--tasks` 使用 YAML
  中的 `task` 名。

### 内置 task 尝试下载数据

- 症状：`gsm8k` 开始访问 Hugging Face 或下载数据。
- 原因：使用了 lm-eval 内置的 Hugging Face task。
- 处理：改用本地 JSON/Parquet 转换结果和自定义 `local_gsm8k` task。

### tokenizer 或 config 无法识别

- 症状：`deepseek_v4` config/tokenizer 无法识别。
- 原因：客户端按 Transformers 加载模型 tokenizer/config，当前容器版本不匹配。
- 处理：使用 `tokenizer_backend=none`，不要给 `local-completions` 强行指定该 tokenizer。

### 请求无响应且服务端没有 access 日志

- 原因：Python HTTP 客户端把 `127.0.0.1` 送进代理。
- 处理：同时设置大写 `NO_PROXY` 和小写 `no_proxy` 为
  `127.0.0.1,localhost`；手工 curl 加 `--noproxy '*'`。

### endpoint 或模型名错误

- 症状：出现 `404`、请求格式错误或 `model not found`。
- 原因：混用了 completions/chat completions endpoint，或把权重路径当成模型名。
- 处理：使用 `local-completions` 与 `/v1/completions`，模型名填写服务的
  `--served-model-name`，当前是 `qwen`。

### 误把 smoke 当成全量

- 症状：只跑了 1 条但以为是完整结果。
- 原因：残留 `--limit 1` 或 `LIMIT=1`。
- 处理：完整运行时清空 `LIMIT`，并在日志确认 `limit: None`。

### Parquet schema 不匹配

- 症状：转换脚本报 `no prompt messages`，或输出行数不对。
- 原因：输入不是当前 GSM8K processed schema，或者读错 split/文件。
- 处理：先检查真实字段和 `wc -l`，为新 schema 编写明确适配器；不要添加静默
  fallback。当前 test 预期 1319 行。

### 精度很低但生成文本看起来正确

- 原因：filter 可能没有抽到最终答案，或模型没有输出 `#### 数字`。
- 处理：查看 `samples_*.jsonl`，核对 strict/flexible 两个 filter 的命中内容，
  再决定是否修改 task。

### 两个 exact-match filter 差距很大

- 原因：strict 只接受 `####` 后的数字，flexible regex 可能拿到推理过程中的第一个数字。
- 处理：不要直接把 flexible 数值当作 canonical score；结合样本检查最终答案格式。

### 健康检查超时

- 可能原因：模型加载慢、端口未监听、NPU/显存被占用或服务端启动失败。
- 处理：先查服务 stdout/stderr、进程和 NPU 占用，不要只提高客户端 `timeout`。

### CANN 或版本环境错误

- 症状：远程命令报 `acl`/CANN 导入错误，或服务启动时报版本判断异常。
- 原因：`remote run` 没有加载 CANN 环境，或源码快照缺少可用 git version。
- 处理：直接执行前先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并显式
  设置与源码匹配的 `VLLM_VERSION`；本流程使用 `0.27.1`。

### 远程 Job 状态滞留

- 症状：评测 Job 显示 `running`，但进度不再变化，或 `remote logs` 看不到最新内容。
- 原因：Job 状态/日志收尾存在缓冲，实际进程可能已经退出。
- 处理：遵守 5 分钟轮询；检查实际 `lm_eval`/`vllm serve` 进程、端口和最终日志，
  确认后清理对应 Job。不要据此停止其他任务。

## Task filter 注意事项

当前生成的 task 同时保存两个 filter：

```yaml
filter_list:
  - name: strict-match
    filter:
      - function: regex
        regex_pattern: '#### (\-?[0-9\.\,]+)'
      - function: take_first
  - name: flexible-extract
    filter:
      - function: regex
        group_select: -1
        regex_pattern: '(-?[$0-9.,]{2,})|(-?[0-9]+)'
      - function: take_first
```

`strict-match` 适合模型按 GSM8K 习惯输出 `#### 18` 的情况；`flexible-extract` 只是辅助诊断，
因为推理文本可能包含多个数字。若要改变答案抽取规则，应先保存一批样本并人工核对，再修改 YAML，
不能为了得到更高分数直接选择某个 filter。

## 最小化复现顺序

遇到全量失败时按以下顺序缩小范围：

1. `/health` 和 `/v1/models`。
2. 用 `curl --noproxy '*'` 向 `/v1/completions` 发送一条最小请求。
3. `LIMIT=1 bash scripts/lm_eval.sh`，确认自定义 task 和输出格式。
4. 查看一条 `samples_*.jsonl`，确认 prompt、raw response、target 和 filter 结果。
5. 再恢复 `NUM_CONCURRENT=8` 跑完整 test；全量命令不使用 `--limit`。

每一步的日志分别留在 `.log/`，转换和 lm-eval 结果留在 `.temp/`；远程结果拉回本地时使用
`remote pull` 并放到带机器和时间戳的独立目录。
