# Qwen3.5 DP1 / TP4 SP 性能对照

## 测试配置

- 日期：2026-09-08；8K TTFT 专项测量已完成。
- 机器：90.90.81.179，NVIDIA H20 × 4（设备 4–7，每卡约 96GB）。
- 模型：`/home/weight/Qwen3.5-35B-A3B`，BF16，未量化。
- 源码：`sp-refactor-new`，commit `49ddb9f27e`；同一份源码比较 SP off / on。
- 并行：DP=1、TP=4，两组均启用 EP，backend=`allgather_reducescatter`。
- Runner：`VLLM_USE_V2_MODEL_RUNNER=0`，两组均为 eager，不启用 CUDA graph。
- 服务：max model len=16384、max batched tokens=8192、max seqs=32、
  GPU memory utilization=0.85，关闭 prefix cache 和 FlashInfer autotune。
- 客户端：本机 `vllm bench serve`，`/v1/completions`，random 数据集，seed=42，
  range ratio=0，ignore EOS，固定输出 128 tokens，请求速率不限。
- 负载：输入 1024 / 并发 1 / 32 请求；输入 1024 / 并发 16 / 64 请求；
  输入 8192 / 并发 16 / 64 请求。每项预热 4 请求，正式重复 2 轮。
- 客户端未显式指定 temperature，使用服务端默认值；本次为性能测量，不包含精度验收。

## 运行环境说明

容器现有 PyTorch 为 `2.13.0+cu130`，安装的 vLLM wheel 版本为
`0.17.2rc1.dev5155+gd62ad46a8`。使用独立的 `.temp/spbench-venv`
（system site packages）和 `PYTHONPATH` 加载本次源码，没有替换全局安装。

源码目录通过符号链接复用 wheel CUDA 扩展和 FlashAttention 配套文件，
补充缺失的 third_party 文件。wheel 缺少当前上游的 `vocab_parallel_embedding`
算子，因此用本次源码的 CUDA kernel 单独编译扩展，通过实验专用 sitecustomize 加载。
两组均使用这套相同环境，embedding kernel 已在 H20 上通过逐元素参考对照。
这不是整套 CUDA 扩展从当前 commit 完整重编的验证。

## 原始产物

- 正式运行目录：`.log/spbench/dp1-tp4-49ddb9f27e-matched-embedding/`。
- 测试驱动：`.temp/spbench/run.sh`；构建与结果检查脚本位于同目录。
- 早期环境不匹配和失败请求数据在 `.log/spbench/dp1-tp4-49ddb9f27e/`，
  不纳入性能对比。

## 结果

结果见下方 8K TTFT 专项；早期其他负载不纳入结论。

## 用户收敛后的测试范围

按最新要求停止上述全套负载，最终仅比较 8K 上下文 TTFT：
输入 8192 tokens、输出 1 token、并发 1、32 请求、预热 4 请求、重复 2 轮，
显式 temperature=0。服务参数和设备保持不变。
新产物目录为 `.log/spbench/ttft8k-dp1-tp4-49ddb9f27e/`，
驱动脚本为 `.temp/spbench/ttft8k.sh`。早期 1K/128 数据不纳入最终结论。

## 8K TTFT 实测结果

| SP | 轮次 | 成功 / 失败 | 平均 TTFT (ms) | P50 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | --- |
| off | 1 | 32 / 0 | 257.02 | 254.40 | 278.61 |
| off | 2 | 32 / 0 | 277.77 | 281.92 | 313.73 |
| on | 1 | 32 / 0 | 260.16 | 258.62 | 294.36 |
| on | 2 | 32 / 0 | 247.88 | 248.09 | 281.70 |

等样本量两轮平均：SP off **267.40 ms**，SP on **254.02 ms**，
TTFT 降低 **5.00%**（13.38 ms）。单轮存在波动，第一轮 SP on 略慢，
不能据这两轮短测认定有稳定的 5% 收益；本结论仅限上述 eager、并发 1 配置。
输出仅 1 token，TPOT / ITL 不适用于本次比较；未完成模型精度验证。

任务 `j-20260908-130443-01` 正常完成（exit 0）。
原始 JSON 和服务日志已回传至
`.log/spbench-results/ttft8k-dp1-tp4-49ddb9f27e/`。
服务已停止，4–7 号 GPU 显存已释放。

## SP-on 对话冒烟检查

沿用上述 Qwen3.5-35B-A3B BF16、DP1/TP4/EP、V1 eager 配置，
调用 `/v1/chat/completions`，temperature=0，seed=42。
任务 `j-20260908-131634-01` 正常完成，共 10 次请求均 HTTP 200。

| 用例 | 观察结果 |
| --- | --- |
| 中文番茄炒蛋做法 | 三步回答完整，正常 stop，无乱码 |
| 英文解释天空颜色 | 两句连贯说明，正常 stop |
| 17×23 | 正确返回 391 |
| 多轮记忆（两次请求） | 正确记住小林、杭州及偏好；推荐茶山徒步未严格避开用户“不喜欢爬山”的约束 |
| 长文本提取（实际 5049 prompt tokens） | 正确返回松鹤7391 |
| 3 路并发 | 光合作用解释、英文翻译及苹果算术均正常；苹果数正确为 6 |
| 默认模板 | 输出推理文本，512 token 用尽后 finish_reason=length，未完成最终回答 |

前 9 次请求显式传 `chat_template_kwargs={"enable_thinking": false}`，
均以 stop 正常结束；最后一次未设置该参数，且服务未配置 reasoning parser。
因此观察结论为：非 thinking 对话路径基本连贯，未观察到乱码或异常循环重复；
默认模板的最终回答尚未验证成功。没有 SP-off 同提示对照，不能把指令遵循问题或
默认模板截断归因于 SP，也不能据此宣布精度与基线等价。

原始请求和响应：`.log/spbench-results/chat-sp-on-49ddb9f27e/all.json`，
同目录保存分用例 JSON 与服务日志。测试后服务已停止。

## Dense SP CUDA graph 后续验证

Qwen3.5-4B，H20 TP4，BF16；不影响上面的 MoE eager 性能数值。
已删除动态 dense SP 禁用 CUDA graph 的逻辑，通过阈值捕获桶及 SP/TP 独立编译缓存
支持图执行。交替运行 1000、1001、17、1001、1000 token 输入，每步输出 4 token。

FULL 在当前 Qwen3.5 后端实际为 decode-only FULL，捕获 1 个 decode 图；
PIECEWISE 捕获 1、1000、1004、1024 四个桶。两者生成 token 均与各自无图基线一致。
FULL 与无编译无图基线的 top-5 logprob 最大差 0.125；
PIECEWISE 与有编译无图基线的所选 token logprob 最大差 0.0148，
共同 top-5 候选最大差 0.3751，top-5 候选集合有少量变化，未验证逐位一致。

3 项聚焦单测通过；真实模型验证驱动为 `.temp/spbench/dense_graph.py`，
产物已回传 `.log/spbench-results/dense-graph/`。未做 dense 性能 benchmark。
