---
name: vllm-ascend-benchmark
description: 整理 vLLM Ascend 在线服务 benchmark 参数、指标和对比方法；不直接执行远程操作。
---

# vLLM Ascend Benchmark

本 skill 只保留性能测试领域知识。远端命令由用户在目标 NPU 容器执行，
本地负责生成命令、读取 JSON 和比较结果。

## 测试命令

```bash
vllm bench serve \
  --model <model-path> \
  --base-url http://127.0.0.1:<port> \
  --num-prompts <count> \
  --max-concurrency <concurrency> \
  --input-len <input-len> \
  --output-len <output-len>
```

多次测试时固定模型、服务参数、设备和请求规模；先执行 warmup，再记录正式运行。
服务应在 benchmark 前完成 `/health` 与 `/v1/models` 检查，结束后由用户停止服务。

## 关注指标

- output throughput、request throughput；
- TTFT、TPOT、ITL 的 mean、median、p90、p99；
- acceptance rate（使用 speculative decoding 时）；
- 首次运行与稳定运行的差异，以及异常请求数。

对比 baseline/改动版本时，每个版本使用相同硬件、模型、环境变量、输入输出长度、
并发度和 warmup 次数。返回结果必须保留配置和原始 JSON，不能只比较单个均值。
