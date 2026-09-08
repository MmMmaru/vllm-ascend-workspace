---
name: vllm-ascend-benchmark
description: 在远程 NPU 机器上用 vllm bench serve 对已启动的 vLLM 服务做在线吞吐 benchmark，涵盖服务确认、脚本下发、后台执行、结果核验与产物拉回；不用于精度评测。
---

# vLLM Ascend Benchmark

在线吞吐 benchmark（`vllm bench serve` 打 `/v1/completions`），产出 TTFT/TPOT/ITL、
请求/token 吞吐、（speculative 时）acceptance rate。精度评测使用 `ais-bench`；
`/v1/completions` 接口的 lm-eval 评测使用 `lm-eval-precision-benchmark`，
不要把三种测试混用。

## 前置条件

- 服务已启动且健康，评测前必须记录服务契约（例：17.111 上的 DSpark 服务）：

```text
served model: qwen
endpoint:     http://127.0.0.1:8010/v1/completions
weight:       /mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8-int4-new
parallel:     DP2 + TP4 + EP, max_model_len 25600, max-num-seqs 16
```

```bash
curl --fail --noproxy '*' http://127.0.0.1:8010/health
curl --fail --noproxy '*' http://127.0.0.1:8010/v1/models
```

- 一切远程操作走仓库内的 remote-plugin CLI，禁止裸 SSH；benchmark 客户端本身
  不占 NPU 卡（不加 `--cards`）。
- 容器内一般配了 http_proxy，直连 `127.0.0.1` 必须绕过代理：凡是健康检查与
  benchmark 命令，一律加 `--noproxy '*'` 或先 `export NO_PROXY`。
- 确认无其他客户端在打同一端口（`ps` 查 `vllm bench` / `ais_bench` 进程），
  避免互相干扰。

## 标准命令（脚本化）

benchmark 命令固定在仓库脚本里，本地维护、经 `remote sync --paths` 同步后执行，
不要在远端手写。模板：`scripts/bench_perf.sh`：

```bash
#!/bin/bash
export PYTHONPATH=/vllm-ascend-workspace/vllm-ascend:/vllm-ascend-workspace/vllm:${PYTHONPATH:-}
export VLLM_VERSION=0.27.1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

RESULT_DIR=/vllm-ascend-workspace/.temp/benchmarks/<name>
mkdir -p "${RESULT_DIR}"

vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8010 \
  --endpoint /v1/completions \
  --served-model-name qwen \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 1024 \
  --num-prompts 100 \
  --num-warmups 20 \
  --max-concurrency 8 \
  --metric-percentiles 50,90,99 \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir "${RESULT_DIR}" \
  --result-filename <name>.json \
  2>&1 | tee /tmp/bench-<name>.log
```

要点：`random-input-len + random-output-len` 必须小于服务 `max_model_len`；
`--max-concurrency` 不超过服务 `--max-num-seqs`；固定 `--seed` 保证可复现；
`--save-result --save-detailed` 把原始 JSON 落到远端 result-dir；
`remote logs` 看不到 `| tee` 重定向后的 stdout，所以同时 `tee` 到 `/tmp` 日志。

## 远程执行与产物拉回

```bash
REMOTE=./.agents/skills/remote-plugin/remote

"${REMOTE}" sync <alias> --paths scripts/bench_perf.sh

"${REMOTE}" run <alias> \
  --background \
  --task "bench-<name>" \
  --cmd "export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost; \
    bash /vllm-ascend-workspace/scripts/bench_perf.sh" \
  --timeout 7200
```

- 按仓库规范每 2 分钟检查一次进度（看 `/tmp/bench-<name>.log` 的进度条），
  不能高频手动轮询。
- Job 记录可能在进程退出后仍显示 `running`（`| tee` 管道未退出）；以 result
  JSON 落盘为准，再用 `remote stop <job-id>` 清理滞留记录。
- 用 `remote pull` 把 result JSON 和 `/tmp` 日志拉回本地 `.log/bench-<name>-<MMDD>/`：

```bash
"${REMOTE}" pull <alias> \
  .temp/benchmarks/<name> /tmp/bench-<name>.log \
  --dest .log/bench-<name>-<MMDD>
```

## 关注指标

- output throughput、request throughput；
- TTFT、TPOT、ITL 的 mean、median、p90、p99；
- acceptance rate（使用 speculative decoding 时）；
- 首次运行与稳定运行的差异，以及异常请求数。

对比 baseline/改动版本时，每个版本使用相同硬件、模型、环境变量、输入输出长度、
并发度和 warmup 次数。返回结果必须保留配置和原始 JSON，不能只比较单个均值。

## 实际运行记录

2026-09-08 在 17.111 上（DeepSeek-V4-Flash-DSpark-w4a8-int4-new，
DP2+TP4+EP，`max_model_len 25600`，random 4096-in/1024-out，并发 8，
100 prompts + 20 warmups）：产物 `.temp/benchmarks/dspark-w4a8-int4-0908/`，
已拉回 `.log/bench-dspark-0908/`。注意该服务同时被编译任务等共享机器，
数字只反映当时整机负载下的表现。

```text
Successful requests:          100
Failed requests:              0
Benchmark duration (s):       750.38
Request throughput (req/s):   0.13
Output token throughput:      136.46 tok/s (peak 168.00)
Total token throughput:       682.32 tok/s
Mean TTFT: 1397.35 ms (p50 1226.30 / p90 1793.61 / p99 3037.19)
Mean TPOT: 55.55 ms (p50 55.75 / p90 58.81 / p99 60.20)
Mean ITL:  56.94 ms (p99 307.74)
```

## 相关参考

- 远程机器、占用、同步、阻塞执行和产物拉回：使用
  `.agents/skills/remote-plugin/SKILL.md`。
- 服务健康检查和启停：使用 `.agents/skills/vllm-ascend-serving/SKILL.md`。
- chat 接口精度评测：使用 `.agents/skills/ais-bench/SKILL.md`。
