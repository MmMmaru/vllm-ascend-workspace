#!/bin/bash
# Online throughput benchmark against local vLLM OpenAI-compatible service.
# Service under test: DeepSeek-V4-Flash-DSpark-w4a8-int4-new, DP2+TP4+EP, port 8010.

export PYTHONPATH=/vllm-ascend-workspace/vllm-ascend:/vllm-ascend-workspace/vllm:${PYTHONPATH:-}
export VLLM_VERSION=0.27.1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

RESULT_DIR=/vllm-ascend-workspace/.temp/benchmarks/dspark-w4a8-int4-0908
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
  --result-filename dspark-w4a8-int4_random4096-1024_c8.json \
  2>&1 | tee /tmp/bench-perf-0908.log
