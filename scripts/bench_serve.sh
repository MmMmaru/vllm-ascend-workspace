
export PYTHONPATH=/vllm-ascend-workspace/vllm-ascend:/vllm-ascend-workspace/vllm:${PYTHONPATH:-}
export VLLM_VERSION=0.27.1
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
  --seed 0

  