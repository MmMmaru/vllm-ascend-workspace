
export PYTHONPATH=/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}
export VLLM_VERSION=0.26.0
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8010 \
  --endpoint /v1/completions \
  --served-model-name qwen \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 2048 \
  --num-prompts 50 \
  --num-warmups 5 \
  --max-concurrency 32 \
  --metric-percentiles 50,90,99 \
  --seed 0