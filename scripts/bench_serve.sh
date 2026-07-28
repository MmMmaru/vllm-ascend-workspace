docker exec xrs_vllm_main bash -lc '

export PYTHONPATH=/home/x50063850/vllm-ascend:/home/x50063850/vllm

vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8010 \
  --endpoint /v1/chat/completions \
  --model qwen \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 4 \
  --num-prompts 100 \
  --num-warmups 10 \
  --request-rate inf \
  --percentile-metrics ttft \
  --metric-percentiles 50,90,99 \
  --save-result \
  --save-detailed \
  --result-dir /home/x50063850/vllm-workspace/.log \
  --result-filename ttft_sp.json \
  --seed 0
'