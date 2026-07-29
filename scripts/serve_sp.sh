#!/usr/bin/env bash

export PYTHONPATH=/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend-workspace/vllm:${PYTHONPATH}
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CACHE_ROOT=/home/x50063850/vllm-ascend-workspace/.temp/bench

input_len=16384
enable_sp=true

mkdir -p /home/x50063850/vllm-ascend-workspace/.log "${VLLM_CACHE_ROOT}"

vllm serve \
  --model /home/weights/Qwen/Qwen3-30B-A3B \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8010 \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 25600 \
  --seed 0 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[2,4],\"pass_config\":{\"enable_sp\":${enable_sp},\"sp_min_token_num\":1024}}" \
  --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}" \
  2>&1 | tee /home/x50063850/vllm-ascend-workspace/.log/bench_sp_${enable_sp}_${input_len}.log
