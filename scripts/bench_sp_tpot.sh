#!/usr/bin/env bash

export PYTHONPATH=/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend-workspace/vllm:${PYTHONPATH}
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CACHE_ROOT=/home/x50063850/vllm-ascend-workspace/.temp/bench

input_len=16384
enable_sp=true

mkdir -p /home/x50063850/vllm-ascend-workspace/.log "${VLLM_CACHE_ROOT}"

vllm bench throughput \
  --model /home/weights/Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 25600 \
  --dataset-name random \
  --random-input-len ${input_len} \
  --enable-expert-parallel \
  --random-output-len 1 \
  --num-prompts 100 \
  --num-warmups 10 \
  --seed 0 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[2,4],\"pass_config\":{\"enable_sp\":${enable_sp},\"sp_min_token_num\":1024}}" \
  --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}" \
  --output-json /home/x50063850/vllm-ascend-workspace/.log/bench_sp_${enable_sp}_${input_len}.json \
  2>&1 | tee /home/x50063850/vllm-ascend-workspace/.log/bench_sp_${enable_sp}_${input_len}.log
