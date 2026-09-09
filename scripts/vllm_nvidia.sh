#!/bin/bash

cd /vllm-ascend-workspace/vllm || exit 1
export CUDA_VISIBLE_DEVICES=0,1,3,4

exec python3 -m vllm.entrypoints.cli.main serve \
    /home/weight/Qwen3.5-35B-A3B \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port 8010 \
    --tensor-parallel-size 4 \
    --enable-sequence-parallel-moe \
    --dtype float16 \
    --max-model-len 1024 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --no-enable-flashinfer-autotune

# export PYTHONPATH=/vllm-ascend-workspace/vllm
# vllm serve \
#     /home/weight/Qwen3.5-35B-A3B \
#     --served-model-name qwen35 \
#     --host 0.0.0.0 \
#     --port 18010 \
#     --tensor-parallel-size 4 \
#     --enable-sequence-parallel-moe \
#     --dtype bfloat16 \
#     --max-model-len 1024 \
#     --max-num-seqs 4 \
#     --gpu-memory-utilization 0.85 \
#     --enforce-eager \
#     --no-enable-flashinfer-autotune