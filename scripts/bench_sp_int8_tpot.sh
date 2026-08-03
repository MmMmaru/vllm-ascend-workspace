#!/usr/bin/env bash

# W8A8(INT8) + SP 长输入吞吐 benchmark（sp-int8 worktree）。
# 消融：enable_sp / fuse_gemm_comms 两组开关。

export PYTHONPATH=/home/x50063850/vllm-ascend-workspace/vllm-ascend/.worktrees/sp-int8:/home/x50063850/vllm-ascend-workspace/vllm:${PYTHONPATH}
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=DEBUG

input_len=16384
enable_sp=${ENABLE_SP:-true}
fuse_gemm_comms=${FUSE_GEMM_COMMS:-true}

export VLLM_CACHE_ROOT=/home/x50063850/vllm-ascend-workspace/.temp/bench_w8a8_sp_${enable_sp}_fuse_${fuse_gemm_comms}

mkdir -p /home/x50063850/vllm-ascend-workspace/.log "${VLLM_CACHE_ROOT}"

vllm bench throughput \
  --model /home/weights/vllm-ascend/Qwen3-30B-A3B-W8A8 \
  --quantization ascend \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --max-model-len 25600 \
  --dataset-name random \
  --random-input-len ${input_len} \
  --enable-expert-parallel \
  --random-output-len 1 \
  --num-prompts 100 \
  --num-warmups 10 \
  --seed 0 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[2,4],\"pass_config\":{\"enable_sp\":${enable_sp},\"sp_min_token_num\":1024,\"fuse_gemm_comms\":${fuse_gemm_comms}}}" \
  --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}" \
  --output-json /home/x50063850/vllm-ascend-workspace/.log/bench_w8a8_sp_${enable_sp}_fuse_${fuse_gemm_comms}_${input_len}.json \
  2>&1 | tee /home/x50063850/vllm-ascend-workspace/.log/bench_w8a8_sp_${enable_sp}_fuse_${fuse_gemm_comms}_${input_len}.log
