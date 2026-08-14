#!/bin/bash

# vllm-ascend TP4 启动脚本
# 对应 launch.json 中的调试配置
WORKSPACE_DIR="/vllm-workspace"
VLLM_DIR="${WORKSPACE_DIR}/vllm"
ASCEND_DIR="${WORKSPACE_DIR}/vllm-ascend"

# 设置环境变量
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${ASCEND_DIR}:${VLLM_DIR}:${PYTHONPATH}"
export VLLM_VERSION=0.26.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400
export HCCL_BUFFSIZE=1024
export VLLM_LOGGING_LEVEL=INFO

# 启动 vllm serve
#Qwen3.5-35B-A3B
#DeepSeek-V4-Flash-w4a8
vllm serve \
    --model /mnt/weight/Qwen3.5-35B-A3B \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port 8010 \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --enable-expert-parallel \
    --profiler_config '{
    "profiler": "torch",
    "torch_profiler_dir": "/profile/profile-'"$(date +%d-%H-%M)"'",
    "torch_profiler_with_stack":true
    }' 2>&1 | tee .temp/debug-"$(date +%d-%H-%M)".log


# sp pass
# --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[2,4],\"pass_config\":{\"enable_sp\":true,\"sp_min_token_num\":1}}" \

# flashcomm
# --additional-config "{\"enable_flashcomm1\":true}" \

# profiling config
# --profiler_config '{
#     "profiler": "torch",
#     "torch_profiler_dir": "/vllm-profile-'"$(date +%d-%H-%M)"'",
#     "torch_profiler_with_stack":true
# }' \