#!/bin/bash

# vllm-ascend TP4 启动脚本
# 对应 launch.json 中的调试配置
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLLM_DIR="${WORKSPACE_DIR}/vllm"
ASCEND_DIR="${WORKSPACE_DIR}/vllm-ascend"
cd "${VLLM_DIR}"

# 设置环境变量
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${ASCEND_DIR}:${VLLM_DIR}:${PYTHONPATH}"
export VLLM_VERSION=0.26.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400
export HCCL_IF_BASE_PORT=50000

# 启动 vllm serve

vllm serve \
    --model /mnt/a800_weight/Qwen3-30B-A3B \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port 8010 \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,4,8,16,32,64,128,256]}' \
    # --enable-expert-parallel \
