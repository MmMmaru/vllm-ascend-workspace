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
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export PYTHONPATH="${ASCEND_DIR}:${VLLM_DIR}:${PYTHONPATH}"
# vllm is built with the local "+empty" suffix in the container; vllm-ascend
# uses this override to select the exact v0.24.0 compatibility lane.
export VLLM_VERSION=0.24.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400

# 启动 vllm serve

vllm serve /home/weights/Qwen/Qwen3-30B-A3B \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port 8010 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --compilation-config '{"pass_config": {"enable_sp": true}}' \
    --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}" 
    # --enable-expert-parallel \
    # --additional-config '{"enable_flashcomm1": true}' \