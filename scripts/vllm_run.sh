#!/bin/bash

# vllm-ascend TP4 启动脚本
# 对应 launch.json 中的调试配置
WORKSPACE_DIR="/vllm-ascend-workspace"
VLLM_DIR="${WORKSPACE_DIR}/vllm"
ASCEND_DIR="${WORKSPACE_DIR}/vllm-ascend"

# 设置环境变量
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONPATH="${ASCEND_DIR}:${VLLM_DIR}:${PYTHONPATH}"
export VLLM_VERSION=0.27.1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400
export HCCL_BUFFSIZE=1024
export VLLM_LOGGING_LEVEL=INFO

# 启动 vllm serve
#Qwen3.5-35B-A3B
#DeepSeek-V4-Flash-w4a8
#/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp

# sp pass
# --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[2,4],\"pass_config\":{\"enable_sp\":true,\"sp_min_token_num\":1}}" \

# disable npugraph ex
# --additional-config "{\"enable_npugraph_ex\":false}" \
# flashcomm
# --additional-config "{\"enable_flashcomm1\":true}" \

# --speculative-config '{"method": "dspark", "num_speculative_tokens": 7}' \
# --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
# "enforce_eager": true

# profiling config
# --profiler_config '{
#     "profiler": "torch",
#     "torch_profiler_dir": "/vllm-profile-'"$(date +%d-%H-%M)"'",
#     "torch_profiler_with_stack":true
# }' 
#
# 2&>1 | tee /vllm-profile-$(date +%d-%H-%M).log

# --additional-config '{
#             "enable_cpu_binding":true,
#             "multistream_overlap_shared_expert":true,
#             "enable_dsa_cp": true
#     }' \
mkdir -p /home/x50063850/vllm-logs

# profiler 默认开启（保持既有行为）；精度评测等场景用 ENABLE_PROFILER=0 关闭
export ENABLE_PROFILER=0
EXTRA_ARGS=()
if [[ "${ENABLE_PROFILER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--profiler_config '{"profiler": "torch", "torch_profiler_dir": "/home/x50063850/vllm-profile-'"$(date +%d-%H-%M)"'", "torch_profiler_with_stack": true}')
fi

vllm serve \
    --model /mnt/share/weight/DeepSeek-V4-Flash-0731 \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8010 \
    --max_model_len 25600 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --enable-expert-parallel \
    --additional-config '{
        "enable_flashcomm1":true,
        "enable_shared_expert_dp":true,
        "enable_dsa_cp":true,
        "enable_cpu_binding":true,
        "multistream_overlap_shared_expert":false
    }' \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 7}' \
    "${EXTRA_ARGS[@]}" \
#  2>&1 | tee /home/x50063850/vllm-logs/vllm-$(date +%d-%H-%M).log

# vllm serve \
#     --model /mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8-int4-new \
#     --served-model-name qwen \
#     --host 127.0.0.1 \
#     --port 8010 \
#     --max_model_len 25600 \
#     --data-parallel-size 1 \
#     --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.92 \
#     --max-num-seqs 16 \
#     --enable-expert-parallel \
#     --async-scheduling \
#     --block-size 16 \
#     --enable-prefix-caching \
#     --api_server_count 1 \
#     --tokenizer-mode deepseek_v4 \
#     --tool-call-parser deepseek_v4 \
#     --enable-auto-tool-choice \
#     --reasoning-parser deepseek_v4 \
#     --trust-remote-code \
#     --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
#     --additional-config '{
#         "enable_flashcomm1":true,
#         "enable_shared_expert_dp":true,
#         "enable_dsa_cp":true,
#         "enable_cpu_binding":true,
#         "multistream_overlap_shared_expert":true
#     }' \
#     --speculative-config '{"method": "dspark", "model": "/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8-int4-new", "num_speculative_tokens": 7, "enforce_eager": true}' \
#     "${EXTRA_ARGS[@]}" \
#  2>&1 | tee /home/x50063850/vllm-logs/vllm-$(date +%d-%H-%M).log