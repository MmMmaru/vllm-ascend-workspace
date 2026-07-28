
cd /home/x50063850/vllm-workspace/vllm-ascend

export PYTHONPATH=/home/x50063850/vllm-ascend:/home/x50063850/vllm:${PYTHONPATH}
export ASCEND_RT_VISIBLE_DEVICES=6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
# export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CACHE_ROOT=/home/x50063850/vllm-workspace/.temp/serve_sp

vllm serve /home/weights/Qwen/Qwen3-30B-A3B \
  --served-model-name qwen3-vl \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[2,4],\"pass_config\":{\"enable_sp\":true,\"sp_min_token_num\":1000}}" \
  --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}"
