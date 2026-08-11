# docker exec xrs_vllm_main bash -lc '
# cd /home/x50063850/vllm-workspace/vllm-ascend

# export PYTHONPATH=/usr/local/Ascend/cann-9.0.1/python/site-packages:/home/x50063850/vllm-workspace/vllm-ascend:/home/x50063850/vllm-workspace/vllm
# export ASCEND_RT_VISIBLE_DEVICES=6,7
# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_COMPILE_CACHE_ROOT=./temp/sp_debug

# pytest -q -s \
#   tests/e2e/pull_request/two_card/test_sequence_parallel_linear.py \
#   2>&1 | tee ../.log/sp_debug.log
# '

docker exec xrs_vllm_main bash -lc '
cd /home/x50063850/vllm-workspace/vllm-ascend

export PYTHONPATH=/usr/local/Ascend/cann-9.0.1/python/site-packages:/home/x50063850/vllm-workspace/vllm-ascend:/home/x50063850/vllm-workspace/vllm
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_COMPILE_CACHE_ROOT=./temp/sp_debug
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export HF_HOME=/home/weights

pytest -q -s \
  tests/e2e/pull_request/two_card/test_sp_pass.py::test_qwen3_vl_sp_tp2 \
  2>&1 | tee ../.log/sp_pass_debug_origin.log
'
