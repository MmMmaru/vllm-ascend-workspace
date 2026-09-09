curl -X POST http://127.0.0.1:8010/start_profile

# 先生成内容
CONTENT=$(printf 'hello%.0s' {1..100})

# 然后使用
curl http://127.0.0.1:8010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "'"$CONTENT"'"}
        ],
        "max_tokens": 10,
        "temperature": 0
    }'

curl -X POST http://127.0.0.1:8010/stop_profile

python3 -c "
from torch_npu.profiler.profiler import analyse
analyse(\"/home/x50063850/vllm-profile-26-17-17/dp0_pp0_tp0_dcp0_ep0_rank0_450730_20260826172308410_ascend_pt\")
"