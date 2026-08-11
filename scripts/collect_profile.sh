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
analyse(\"/vllm-profile\")
"