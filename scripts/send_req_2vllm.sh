curl http://127.0.0.1:8010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "Please introduce yourself"}
        ],
        "max_tokens": 1000,
        "temperature": 0
    }'