#!/bin/bash

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

/usr/local/python3.11.10/bin/ais_bench \
  --models vllm_api_general_chat \
  --datasets gsm8k_gen_0_shot_cot_chat_prompt \
  --dump-eval-details \
  --max-num-workers 8 \
  --num-prompts 200 \
  2>&1 | tee /tmp/aisbench-gsm8k-200.log