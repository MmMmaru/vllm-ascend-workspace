#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
cd -- "${WORKSPACE_DIR}"

SPLIT="${SPLIT:-test}"
DATASET_DIR="${DATASET_DIR:-/mnt/share/datasets/gsm8k}"
DATASET_FILE="${DATASET_FILE:-${DATASET_DIR}/${SPLIT}.parquet}"
TASK_NAME="${TASK_NAME:-local_gsm8k}"
TASK_DIR="${TASK_DIR:-${WORKSPACE_DIR}/.temp/lm_eval/${TASK_NAME}_${SPLIT}}"

MODEL="${MODEL:-qwen}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8010/v1/completions}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
MAX_RETRIES="${MAX_RETRIES:-1}"
TIMEOUT="${TIMEOUT:-300}"
NUM_CONCURRENT="${NUM_CONCURRENT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/.log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/lm_eval_${TASK_NAME}_${SPLIT}_$(date +%Y%m%d_%H%M%S).log}"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

mkdir -p "${TASK_DIR}" "${LOG_DIR}"

python "${SCRIPT_DIR}/prepare_lm_eval_gsm8k.py" \
  --input "${DATASET_FILE}" \
  --output "${TASK_DIR}/${SPLIT}.jsonl" \
  --task-yaml "${TASK_DIR}/${TASK_NAME}.yaml" \
  --task-name "${TASK_NAME}" \
  --split "${SPLIT}"

MODEL_ARGS="model=${MODEL},base_url=${BASE_URL},tokenizer_backend=none,max_gen_toks=${MAX_GEN_TOKS},max_retries=${MAX_RETRIES},timeout=${TIMEOUT},num_concurrent=${NUM_CONCURRENT}"

LM_EVAL_ARGS=(
  --model local-completions
  --model_args "${MODEL_ARGS}"
  --include_path "${TASK_DIR}"
  --tasks "${TASK_NAME}"
  --batch_size "${BATCH_SIZE}"
  --log_samples
  --output_path "${TASK_DIR}/results"
)

if [[ -n "${LIMIT}" ]]; then
  LM_EVAL_ARGS+=(--limit "${LIMIT}")
fi

"${LM_EVAL_BIN}" "${LM_EVAL_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"