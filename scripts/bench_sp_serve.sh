#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLLM_DIR="${WORKSPACE_DIR}/vllm"

MODEL="${TEST_MODEL:-Qwen/Qwen3-30B-A3B}"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5}"
TP_SIZE="${TEST_TP_SIZE:-4}"
PORT="${TEST_PORT:-8010}"
INPUT_LEN="${TEST_INPUT_LEN:-16384}"
OUTPUT_LEN="${TEST_OUTPUT_LEN:-4}"
NUM_PROMPTS="${TEST_NUM_PROMPTS:-100}"
NUM_WARMUPS="${TEST_NUM_WARMUPS:-10}"
MAX_MODEL_LEN="${TEST_MAX_MODEL_LEN:-40960}"
STARTUP_TIMEOUT="${TEST_STARTUP_TIMEOUT:-600}"

LOG_DIR="${WORKSPACE_DIR}/.log"
CACHE_DIR="${WORKSPACE_DIR}/.temp/bench_sp_serve"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SERVE_LOG="${LOG_DIR}/bench_sp_serve_${RUN_ID}.log"
BENCH_LOG="${LOG_DIR}/bench_sp_bench_${RUN_ID}.log"
RESULT_FILE="${LOG_DIR}/bench_sp_serve_${RUN_ID}.json"
BASE_URL="http://127.0.0.1:${PORT}"
SERVE_PID=""

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VLLM_CACHE_ROOT="${CACHE_DIR}"
export PYTHONPATH="${WORKSPACE_DIR}/vllm-ascend:${VLLM_DIR}:${PYTHONPATH:-}"

COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"pass_config":{"enable_sp":true,"sp_min_token_num":1}}'
ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":false}}'

stop_server() {
    if [[ -z "${SERVE_PID}" ]] || ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        return
    fi

    kill "${SERVE_PID}" 2>/dev/null || true
    for _ in {1..30}; do
        if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
            return
        fi
        sleep 1
    done
    kill -KILL "${SERVE_PID}" 2>/dev/null || true
}

trap stop_server EXIT INT TERM

if curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "服务端口已被占用: ${BASE_URL}" >&2
    exit 1
fi

cd "${VLLM_DIR}"
vllm serve "${MODEL}" \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --seed 0 \
    --trust-remote-code \
    --compilation-config "${COMPILATION_CONFIG}" \
    --additional-config "${ADDITIONAL_CONFIG}" \
    >"${SERVE_LOG}" 2>&1 &
SERVE_PID=$!

ready=0
for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
    if curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        tail -80 "${SERVE_LOG}" >&2
        exit 1
    fi
    sleep 1
done

if [[ "${ready}" != "1" ]]; then
    echo "服务启动超时，日志: ${SERVE_LOG}" >&2
    tail -80 "${SERVE_LOG}" >&2
    exit 1
fi

vllm bench serve \
    --backend openai \
    --base-url "${BASE_URL}" \
    --endpoint /v1/completions \
    --model qwen \
    --tokenizer "${MODEL}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --num-warmups "${NUM_WARMUPS}" \
    --request-rate inf \
    --temperature 0 \
    --percentile-metrics ttft \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "${LOG_DIR}" \
    --result-filename "$(basename "${RESULT_FILE}")" \
    --seed 0 \
    2>&1 | tee "${BENCH_LOG}"

echo "结果文件: ${RESULT_FILE}"
cat "${RESULT_FILE}"
