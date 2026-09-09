#!/bin/bash
# DSpark eager pair: both rounds use --enforce-eager (skip buggy graph capture
# on this snapshot) with identical dsa_cp additional-config. Isolates DSpark
# draft benefit: baseline-eager vs dspark-enabled-eager.
set -u

WORKSPACE_DIR="${WORKSPACE_DIR:-/vllm-workspace}"
VLLM_DIR="${WORKSPACE_DIR}/vllm"
ASCEND_DIR="${WORKSPACE_DIR}/vllm-ascend"
OUTDIR="${OUTDIR:-${WORKSPACE_DIR}/outputs/dsv4-dspark-eager-$(date +%d-%H-%M)}"
mkdir -p "${OUTDIR}"

CARDS="${CARDS:-2,3,4,5,6,7,8,9}"
PORT="${PORT:-8020}"
SERVED_NAME=dsv4
DSPARK_WEIGHT="${DSPARK_WEIGHT:-/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${ASCEND_DIR}:${VLLM_DIR}:${PYTHONPATH:-}"
export VLLM_VERSION=0.27.1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HCCL_BUFFSIZE=1024
export VLLM_LOGGING_LEVEL=INFO
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES="${CARDS}"

COMMON_ARGS=(
    --host 127.0.0.1 --port ${PORT}
    --max-model-len 32768
    --max-num-batched-tokens 10240
    --served-model-name ${SERVED_NAME}
    --gpu-memory-utilization 0.9
    --max-num-seqs 64
    --data-parallel-size 2
    --tensor-parallel-size 4
    --enable-expert-parallel
    --enforce-eager
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'
    --quantization ascend
    --block-size 32
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}'
)

run_round() {
    local name=$1
    shift
    echo "===== ROUND ${name} =====" | tee -a "${OUTDIR}/progress.log"
    vllm serve --model "${DSPARK_WEIGHT}" "${COMMON_ARGS[@]}" "$@" \
        >"${OUTDIR}/${name}.serve.log" 2>&1 &
    SERVE_PID=$!
    for i in $(seq 1 240); do
        if curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
            echo "health OK after ~$((i * 10))s" | tee -a "${OUTDIR}/progress.log"
            break
        fi
        if ! kill -0 ${SERVE_PID} 2>/dev/null; then
            echo "${name}: SERVE FAILED" | tee -a "${OUTDIR}/progress.log"
            return 1
        fi
        sleep 10
    done
    curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1 || { echo "${name}: HEALTH TIMEOUT" | tee -a "${OUTDIR}/progress.log"; return 1; }
    vllm bench serve \
        --backend openai \
        --base-url http://127.0.0.1:${PORT} \
        --endpoint /v1/completions \
        --served-model-name ${SERVED_NAME} \
        --dataset-name random \
        --random-input-len 1024 \
        --random-output-len 256 \
        --num-prompts 200 \
        --num-warmups 5 \
        --max-concurrency 32 \
        --metric-percentiles 50,90,99 \
        --seed 0 \
        2>&1 | tee "${OUTDIR}/${name}.bench.log"
    echo "${name}: bench exit=${PIPESTATUS[0]}" | tee -a "${OUTDIR}/progress.log"
    grep -a -i "acceptance" "${OUTDIR}/${name}.bench.log" | head -8 | tee -a "${OUTDIR}/progress.log"
    kill ${SERVE_PID} 2>/dev/null; wait ${SERVE_PID} 2>/dev/null
    sleep 30
    echo "${name}: DONE" | tee -a "${OUTDIR}/progress.log"
}

run_round dspark-baseline-eager
run_round dspark-enabled-eager \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": true}'
echo "ALL ROUNDS DONE"
