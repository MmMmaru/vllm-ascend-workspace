#!/bin/bash
# DeepSeek-V4 MTP / DSpark performance & acceptance test (17.119, DP2TP4, 8 cards).
# 4 rounds, each vs its own weight's baseline:
#   1. MTP weight baseline (no spec)      MODEL=/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp
#   2. MTP weight + mtp K=1               same weight + --speculative-config mtp
#   3. DSpark weight baseline (no spec)   MODEL=/mnt/weight/DeepSeek-V4-Flash-0731-DSpark-W8A8-FlexSmooth-msModelSlim-20260901
#   4. DSpark weight + dspark K=5         same weight + --speculative-config dspark (block_size=5 in config.json)
# Bench: random dataset, input 1024 / output 256 / concurrency 32 / 200 prompts + 5 warmups.
set -u

WORKSPACE_DIR="${WORKSPACE_DIR:-/vllm-ascend-workspace}"
VLLM_DIR="${WORKSPACE_DIR}/vllm"
ASCEND_DIR="${WORKSPACE_DIR}/vllm-ascend"
OUTDIR="${OUTDIR:-${WORKSPACE_DIR}/outputs/dsv4-spec-bench-$(date +%d-%H-%M)}"
mkdir -p "${OUTDIR}"

CARDS="${CARDS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8010}"
SERVED_NAME=dsv4
MTP_WEIGHT="${MTP_WEIGHT:-/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp}"
DSPARK_WEIGHT="${DSPARK_WEIGHT:-/mnt/weight/DeepSeek-V4-Flash-0731-DSpark-W8A8-FlexSmooth-msModelSlim-20260901}"

# Fail fast if any selected card has <50GB free (leaked/occupied HBM).
echo "VRAM precheck on cards: ${CARDS}"
export ASCEND_RT_VISIBLE_DEVICES="${CARDS}"
python3 - <<'EOF'
import sys
try:
    import torch_npu
    n = torch_npu.npu.device_count()
    free = {}
    for i in range(n):
        f, _ = torch_npu.npu.mem_get_info(f"npu:{i}")
        free[i] = f / 1024**3
except Exception as e:
    print(f"precheck skipped ({e})")
    sys.exit(0)
print("free GiB per visible device:", {k: round(v, 1) for k, v in free.items()})
bad = {k: round(v, 1) for k, v in free.items() if v < 50}
if bad:
    print(f"ABORT: cards with <50GB free: {bad}")
    sys.exit(1)
EOF
[ $? -eq 0 ] || { echo "VRAM precheck failed, aborting."; exit 1; }

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${ASCEND_DIR}:${VLLM_DIR}:${PYTHONPATH:-}"
export VLLM_VERSION=0.27.1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400
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
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'
    --quantization ascend
    --block-size 32
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
)
MTP_ADDITIONAL='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
DSPARK_ADDITIONAL='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}'

wait_healthy() {
    local logfile=$1
    for i in $(seq 1 180); do
        if curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
            echo "health OK after ~$((i * 10))s"
            return 0
        fi
        if ! kill -0 ${SERVE_PID} 2>/dev/null; then
            echo "serve process died, see ${logfile}"
            return 1
        fi
        sleep 10
    done
    echo "health check timeout, see ${logfile}"
    return 1
}

run_round() {
    local name=$1      # e.g. mtp-baseline
    local model=$2
    local additional=$3
    shift 3            # remaining args = extra speculative args (maybe empty)
    local logfile="${OUTDIR}/${name}.serve.log"
    local benchlog="${OUTDIR}/${name}.bench.log"

    echo "===== ROUND ${name} =====" | tee -a "${OUTDIR}/progress.log"
    echo "model=${model} extra=[$*]" | tee -a "${OUTDIR}/progress.log"

    vllm serve --model "${model}" "${COMMON_ARGS[@]}" \
        --additional-config "${additional}" "$@" \
        >"${logfile}" 2>&1 &
    SERVE_PID=$!
    echo "${name} serve pid=${SERVE_PID}" | tee -a "${OUTDIR}/progress.log"

    if ! wait_healthy "${logfile}"; then
        kill ${SERVE_PID} 2>/dev/null; wait ${SERVE_PID} 2>/dev/null
        echo "${name}: SERVE FAILED" | tee -a "${OUTDIR}/progress.log"
        return 1
    fi
    curl -sf http://127.0.0.1:${PORT}/v1/models | head -c 500; echo

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
        2>&1 | tee "${benchlog}"
    echo "${name}: bench exit=${PIPESTATUS[0]}" | tee -a "${OUTDIR}/progress.log"

    echo "--- ${name} acceptance lines ---" | tee -a "${OUTDIR}/progress.log"
    grep -a -i "acceptance" "${logfile}" | tail -20 | tee -a "${OUTDIR}/progress.log"

    kill ${SERVE_PID} 2>/dev/null; wait ${SERVE_PID} 2>/dev/null
    sleep 30
    echo "${name}: DONE" | tee -a "${OUTDIR}/progress.log"
}

run_round mtp-baseline "${MTP_WEIGHT}" "${MTP_ADDITIONAL}"
run_round mtp-enabled "${MTP_WEIGHT}" "${MTP_ADDITIONAL}" \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}'
run_round dspark-baseline "${DSPARK_WEIGHT}" "${DSPARK_ADDITIONAL}"
run_round dspark-enabled "${DSPARK_WEIGHT}" "${DSPARK_ADDITIONAL}" \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": true}'

echo "ALL ROUNDS DONE, results in ${OUTDIR}"
ls -lh "${OUTDIR}"
