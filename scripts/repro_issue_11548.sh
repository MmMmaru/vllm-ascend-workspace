#!/usr/bin/env bash

# Reproduce vllm-ascend issue #11548 in one command:
#
#   ./scripts/repro_issue_11548.sh
#
# The script starts vLLM in xrs_vllm0.22.1, waits for /health, then runs:
#   completion -> sleep(level=1) -> wake_up -> completion
#
# Logs are written to logs/issue-11548-dp1-<run-id>-{server,request}.log.

set -eo pipefail

CONTAINER=xrs_vllm0.22.1
PORT=33333
MODEL=/home/weights/Qwen/Qwen3-30B-A3B
CONTAINER_WORKSPACE=/home/x50063850/vllm-workspace
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CONTAINER_LOG_DIR="$CONTAINER_WORKSPACE/logs"
SERVER_LOG="$LOG_DIR/issue-11548-dp1-$RUN_ID-server.log"
REQUEST_LOG="$LOG_DIR/issue-11548-dp1-$RUN_ID-request.log"
SERVER_LOG_IN_CONTAINER="$CONTAINER_LOG_DIR/issue-11548-dp1-$RUN_ID-server.log"
PID_FILE_IN_CONTAINER="/tmp/issue-11548-dp1-$RUN_ID.pid"

mkdir -p "$LOG_DIR"

cleanup() {
    docker exec "$CONTAINER" bash -lc '
        if test -f "'"$PID_FILE_IN_CONTAINER"'"; then
            pid=$(cat "'"$PID_FILE_IN_CONTAINER"'")
            kill "$pid" 2>/dev/null || true
        fi
    ' >/dev/null 2>&1 || true
}

wait_for_health() {
    local attempt
    for attempt in $(seq 1 120); do
        if docker exec "$CONTAINER" curl --noproxy "*" -sS --max-time 3 \
            "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done

    echo "vLLM did not become healthy; see $SERVER_LOG" >&2
    return 1
}

completion_request() {
    docker exec "$CONTAINER" curl --noproxy "*" -sS \
        "http://127.0.0.1:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{
            "prompt": "Beijing is a",
            "max_tokens": 5,
            "temperature": 0
        }'
}

trap cleanup EXIT INT TERM

echo "Starting vLLM in $CONTAINER with DP=1, TP=4."
echo "Server log:  $SERVER_LOG"
echo "Request log: $REQUEST_LOG"

: >"$SERVER_LOG"

# PYTHONPATH is expanded inside the container after set_env.sh. This keeps
# the Ascend ACL Python package visible to the worker processes.
docker exec -d "$CONTAINER" bash -lc '
    set -eo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_MODELSCOPE=True
    export VLLM_ASCEND_ENABLE_NZ=0
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export PYTHONPATH="/home/x50063850/vllm-workspace/vllm-ascend:/home/x50063850/vllm:$PYTHONPATH"
    python3 -m vllm.entrypoints.cli.main serve "'"$MODEL"'" \
        --data-parallel-size 1 \
        --enable-expert-parallel \
        --tensor-parallel-size 4 \
        --enable-sleep-mode \
        --enforce-eager \
        --port "'"$PORT"'" \
        >"'"$SERVER_LOG_IN_CONTAINER"'" 2>&1 &
    echo $! >"'"$PID_FILE_IN_CONTAINER"'"
    wait $!
' >/dev/null

wait_for_health

{
    echo '--- 1. First completion request ---'
    completion_request
    echo

    echo '--- 2. Sleep level 1 ---'
    docker exec "$CONTAINER" curl --noproxy "*" -sS \
        "http://127.0.0.1:$PORT/sleep" \
        -H 'Content-Type: application/json' \
        -d '{"level":"1"}'
    echo

    echo '--- 3. Wake up ---'
    docker exec "$CONTAINER" curl --noproxy "*" -sS -X POST \
        "http://127.0.0.1:$PORT/wake_up"
    echo

    echo '--- 4. Second completion request (issue trigger) ---'
    completion_request
    echo
} 2>&1 | tee "$REQUEST_LOG"

echo "Finished. Logs retained at:"
echo "  $SERVER_LOG"
echo "  $REQUEST_LOG"
