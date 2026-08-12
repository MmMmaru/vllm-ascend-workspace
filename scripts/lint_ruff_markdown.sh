#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_DIR="${SCRIPT_DIR}/../vllm-ascend"

if ! command -v pre-commit >/dev/null 2>&1; then
    echo "pre-commit is required. Install it with: python -m pip install -r vllm-ascend/requirements-lint.txt" >&2
    exit 1
fi

cd "${ASCEND_DIR}"

run_hook() {
    local hook="$1"
    shift

    pre-commit run "${hook}" --all-files "$@" && return 0
    pre-commit run "${hook}" --all-files "$@"
}

run_hook ruff-format
run_hook ruff-check
run_hook markdownlint --hook-stage manual
