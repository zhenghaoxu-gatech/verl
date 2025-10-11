#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}
ACTOR_TOKENIZER_PATH=${ACTOR_TOKENIZER_PATH:-${ACTOR_MODEL_PATH}}
OUTPUT_DIR=${OUTPUT_DIR:-${HOME}/outputs/think_rm/baseline_eval}
DATASET=${DATASET:-allenai/reward-bench-2}
SPLIT=${SPLIT:-test}
RESULTS_FILE=${RESULTS_FILE:-rewardbench2_baseline_metrics.json}
ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
ACTOR_BACKEND=${ACTOR_BACKEND:-server}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
ACTOR_DATA_PARALLEL_SIZE=${ACTOR_DATA_PARALLEL_SIZE:-8}
ACTOR_SERVER_PORT=${ACTOR_SERVER_PORT:-8000}
ACTOR_SERVER_STARTUP_TIMEOUT=${ACTOR_SERVER_STARTUP_TIMEOUT:-600}
ACTOR_SERVER_SHUTDOWN_TIMEOUT=${ACTOR_SERVER_SHUTDOWN_TIMEOUT:-120}
ACTOR_REQUEST_CONCURRENCY=${ACTOR_REQUEST_CONCURRENCY:-64}
ACTOR_REQUEST_TIMEOUT=${ACTOR_REQUEST_TIMEOUT:-120}
MAX_EXAMPLES=${MAX_EXAMPLES:-}

cd "${REPO_ROOT}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --results-file "${RESULTS_FILE}"
  --actor-batch-size "${ACTOR_BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --actor-backend "${ACTOR_BACKEND}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --actor-data-parallel-size "${ACTOR_DATA_PARALLEL_SIZE}"
  --server-port "${ACTOR_SERVER_PORT}"
  --server-startup-timeout "${ACTOR_SERVER_STARTUP_TIMEOUT}"
  --server-shutdown-timeout "${ACTOR_SERVER_SHUTDOWN_TIMEOUT}"
  --actor-request-concurrency "${ACTOR_REQUEST_CONCURRENCY}"
  --actor-request-timeout "${ACTOR_REQUEST_TIMEOUT}"
  --actor-hf-path "${ACTOR_MODEL_PATH}"
  --actor-hf-tokenizer "${ACTOR_TOKENIZER_PATH}"
  --skip-critic
)

if [[ -n "${MAX_EXAMPLES}" ]]; then
  ARGS+=(--max-examples "${MAX_EXAMPLES}")
fi

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_GROUP:-}" ]]; then
  ARGS+=(--wandb-group "${WANDB_GROUP}")
fi

if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
  ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
fi

if [[ -n "${WANDB_MODE:-}" ]]; then
  ARGS+=(--wandb-mode "${WANDB_MODE}")
fi

python "${SCRIPT_DIR}/eval_rewardbench2.py" "${ARGS[@]}"
