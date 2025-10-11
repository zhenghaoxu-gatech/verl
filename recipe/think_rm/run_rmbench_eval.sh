#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

DEFAULT_CHECKPOINT_ROOT=${CKPT_DIR_LOCAL:-${HOME}/outputs/think_rm/checkpoints}  # shellcheck disable=SC2153
CHECKPOINT_ROOT=${CHECKPOINT_ROOT-${DEFAULT_CHECKPOINT_ROOT}}  # preserve explicit empty value if provided
OUTPUT_DIR=${OUTPUT_DIR:-${THINK_RM_EVAL_DIR:-${HOME}/outputs/think_rm/rmbench}}  # shellcheck disable=SC2153
DATASET=${DATASET:-THU-KEG/RM-Bench}
dataset_lc=$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
RESULTS_FILE=${RESULTS_FILE:-rmbench_metrics.json}
ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE:-4}
CRITIC_BATCH_SIZE=${CRITIC_BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
THRESHOLD=${THRESHOLD:-0.5}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
REUSE_EXPORT=${REUSE_EXPORT:-0}
ACTOR_BACKEND=${ACTOR_BACKEND:-server}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
ACTOR_DATA_PARALLEL_SIZE=${ACTOR_DATA_PARALLEL_SIZE:-8}
ACTOR_SERVER_PORT=${ACTOR_SERVER_PORT:-8000}
ACTOR_SERVER_STARTUP_TIMEOUT=${ACTOR_SERVER_STARTUP_TIMEOUT:-600}
ACTOR_SERVER_SHUTDOWN_TIMEOUT=${ACTOR_SERVER_SHUTDOWN_TIMEOUT:-120}
ACTOR_REQUEST_CONCURRENCY=${ACTOR_REQUEST_CONCURRENCY:-64}
ACTOR_REQUEST_TIMEOUT=${ACTOR_REQUEST_TIMEOUT:-120}
CRITIC_NUM_WORKERS=${CRITIC_NUM_WORKERS:-}
CRITIC_LOSS_TYPE=${CRITIC_LOSS_TYPE:-mle}
SPLIT=train

cd "${REPO_ROOT}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --results-file "${RESULTS_FILE}"
  --actor-batch-size "${ACTOR_BATCH_SIZE}"
  --critic-batch-size "${CRITIC_BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --threshold "${THRESHOLD}"
  --actor-backend "${ACTOR_BACKEND}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --actor-data-parallel-size "${ACTOR_DATA_PARALLEL_SIZE}"
  --server-port "${ACTOR_SERVER_PORT}"
  --server-startup-timeout "${ACTOR_SERVER_STARTUP_TIMEOUT}"
  --server-shutdown-timeout "${ACTOR_SERVER_SHUTDOWN_TIMEOUT}"
  --actor-request-concurrency "${ACTOR_REQUEST_CONCURRENCY}"
  --actor-request-timeout "${ACTOR_REQUEST_TIMEOUT}"
  --critic-loss-type "${CRITIC_LOSS_TYPE}"
)

if [[ -n "${MAX_EXAMPLES}" ]]; then
  ARGS+=(--max-examples "${MAX_EXAMPLES}")
fi

if [[ -n "${CRITIC_NUM_WORKERS}" ]]; then
  ARGS+=(--critic-num-workers "${CRITIC_NUM_WORKERS}")
fi

if [[ "${REUSE_EXPORT}" == "1" || "${REUSE_EXPORT}" == "true" ]]; then
  ARGS+=(--reuse-export)
fi

if [[ -n "${CHECKPOINT_STEP:-}" ]]; then
  ARGS+=(--checkpoint-step "${CHECKPOINT_STEP}")
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

if [[ -n "${CHECKPOINT_ROOT}" ]]; then
  if [[ -d "${CHECKPOINT_ROOT}" ]]; then
    ARGS+=(--checkpoint-root "${CHECKPOINT_ROOT}")
  else
    echo "Warning: checkpoint root not found at ${CHECKPOINT_ROOT}, skipping checkpoint-based evaluation." >&2
  fi
fi

python "${SCRIPT_DIR}/eval_rmbench.py" "${ARGS[@]}" "$@"
