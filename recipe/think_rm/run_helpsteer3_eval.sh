#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CHECKPOINT_ROOT_WAS_SET=${CHECKPOINT_ROOT+x}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${CKPT_DIR_LOCAL:-${HOME}/outputs/think_rm/checkpoints}}  # shellcheck disable=SC2153
OUTPUT_DIR=${OUTPUT_DIR:-${THINK_RM_EVAL_DIR:-${HOME}/outputs/think_rm/helpsteer3}}
RESULTS_FILE=${RESULTS_FILE:-helpsteer3_metrics.json}
DATASET=${DATASET:-nvidia/HelpSteer3}

TRAIN_SPLIT=${TRAIN_SPLIT:-train}
TRAIN_SUBSET_TAG=${TRAIN_SUBSET_TAG:-train_head}
TRAIN_LIMIT=${TRAIN_LIMIT:-2000}

VALIDATION_SPLIT=${VALIDATION_SPLIT:-validation}
VALIDATION_SUBSET_TAG=${VALIDATION_SUBSET_TAG:-validation}
VALIDATION_LIMIT=${VALIDATION_LIMIT:-}

ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE:-4}
CRITIC_BATCH_SIZE=${CRITIC_BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
THRESHOLD=${THRESHOLD:-0.5}

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

ACTOR_HF_PATH=${ACTOR_HF_PATH:-}
ACTOR_HF_TOKENIZER=${ACTOR_HF_TOKENIZER:-}
CRITIC_HF_PATH=${CRITIC_HF_PATH:-}
CRITIC_HF_TOKENIZER=${CRITIC_HF_TOKENIZER:-}

if [[ -n "${ACTOR_HF_PATH}" && -z "${CHECKPOINT_ROOT_WAS_SET}" ]]; then
  CHECKPOINT_ROOT=""
fi

CHECKPOINT_STEP=${CHECKPOINT_STEP:-}
REUSE_EXPORT=${REUSE_EXPORT:-0}
SKIP_CRITIC=${SKIP_CRITIC:-0}

WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_MODE=${WANDB_MODE:-}

cd "${REPO_ROOT}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --results-file "${RESULTS_FILE}"
  --dataset "${DATASET}"
  --train-split "${TRAIN_SPLIT}"
  --train-subset-tag "${TRAIN_SUBSET_TAG}"
  --train-limit "${TRAIN_LIMIT}"
  --validation-split "${VALIDATION_SPLIT}"
  --validation-subset-tag "${VALIDATION_SUBSET_TAG}"
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

if [[ -n "${CHECKPOINT_ROOT}" ]]; then
  ARGS+=(--checkpoint-root "${CHECKPOINT_ROOT}")
fi

if [[ -n "${CHECKPOINT_STEP}" && -n "${CHECKPOINT_ROOT}" ]]; then
  ARGS+=(--checkpoint-step "${CHECKPOINT_STEP}")
fi

if [[ -n "${ACTOR_HF_PATH}" ]]; then
  ARGS+=(--actor-hf-path "${ACTOR_HF_PATH}")
fi

if [[ -n "${ACTOR_HF_TOKENIZER}" ]]; then
  ARGS+=(--actor-hf-tokenizer "${ACTOR_HF_TOKENIZER}")
fi

if [[ -n "${CRITIC_HF_PATH}" ]]; then
  ARGS+=(--critic-hf-path "${CRITIC_HF_PATH}")
fi

if [[ -n "${CRITIC_HF_TOKENIZER}" ]]; then
  ARGS+=(--critic-hf-tokenizer "${CRITIC_HF_TOKENIZER}")
fi

if [[ -n "${CRITIC_NUM_WORKERS}" ]]; then
  ARGS+=(--critic-num-workers "${CRITIC_NUM_WORKERS}")
fi

if [[ -n "${VALIDATION_LIMIT}" ]]; then
  ARGS+=(--validation-limit "${VALIDATION_LIMIT}")
fi

if [[ "${REUSE_EXPORT}" == "1" || "${REUSE_EXPORT,,}" == "true" ]]; then
  ARGS+=(--reuse-export)
fi

if [[ "${SKIP_CRITIC}" == "1" || "${SKIP_CRITIC,,}" == "true" ]]; then
  ARGS+=(--skip-critic)
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_GROUP}" ]]; then
  ARGS+=(--wandb-group "${WANDB_GROUP}")
fi

if [[ -n "${WANDB_RUN_NAME}" ]]; then
  ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
fi

if [[ -n "${WANDB_MODE}" ]]; then
  ARGS+=(--wandb-mode "${WANDB_MODE}")
fi

python "${SCRIPT_DIR}/eval_helpsteer3.py" "${ARGS[@]}"
