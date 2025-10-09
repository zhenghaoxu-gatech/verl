#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-${HOME}/data/dapo_rloo}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/dapo-math-17k.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_ROOT}/aime-2024.parquet}
OVERWRITE=${OVERWRITE:-0}

mkdir -p "${DATA_ROOT}"

if [ ! -f "${TRAIN_PARQUET}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  # wget -O "${TRAIN_PARQUET}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
  wget -O "${TRAIN_PARQUET}" "https://huggingface.co/datasets/fengyao1909/dapo-math-17k-deduplicated/resolve/main/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${VAL_PARQUET}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${VAL_PARQUET}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
  
fi
