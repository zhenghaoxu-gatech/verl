#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <s3-prefix> <local-dir> <step>" >&2
  exit 1
fi

S3_PREFIX="${1%/}"
LOCAL_DIR="$2"
STEP_INPUT="$3"

if [[ "${STEP_INPUT}" == global_step_* ]]; then
  STEP_DIR="${STEP_INPUT}"
else
  STEP_DIR="global_step_${STEP_INPUT}"
fi

REMOTE_PATH="${S3_PREFIX}/${STEP_DIR}"

if ! aws s3 ls "${REMOTE_PATH}/" >/dev/null 2>&1; then
  echo "Checkpoint directory ${REMOTE_PATH} not found." >&2
  exit 1
fi

mkdir -p "${LOCAL_DIR}"

echo "Downloading checkpoint ${STEP_DIR} from ${S3_PREFIX} ..."
aws s3 sync "${REMOTE_PATH}/" "${LOCAL_DIR}/${STEP_DIR}/"

echo "${STEP_DIR}" > "${LOCAL_DIR}/latest_checkpointed_iteration.txt"
echo "Checkpoint ${STEP_DIR} downloaded to ${LOCAL_DIR}/${STEP_DIR}"
