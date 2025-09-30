#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <s3-prefix> <local-dir>" >&2
  exit 1
fi

S3_PREFIX="${1%/}"
LOCAL_DIR="$2"
LATEST_MARKER="latest_checkpointed_iteration.txt"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required but was not found in PATH" >&2
  exit 1
fi

if ! aws s3 ls "${S3_PREFIX}/" >/dev/null 2>&1; then
  echo "No checkpoints found at ${S3_PREFIX}; skipping download." >&2
  exit 0
fi

mkdir -p "${LOCAL_DIR}"

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

MARKER_TMP="${TMP_DIR}/${LATEST_MARKER}"
if ! aws s3 cp "${S3_PREFIX}/${LATEST_MARKER}" "${MARKER_TMP}" >/dev/null 2>&1; then
  echo "Marker ${LATEST_MARKER} missing at ${S3_PREFIX}; falling back to full sync." >&2
  aws s3 sync "${S3_PREFIX}/" "${LOCAL_DIR}/"
  exit 0
fi

LATEST_STEP=$(tr -d '[:space:]' < "${MARKER_TMP}")
if [[ -z "${LATEST_STEP}" ]]; then
  echo "Marker file ${LATEST_MARKER} is empty; falling back to full sync." >&2
  aws s3 sync "${S3_PREFIX}/" "${LOCAL_DIR}/"
  exit 0
fi

if [[ "${LATEST_STEP}" == global_step_* ]]; then
  STEP_DIR="${LATEST_STEP}"
else
  STEP_DIR="global_step_${LATEST_STEP}"
fi

REMOTE_STEP_PATH="${S3_PREFIX}/${STEP_DIR}"
if ! aws s3 ls "${REMOTE_STEP_PATH}/" >/dev/null 2>&1; then
  echo "Checkpoint directory ${REMOTE_STEP_PATH} missing; falling back to full sync." >&2
  aws s3 sync "${S3_PREFIX}/" "${LOCAL_DIR}/"
  exit 0
fi

aws s3 cp "${S3_PREFIX}/${LATEST_MARKER}" "${LOCAL_DIR}/${LATEST_MARKER}"
aws s3 sync "${REMOTE_STEP_PATH}/" "${LOCAL_DIR}/${STEP_DIR}/"

echo "Downloaded checkpoint ${STEP_DIR} from ${S3_PREFIX} to ${LOCAL_DIR}."
