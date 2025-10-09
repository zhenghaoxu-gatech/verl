#!/bin/bash

############# BEGIN OF CONFIG

INITIATIVE_ID=Rufus-shared
INITIATIVE_ID=Rufus-post-training
# INITIATIVE_ID=RufusPilotInitiative

DATA_ROOT=/root/data/dapo_rloo
OUTPUT_ROOT=/root/outputs/dapo_rloo
S3_OUTPUT=s3://shopqa-users/zxugt/results/dapo_rloo

PROJECT_NAME=${PROJECT_NAME:-verl_dapo_rloo}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_KEY=${WANDB_KEY:-5ddff28c7bb4d39af5a6d17e495058f834313ce6}
WANDB_PROJECT=${WANDB_PROJECT:-verl_dapo}
WANDB_GROUP=${WANDB_GROUP:-dapo_rloo}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
ROLLOUT_SAMPLES=${ROLLOUT_SAMPLES:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
CRITIC_PPO_MINI_BATCH_SIZE=${CRITIC_PPO_MINI_BATCH_SIZE:-64}
PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-5}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
CRITIC_VALUE_LOSS_TYPE=mle
# CRITIC_VALUE_LOSS_TYPE=squared
IS=seq_prod
IS=seq_mean
# IS=token

if [ "${IS}" = "seq_prod" ]; then
  : "${CLIP_RATIO:=0.8}"
  : "${CLIP_RATIO_LOW:=0.8}"
  : "${CLIP_RATIO_HIGH:=5.0}"
  : "${CLIP_RATIO_C:=6.0}"
fi

: "${CLIP_RATIO:=0.2}"
: "${CLIP_RATIO_LOW:=0.2}"
: "${CLIP_RATIO_HIGH:=0.2}"
: "${CLIP_RATIO_C:=3.0}"

MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
CONFIG_NAME=bs${TRAIN_BATCH_SIZE}_${PPO_MINI_BATCH_SIZE}_${IS}_${CLIP_RATIO_LOW}_${CLIP_RATIO_HIGH}_n${ROLLOUT_SAMPLES}_l${CRITIC_VALUE_LOSS_TYPE}
EXPERIMENT_NAME=qwen2_5_7b_dapo_rloo_${CONFIG_NAME}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-s3://shopqa-users/zxugt/checkpoints/verl}

JOB_NAME=dapo-qwen2_5-7b-rloo_${CONFIG_NAME}
############# END OF CONFIG

case "$INITIATIVE_ID" in
    "2025SFAIInternNonProdJobs" | "RufusPilotInitiative")
        INSTANCE_TYPE=p4de.24xlarge
        GREENLAND_REGION=us-east-1
        ;;
    "Rufus-shared" | "Rufus-post-training")
        INSTANCE_TYPE=p5en.48xlarge
        GREENLAND_REGION=us-west-2
        ;;
    *)
        echo "Unknown INITIATIVE_ID"
        exit 1
        ;;
esac

REGION=us-east-1
INSTANCE_COUNT=1
IS_PRODUCTION=false
NUM_NODES=${INSTANCE_COUNT}
ROLE=arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole
NUM_GPUS_PER_NODE=8
NUM_CPUS_PER_NODE=96
MEMORY_PER_NODE=1143265

WORKING_DIR=/root/verl
DATA_ROOT_LOCAL=${DATA_ROOT}
OUTPUT_ROOT_LOCAL=${OUTPUT_ROOT}
CKPT_DIR_LOCAL=${OUTPUT_ROOT_LOCAL}/checkpoints
TRAIN_PARQUET_LOCAL=${DATA_ROOT_LOCAL}/dapo-math-17k.parquet
VAL_PARQUET_LOCAL=${DATA_ROOT_LOCAL}/aime-2024.parquet
CKPT_S3_PREFIX_LOCAL=${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}


set -euo pipefail; \
cd ${WORKING_DIR} && \
pip3 install --no-deps -e .[vllm] && \
mkdir -p ${DATA_ROOT_LOCAL} ${OUTPUT_ROOT_LOCAL} ${CKPT_DIR_LOCAL} && \
if [ ! -f ${TRAIN_PARQUET_LOCAL} ] || [ ! -f ${VAL_PARQUET_LOCAL} ]; then \
  DATA_ROOT=${DATA_ROOT_LOCAL} TRAIN_PARQUET=${TRAIN_PARQUET_LOCAL} VAL_PARQUET=${VAL_PARQUET_LOCAL} bash recipe/dapo_rloo/prepare_dapo_data.sh; \
fi && \
export WANDB_MODE=${WANDB_MODE} && \
export WANDB_PROJECT=${WANDB_PROJECT} && \
export WANDB_GROUP=${WANDB_GROUP} && \
export WANDB_KEY=${WANDB_KEY} && \
export WANDB_API_KEY=${WANDB_KEY} && \
export HF_HOME=/root/.cache/huggingface && \
bash recipe/dapo_rloo/download_latest_checkpoint.sh ${CKPT_S3_PREFIX_LOCAL} ${CKPT_DIR_LOCAL} && \
{ \
  while true; do \
    aws s3 sync ${CKPT_DIR_LOCAL} ${CKPT_S3_PREFIX_LOCAL} --exclude latest_checkpointed_iteration.txt && \
    if [ -f ${CKPT_DIR_LOCAL}/latest_checkpointed_iteration.txt ]; then \
      aws s3 cp ${CKPT_DIR_LOCAL}/latest_checkpointed_iteration.txt ${CKPT_S3_PREFIX_LOCAL}/latest_checkpointed_iteration.txt; \
    fi && \
    sleep 300; \
  done & \
} && \
SYNC_PID=$! && \
trap 'kill \$SYNC_PID 2>/dev/null || true' EXIT && \
TRAIN_PARQUET=${TRAIN_PARQUET_LOCAL} VAL_PARQUET=${VAL_PARQUET_LOCAL} PROJECT_NAME=${PROJECT_NAME} EXPERIMENT_NAME=${EXPERIMENT_NAME} \
N_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} N_NODES=${NUM_NODES} \
bash recipe/dapo_rloo/run_dapo_rloo_2layer.sh \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.gen_batch_size=${GEN_BATCH_SIZE} \
  data.val_batch_size=null \
  data.max_prompt_length=${PROMPT_LENGTH} \
  data.max_response_length=${RESPONSE_LENGTH} \
  data.filter_overlong_prompts_workers=64 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.importance_ratio_mode=${IS} \
  actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
  actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.n=${ROLLOUT_SAMPLES} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
  critic.ppo_mini_batch_size=${CRITIC_PPO_MINI_BATCH_SIZE} \
  critic.use_dynamic_bsz=True \
  critic.value_loss_type=${CRITIC_VALUE_LOSS_TYPE} \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.log_value_calibration_metrics=True \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=2 \
  trainer.default_local_dir=${CKPT_DIR_LOCAL} \
  trainer.resume_mode=auto && \
kill $SYNC_PID 2>/dev/null || true && \
wait $SYNC_PID 2>/dev/null || true && \
aws s3 sync ${CKPT_DIR_LOCAL} ${CKPT_S3_PREFIX_LOCAL} --exclude latest_checkpointed_iteration.txt && \
if [ -f ${CKPT_DIR_LOCAL}/latest_checkpointed_iteration.txt ]; then \
  aws s3 cp ${CKPT_DIR_LOCAL}/latest_checkpointed_iteration.txt ${CKPT_S3_PREFIX_LOCAL}/latest_checkpointed_iteration.txt; \
fi && \
if [ -n ${S3_OUTPUT} ]; then \
  aws s3 sync ${OUTPUT_ROOT_LOCAL} ${S3_OUTPUT}; \
fi
