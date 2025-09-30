#!/bin/bash

############# BEGIN OF CONFIG

INITIATIVE_ID=Rufus-shared
# INITIATIVE_ID=Rufus-post-training
# INITIATIVE_ID=RufusPilotInitiative


DATA_ROOT=/root/data/think_rm
OUTPUT_ROOT=/root/outputs/think_rm
S3_OUTPUT=s3://shopqa-users/zxugt/results/think_rm

PROJECT_NAME=${PROJECT_NAME:-verl_think_rm}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_KEY=${WANDB_KEY:-5ddff28c7bb4d39af5a6d17e495058f834313ce6}
WANDB_PROJECT=${WANDB_PROJECT:-verl_think_rm}
WANDB_GROUP=${WANDB_GROUP:-think_rm}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PROMPT_LENGTH=${PROMPT_LENGTH:-4096}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-4096}
ROLLOUT_SAMPLES=${ROLLOUT_SAMPLES:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
CRITIC_PPO_MINI_BATCH_SIZE=${CRITIC_PPO_MINI_BATCH_SIZE:-128}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-5}
CRITIC_VALUE_LOSS_TYPE=squared
# CRITIC_VALUE_LOSS_TYPE=mle

MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_think_rm_p${PROMPT_LENGTH}_r${RESPONSE_LENGTH}_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_SAMPLES}_l${CRITIC_VALUE_LOSS_TYPE}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-s3://shopqa-users/zxugt/checkpoints/verl}

JOB_NAME=think-rm-qwen3-4b-rloo_p${PROMPT_LENGTH}_r${RESPONSE_LENGTH}_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_SAMPLES}_l${CRITIC_VALUE_LOSS_TYPE}
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
TRAIN_PARQUET_LOCAL=${DATA_ROOT_LOCAL}/rl/train.parquet
VAL_PARQUET_LOCAL=${DATA_ROOT_LOCAL}/rl/validation.parquet
CKPT_S3_PREFIX_LOCAL=${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}


set -euo pipefail; \
cd ${WORKING_DIR} && \
pip3 install -e .[vllm] && \
mkdir -p ${DATA_ROOT_LOCAL}/rl ${OUTPUT_ROOT_LOCAL} ${CKPT_DIR_LOCAL} && \
if [ ! -f ${TRAIN_PARQUET_LOCAL} ] || [ ! -f ${VAL_PARQUET_LOCAL} ]; then \
  python recipe/think_rm/prepare_helpsteer3.py --output-dir ${DATA_ROOT_LOCAL} --splits train validation; \
fi && \
export WANDB_MODE=${WANDB_MODE} && \
export WANDB_PROJECT=${WANDB_PROJECT} && \
export WANDB_GROUP=${WANDB_GROUP} && \
export WANDB_KEY=${WANDB_KEY} && \
export WANDB_API_KEY=${WANDB_KEY} && \
export HF_HOME=/root/.cache/huggingface && \
bash recipe/think_rm/download_latest_checkpoint.sh ${CKPT_S3_PREFIX_LOCAL} ${CKPT_DIR_LOCAL} && \
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
bash recipe/think_rm/run_think_rm_2layer.sh \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.val_batch_size=null \
  data.max_prompt_length=${PROMPT_LENGTH} \
  data.max_response_length=${RESPONSE_LENGTH} \
  data.filter_overlong_prompts_workers=64 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
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
