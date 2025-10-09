#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DATA_ROOT=${DATA_ROOT:-${HOME}/data/dapo_rloo}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/dapo-math-17k.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_ROOT}/aime-2024.parquet}
PROJECT_NAME=${PROJECT_NAME:-verl_dapo_rloo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_dapo_rloo}

ACTOR_LR=${ACTOR_LR:-1e-6}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
ROLLOUT_SAMPLES=${ROLLOUT_SAMPLES:-16}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
CRITIC_PPO_MINI_BATCH_SIZE=${CRITIC_PPO_MINI_BATCH_SIZE:-32}
PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
CRITIC_VALUE_LOSS_TYPE=${CRITIC_VALUE_LOSS_TYPE:-mle}
CRITIC_VALUE_HEAD_HIDDEN_SIZES=${CRITIC_VALUE_HEAD_HIDDEN_SIZES:-"[2560]"}
CRITIC_VALUE_HEAD_ACTIVATION=${CRITIC_VALUE_HEAD_ACTIVATION:-silu}
CRITIC_VALUE_HEAD_DROPOUT=${CRITIC_VALUE_HEAD_DROPOUT:-0.0}
OVERLONG_BUFFER_ENABLE=${OVERLONG_BUFFER_ENABLE:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-2048}
OVERLONG_BUFFER_PENALTY=${OVERLONG_BUFFER_PENALTY:-1.0}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.7}
VAL_TOP_K=${VAL_TOP_K:--1}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}

ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}}
REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}}
CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU=${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
CRITIC_PPO_INFER_MAX_TOKEN_LEN_PER_GPU=${CRITIC_PPO_INFER_MAX_TOKEN_LEN_PER_GPU:-${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU}}

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}

MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))

cd "${REPO_ROOT}"

python -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=rloo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${VAL_PARQUET} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=null \
    data.gen_batch_size=${GEN_BATCH_SIZE} \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.shuffle=True \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU} \
    +actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.importance_ratio_mode=token \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_SAMPLES} \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    critic.enable=True \
    critic.model.path=${ACTOR_MODEL_PATH} \
    critic.model.trust_remote_code=${TRUST_REMOTE_CODE} \
    critic.value_loss_type=${CRITIC_VALUE_LOSS_TYPE} \
    critic.model.value_head.hidden_sizes=${CRITIC_VALUE_HEAD_HIDDEN_SIZES} \
    critic.model.value_head.activation=${CRITIC_VALUE_HEAD_ACTIVATION} \
    critic.model.value_head.dropout=${CRITIC_VALUE_HEAD_DROPOUT} \
    critic.model.use_remove_padding=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.ppo_mini_batch_size=${CRITIC_PPO_MINI_BATCH_SIZE} \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU} \
    +critic.ppo_infer_max_token_len_per_gpu=${CRITIC_PPO_INFER_MAX_TOKEN_LEN_PER_GPU} \
    critic.optim.lr=5e-6 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${OVERLONG_BUFFER_ENABLE} \
    reward_model.overlong_buffer.len=${OVERLONG_BUFFER_LEN} \
    reward_model.overlong_buffer.penalty_factor=${OVERLONG_BUFFER_PENALTY} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.log_value_calibration_metrics=True \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
    trainer.nnodes=${N_NODES:-1} \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=True \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR:-${HOME}/verl/ckpts/${EXPERIMENT_NAME}} \
    trainer.resume_mode=auto \
    "$@"
