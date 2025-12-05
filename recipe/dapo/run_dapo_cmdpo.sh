#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DATA_ROOT=${DATA_ROOT:-${HOME}/data/dapo_pmd}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/dapo-math-17k.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_ROOT}/aime-2024.parquet}
PROJECT_NAME=${PROJECT_NAME:-verl_dapo_pmd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_dapo_pmd}

CRITIC_VALUE_LOSS_TYPE=${CRITIC_VALUE_LOSS_TYPE:-mle}
CRITIC_VALUE_HEAD_HIDDEN_SIZES=${CRITIC_VALUE_HEAD_HIDDEN_SIZES:-"[2560]"}
CRITIC_VALUE_HEAD_ACTIVATION=${CRITIC_VALUE_HEAD_ACTIVATION:-silu}
CRITIC_VALUE_HEAD_DROPOUT=${CRITIC_VALUE_HEAD_DROPOUT:-0.0}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-sum-norm}
IMPORTANCE_RATIO_MODE=${IMPORTANCE_RATIO_MODE:-token}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-partition}
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-pmd}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-"Qwen/Qwen2.5-7B"}
ACTOR_LR=${ACTOR_LR:-5e-7}
USE_CRITIC=${USE_CRITIC:-True}
ACTOR_DTYPE=${ACTOR_DTYPE:-"bfloat16"}
ROLLOUT_DTYPE=${ROLLOUT_DTYPE:-"bfloat16"}
PMD_TAU=${PMD_TAU:-0.1}
PMD_ALPHA=${PMD_ALPHA:-1.0}
RESET_OPTIMIZER_FREQ=${RESET_OPTIMIZER_FREQ:-0}
PPO_EPOCHS=${PPO_EPOCHS:-1}

PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN * 2))}
ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN * 3))}
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU}}
REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU}}

VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.7}
VAL_TOP_K=${VAL_TOP_K:--1}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}

REWARD_OVERLONG_ENABLE=${REWARD_OVERLONG_ENABLE:-False}
REWARD_OVERLONG_LEN=${REWARD_OVERLONG_LEN:-0}
REWARD_OVERLONG_PENALTY=${REWARD_OVERLONG_PENALTY:-0.0}
REWARD_OVERLONG_LOG=${REWARD_OVERLONG_LOG:-False}

ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_CHUNKED_PREFILL=${ROLLOUT_CHUNKED_PREFILL:-True}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}

cd "${REPO_ROOT}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    -- python -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    +algorithm.partition_tau=${PMD_TAU} \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${VAL_PARQUET} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.train_batch_size=512 \
    data.val_batch_size=null \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.shuffle=True \
    actor_rollout_ref.model.path=${BASE_MODEL_NAME} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.importance_ratio_mode=${IMPORTANCE_RATIO_MODE} \
    actor_rollout_ref.actor.dtype=${ACTOR_DTYPE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE:-1} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE} \
    +actor_rollout_ref.actor.policy_loss.pmd_tau=${PMD_TAU} \
    +actor_rollout_ref.actor.policy_loss.pmd_alpha=${PMD_ALPHA} \
    +actor_rollout_ref.actor.reset_optimizer_states_freq=${RESET_OPTIMIZER_FREQ} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-1} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_CHUNKED_PREFILL} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE:-1} \
    critic.enable=${USE_CRITIC} \
    critic.model.path=${BASE_MODEL_NAME} \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.trust_remote_code=True \
    critic.value_loss_type=${CRITIC_VALUE_LOSS_TYPE} \
    critic.model.value_head.hidden_sizes=${CRITIC_VALUE_HEAD_HIDDEN_SIZES} \
    critic.model.value_head.activation=${CRITIC_VALUE_HEAD_ACTIVATION} \
    critic.model.value_head.dropout=${CRITIC_VALUE_HEAD_DROPOUT} \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.ppo_mini_batch_size=32 \
    critic.use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU} \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.optim.lr=5e-6 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${REWARD_OVERLONG_ENABLE} \
    reward_model.overlong_buffer.len=${REWARD_OVERLONG_LEN} \
    reward_model.overlong_buffer.penalty_factor=${REWARD_OVERLONG_PENALTY} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.log_value_calibration_metrics=True \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
    trainer.nnodes=${N_NODES:-1} \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=True \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR:-${HOME}/verl/ckpts/${EXPERIMENT_NAME}} \
    trainer.resume_mode=auto \
    "${@}"
