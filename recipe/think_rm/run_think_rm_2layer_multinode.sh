#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DATA_ROOT=${DATA_ROOT:-${HOME}/data/think_rm}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/rl/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_ROOT}/rl/validation.parquet}
PROJECT_NAME=${PROJECT_NAME:-verl_think_rm}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_think_rm_rloo}

CRITIC_VALUE_LOSS_TYPE=${CRITIC_VALUE_LOSS_TYPE:-mle}
CRITIC_VALUE_HEAD_HIDDEN_SIZES=${CRITIC_VALUE_HEAD_HIDDEN_SIZES:-"[2560]"}
CRITIC_VALUE_HEAD_ACTIVATION=${CRITIC_VALUE_HEAD_ACTIVATION:-silu}
CRITIC_VALUE_HEAD_DROPOUT=${CRITIC_VALUE_HEAD_DROPOUT:-0.0}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token"}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-"Qwen/Qwen3-4B-Thinking-2507"}
ACTOR_LR=${ACTOR_LR:-1e-6}
USE_CRITIC=${USE_CRITIC:-True}
ACTOR_DTYPE=${ACTOR_DTYPE:-"bfloat16"}
ROLLOUT_DTYPE=${ROLLOUT_DTYPE:-"bfloat16"}

cd "${REPO_ROOT}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${VAL_PARQUET} \
    data.train_batch_size=256 \
    data.val_batch_size=null \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=${BASE_MODEL_NAME} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.dtype=${ACTOR_DTYPE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE:-1} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-1} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
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
    critic.ppo_mini_batch_size=64 \
    critic.use_dynamic_bsz=True \
    critic.optim.lr=5e-6 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.log_value_calibration_metrics=True \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
    trainer.nnodes=${N_NODES:-1} \
    trainer.logger='["console","wandb"]' \
    custom_reward_function.path=${SCRIPT_DIR}/reward_fn.py \
    custom_reward_function.name=compute_binary_reward \
    "${@}"
