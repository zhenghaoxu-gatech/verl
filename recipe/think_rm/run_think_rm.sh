#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

DATA_ROOT=${DATA_ROOT:-${HOME}/data/think_rm}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/rl/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_ROOT}/rl/validation.parquet}
PROJECT_NAME=${PROJECT_NAME:-verl_think_rm}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_think_rm_rloo}

cd "${REPO_ROOT}"

python -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Thinking-2507 \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    critic.enable=True \
    critic.model.path=Qwen/Qwen3-4B-Thinking-2507 \
    critic.model.trust_remote_code=True \
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
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
    trainer.nnodes=${N_NODES:-1} \
    trainer.logger='["console","wandb"]' \
    custom_reward_function.path=${SCRIPT_DIR}/reward_fn.py \
    custom_reward_function.name=compute_binary_reward \
    "${@}"
