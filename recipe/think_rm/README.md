# Think RM (HelpSteer3)

- Expands `nvidia/HelpSteer3` individual preferences into per-annotator RLHF records via `prepare_helpsteer3.py`.
- Trains a generative reward model (Qwen/Qwen3-4B-Thinking-2507) with RLOO advantages and a scalar critic.
- Uses `recipe/think_rm/reward_fn.py` to score rollouts with a binary reward (1 when the predicted sign matches the label).

## Data preparation

- Install dependencies inside the Verl image: `pip install datasets pandas pyarrow` if missing.
- Run `python recipe/think_rm/prepare_helpsteer3.py --output-dir /path/to/data` to create `rl/train.parquet` and `rl/validation.parquet`.
- Run `python recipe/think_rm/prepare_skywork_reward.py --output-dir /path/to/data --split train` to expand the Skywork preference pairs (each pair becomes forward/backward records in `rl/train.parquet`, automatically skipping pairs with mismatched contexts).
- Run `python recipe/think_rm/prepare_arena_human_preference.py --output-dir /path/to/data --split train` to convert the LMSYS Arena votes into per-comparison records (ties and both_bad votes are merged into the `tie` label so the reward model can learn neutrality).
- Optional flags: `--max-samples` to cap expanded pairs, `--dump-metadata` to emit simple JSON stats.

## Training entry point

- `recipe/think_rm/run_think_rm.sh` calls `python -m verl.trainer.main_ppo` with the relevant overrides.
- Override `TRAIN_PARQUET` or `VAL_PARQUET` env vars to point at custom locations.
- Enable logging backends by exporting `WANDB_*` variables before running when online tracking is desired.
- Submit to Greenland with `submit/train_think_rm.sh`, which prepares data inside the `zxugt-rlhf:verl` image and launches the run script.

## Reward and critic

- `reward_fn.compute_binary_reward` looks for the `<label>1|2|0</label>` verdict (fallback to legacy formats) and returns `{score: 0|1}`.
- `critic.enable=True` in the run script so the critic learns from the same Monte Carlo rollouts (lambda=gamma=1).
