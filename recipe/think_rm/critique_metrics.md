# Critic Calibration Metrics

The think_rm training scripts now log a set of critic-oriented metrics that help
track how well the value head matches the stochastic rewards implied by the
HelpSteer3 annotations. These appear in the trainer logs when
`trainer.log_value_calibration_metrics=True`.

- **critic/value_calibration/brier** – Mean squared error between the critic’s
  predicted probability for the actor’s verdict and the annotator-average
  probability for that verdict. Lower is better.
- **critic/value_calibration/logloss** – Cross-entropy against the same
  annotator-average target, highlighting over-confident mistakes.
- **critic/value_calibration/mae** / **mean_gap** – Absolute and signed
  difference between the critic probability and the annotator mean.
- **critic/value_calibration/pred_mean** – Average critic prediction across the
  batch.
- **critic/value_calibration/expected_mean** – Average of the annotator means
  for the actor’s verdict.
- **critic/value_calibration/observed_mean** – Average Monte-Carlo reward from
  the single annotator label sampled for training; useful to gauge variance.
- **critic/value_calibration/accuracy** – Fraction of samples where the 0.5
  threshold agrees with the annotator majority.
- **critic/value_calibration/ece** – Expected calibration error computed from a
  10-bin histogram.
- **critic/value_calibration/count** – Number of samples contributing to the
  calibration aggregates (i.e., those with parseable verdicts).
- **critic/value_calibration/total_count** – Total samples examined; the gap to
  `count` reveals how many verdicts were skipped because they could not be
  parsed.
- **critic/value_calibration/raw_mae** / **raw_mse** – Error measured on the
  unclamped logits (only emitted when using squared-loss value training).
- **critic/value_calibration/sample_brier** / **sample_mae** – Same error
  metrics but computed against the Monte-Carlo reward; these expose the noise
  floor introduced by single annotator draws.

Samples without a parseable verdict keep their per-sample probability as `NaN`
so that summary statistics can ignore them while still tracking how many were
skipped. All other PPO metrics (value variance, advantage gaps, etc.) remain
unchanged.

