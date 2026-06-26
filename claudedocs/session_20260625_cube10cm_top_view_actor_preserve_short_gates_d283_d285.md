# Cube10cm Top-View Short Preserved-Actor PPO Gates D283-D285

Date: 2026-06-25 KST

## Scope

- Branch context: professor 10cm / 0.72kg cube top-view visual trajectory
  dataset branch.
- No long PPO, render, cleanup, B200/SSH, RunPod runtime, RoArm deployment, or
  Track A work was run.
- Goal: test whether D282 actor-preservation makes short PPO promotion safe.

## Shared Runtime Contract

- Env: `RoArm-CubeTap10cm-Direct-v0`
- Contact proxy: `link5_collision_aabb`
- Reset: D256 train-clean frame-0 reset, `linspace`
- Episode: `episode_length_s=6.0`, `--no_init_at_random_ep_len`
- Warm start: D280 distilled actor
- Teacher sidecar: D257 state-action teacher, `bc_teacher_blend=0.0`,
  imitation reward scale `0.05`, `env_target`, `direct_steps`
- PPO size: `10` iterations, `num_envs=32`, `num_steps_per_env=24`
- Training termination: `tap_stop_after_useful_seen`,
  `tap_useful_terminate`, and `tap_overshoot_terminate`
- Saved-checkpoint eval/trace protocol: no `tap_useful_terminate`, with
  `tap_stop_after_useful_seen` and `vertical_gate_mode=min_contact`

## D283 Actor-Preserve095, Noise 0.1

Command class:

- `--actor_preserve_blend 0.95`
- `--init_noise_std 0.1`

TensorBoard gate:

- verdict `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- issue `joint-delta cap rate too high: max=0.6579861640930176`
- reward `-6.631277561187744 -> 5.1462788581848145`
- useful last `0.03125`
- overshoot max `0.00911458395421505`
- BC imitation MSE last `0.0750335305929184`
- action abs max last `0.8954037427902222`

Saved `model_9.pt` corrected teacher-off eval:

- verdict `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
- useful/overshoot/joint-cap `0.71875/0.0/0.2135416716337204`
- policy action abs max trace `3.6983871459960938`

Saved `model_9.pt` corrected actor-vs-teacher trace:

- verdict `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
- MSE/cosine `0.05202638357877731/0.6651849150657654`
- useful/overshoot `0.71875/0.0`
- actor raw clip exceed max `0.109375`

Interpretation: the saved deterministic policy remained usable, but collection
failed. This blocks promotion.

## D284 Actor-Preserve095, Noise 0.02

Command class:

- `--actor_preserve_blend 0.95`
- `--init_noise_std 0.02`

TensorBoard gate:

- verdict `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- issue `joint-delta cap rate too high: max=0.6430121660232544`
- reward `-6.0151824951171875 -> 5.879624843597412`
- useful last `0.03125`
- overshoot max `0.00911458395421505`
- BC imitation MSE last `0.07686792314052582`
- action abs max last `0.8672488927841187`
- noise last `0.020000148564577103`

Interpretation: reducing exploration noise did not fix collection.

## D285 Actor-Freeze, Noise 0.02

Command class:

- `--actor_preserve_blend 1.0`
- `--init_noise_std 0.02`

Runtime confirmed full actor restoration:

```text
actor_preserve_after_update blend=1.000000 ... max_post_restore_delta=0.000000000
```

TensorBoard gate:

- verdict `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- issue `joint-delta cap rate too high: max=0.6536458730697632`
- reward `-6.012892246246338 -> 5.879622936248779`
- useful last `0.03125`
- overshoot max `0.00911458395421505`
- BC imitation MSE last `0.06847621500492096`
- action abs max last `0.8579933643341064`
- noise last `0.019999999552965164`
- D256 reset episode-index mean last `660.375`

Interpretation: even a fully frozen actor fails collection as reset samples move
through later D256 episode-index regions. Actor update drift is not the only
blocker.

## Decision

Verdict:

```text
D285_COLLECTION_GATE_FAILS_EVEN_ACTOR_FREEZE_NO_LONG_PPO
```

Do not run long PPO. Do not promote D283, D284, or D285.

Current diagnosis:

- Actor-preservation can preserve saved deterministic checkpoints.
- TensorBoard collection still fails.
- Lower exploration noise does not fix it.
- Full actor freeze does not fix it.
- The likely blocker is reset-sample/state-distribution plus action-cap
  pressure during collection.

Next work:

1. Bin D256 reset rows by episode index and state features.
2. Evaluate the D280 actor action magnitude and joint-cap rate per bin.
3. Restrict or stratify reset samples to bins that are contact/useful and
   action-cap safe.
4. Alternatively add an explicit action-cap or teacher-KL constraint before
   another short PPO gate.

Reward increase alone is not a policy-progress claim.

## Validation

- D283, D284, and D285 runtimes exited cleanly.
- D283 saved-checkpoint teacher-off eval and actor-vs-teacher trace completed.
- D284/D285 were not promoted after TensorBoard failures.
- Post-run process check found no narrowed PPO/trace/eval/torchrun/rl_games
  process.
- GPU returned to baseline class state:
  `NVIDIA GeForce RTX 4090 Laptop GPU, 833 MiB used, 15111 MiB free, 0% util`.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md#d283-d285`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `roarm_rl/train_cube_push_ppo.py`
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
- `sim_scripts/cube10cm_top_view_actor_teacher_trace.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/tensorboard_scalar_gate_d283_preserve095_10.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/teacher_off_eval_model9_no_useful_term/teacher_off_policy_eval_summary_d283_preserve095_10_model9_no_useful_term.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/actor_teacher_trace_model9_no_useful_term/actor_teacher_trace_summary_d279.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d284/tap10cm/ppo_preserve095_noise002_10_smoke/cube10cm_d284_preserve095_noise002_10_smoke/tensorboard_scalar_gate_d284_preserve095_noise002_10.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/tensorboard_scalar_gate_d285_actorfreeze_noise002_10.json`
