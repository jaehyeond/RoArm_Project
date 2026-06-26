# D281 Cube10cm Top-view Env Stop Contract / PPO Update Safety

Date: 2026-06-25 KST

## Scope

- Track: professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.
- Contract: `tap10cm + link5_collision_aabb + D256 frame-0 reset + fixed +x + episode_length_s=6.0`.
- Goal: encode the D280 stop-after-useful diagnostic as an env contract and test whether PPO can update a warm-start actor without breaking it.
- Not done: long PPO, learned-policy claim, RoArm readiness claim, RunPod/B200 work, cleanup, Track A.

## Code Changes

- Added default-off tap env flags:
  - `RoArmCubeTap10cmEnvCfg.tap_stop_after_useful_seen`
  - `RoArmCubeTap10cmEnvCfg.tap_useful_terminate`
- Added env-side useful-hold action zeroing and min-contact vertical tracking.
- Added TensorBoard/env scalars:
  - `cube_tap_min_contact_vertical_offset_m`
  - `cube_tap_min_contact_vertical_finite_rate`
  - `cube_tap_stop_after_useful_hold_rate`
- Added teacher-off/trace support for env stop/useful terminate and min-contact vertical gate.
- Added PPO CLI controls:
  - `--tap_stop_after_useful_seen`
  - `--tap_useful_terminate`
  - `--init_noise_std`
  - `--ppo_learning_rate`
  - `--ppo_num_learning_epochs`
  - `--ppo_num_mini_batches`
  - `--ppo_entropy_coef`
  - `--ppo_clip_param`
  - `--ppo_desired_kl`
  - `--ppo_max_grad_norm`
- Important runtime detail: warm-start checkpoints restore rsl_rl action-noise `std`, so `--init_noise_std` is now applied again after `runner.load()`.

## Pre-PPO Contract Checks

- D280 actor teacher-off eval with env stop and min-contact gate:
  - verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
  - useful: `0.71875`
  - overshoot: `0.0`
  - joint cap max trace: `0.2135416716337204`
  - vertical gate mode/value: `min_contact` / `0.0`
  - D256 reset active: `1.0`
  - BC teacher blend: `0.0`
- D280 actor-vs-teacher trace with env stop and min-contact gate:
  - verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
  - diagnostic class: `no_major_trace_blocker`
  - MSE/MAE/cosine: `0.05346343293786049` / `0.13944056630134583` / `0.6641471982002258`
  - useful: `0.71875`
  - overshoot: `0.0`
  - joint cap max trace: `0.2135416716337204`

Interpretation: the stop-after-useful contract fixes the D280 diagnostic failure mode before PPO. This is not a learned-policy claim.

## PPO Smoke Results

### Default PPO noise

- One 1-iteration warm-start PPO smoke used D280 actor, D256 reset, AABB contact proxy, `bc_teacher_blend=0.0`, imitation reward scale `0.05`, env useful stop/terminate, and `tap_overshoot_terminate`.
- TensorBoard gate verdict:
  `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- Hard issue:
  `joint-delta cap rate too high: max=0.3923611342906952`
- Root-cause evidence:
  PPO collection used `Mean action noise std: 0.80`.

### Noise 0.1 with post-load std override

- Added post-warm-start std override and reran with `--init_noise_std 0.1`.
- PPO output confirmed:
  - `ppo_init_noise_std_after_warm_start=0.100000`
  - `Mean action noise std: 0.10`
- TensorBoard gate verdict:
  `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- Hard issues: none.
- Key scalars:
  - useful `0.0716145858168602`
  - overshoot `0.01171875`
  - joint cap `0.1571180671453476`
  - BC imitation MSE `0.03936203941702843`
  - D256 reset active `1.0`
- Warnings:
  short run only, raw TCP diagnostic high, and max displacement along still tiny (`1.3730334103456698e-05m`).

Saved-checkpoint eval invalidated promotion:

- `model_0.pt` teacher-off frozen eval:
  - verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
  - useful: `0.0`
  - overshoot: `0.90625`
  - joint cap max trace: `0.7135416865348816`
- `model_0.pt` actor-vs-teacher trace:
  - verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
  - class: `actor_teacher_mismatch_plus_unsafe_physics`
  - MSE/cosine: `0.2621913254261017` / `0.6388046145439148`
  - useful: `0.0`
  - overshoot: `0.90625`

### Conservative PPO update

- Reran one 1-iteration smoke with:
  - `--init_noise_std 0.1`
  - `--ppo_learning_rate 1e-6`
  - `--ppo_num_learning_epochs 1`
  - `--ppo_entropy_coef 0.0`
  - `--ppo_clip_param 0.05`
  - `--ppo_desired_kl 0.001`
  - `--ppo_max_grad_norm 0.1`
- TensorBoard gate verdict:
  `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- Hard issues: none.
- Key collection scalars:
  - useful `0.07421875`
  - overshoot `0.00911458395421505`
  - joint cap `0.1549479365348816`
  - BC imitation MSE `0.04458475112915039`
  - D256 reset active `1.0`

Saved-checkpoint eval still blocked promotion:

- `model_0.pt` teacher-off frozen eval:
  - verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
  - useful: `0.0`
  - contact/reaction: `0.3125` / `0.3125`
  - overshoot: `0.34375`
  - joint cap max trace: `0.7447916865348816`
- `model_0.pt` actor-vs-teacher trace:
  - verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
  - class: `actor_teacher_mismatch_plus_unsafe_physics`
  - MSE/cosine: `0.150615` / `0.777609`
  - useful: `0.0`
  - overshoot: `0.34375`

## Decision

- D281 passes the pre-PPO env contract gate.
- D281 fixes PPO collection action-noise safety when `--init_noise_std 0.1` is applied after warm-start load.
- D281 does not produce a promotable PPO checkpoint.
- TensorBoard collection gates are necessary but not sufficient. The saved checkpoint must also pass teacher-off frozen eval and actor-vs-teacher trace.
- Do not run long PPO or a PPO ladder from D281.
- Do not claim learned policy, teacher-off success, or RoArm readiness.

## Next Work

- Treat PPO actor update as the current blocker, not reset, AABB contact, or pre-PPO actor/teacher bridge.
- Add a real actor-preservation mechanism before another PPO ladder:
  - auxiliary supervised teacher/action loss inside PPO, or
  - KL/behavior-cloning regularization to the D280 actor/teacher, or
  - staged actor-freeze/value-only warmup followed by very small actor updates.
- Any next PPO smoke must use:
  - D256 reset/AABB contract,
  - env stop/useful termination,
  - post-load low action-noise override,
  - TensorBoard gate,
  - saved-checkpoint teacher-off eval,
  - saved-checkpoint actor-vs-teacher trace.

Sources:

- `START_HERE.md`
- `roarm_rl/roarm_cube_push_env.py`
- `roarm_rl/train_cube_push_ppo.py`
- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
- `sim_scripts/cube10cm_top_view_actor_teacher_trace.py`
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/teacher_off_eval_env_stop_min_contact/teacher_off_policy_eval_summary_d281_env_stop_min_contact.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/actor_teacher_trace_env_stop_min_contact/actor_teacher_trace_summary_d279.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_smoke/cube10cm_d281_warmstart_useful_stop_smoke/tensorboard_scalar_gate_d281.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_noise010_reloadstd_smoke/cube10cm_d281_warmstart_useful_stop_noise010_reloadstd_smoke/tensorboard_scalar_gate_d281_noise010_reloadstd.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_noise010_reloadstd_smoke/cube10cm_d281_warmstart_useful_stop_noise010_reloadstd_smoke/teacher_off_eval_after_noise010_smoke/teacher_off_policy_eval_summary_d281_after_noise010_smoke.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_noise010_reloadstd_smoke/cube10cm_d281_warmstart_useful_stop_noise010_reloadstd_smoke/actor_teacher_trace_after_noise010_smoke/actor_teacher_trace_summary_d279.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/tensorboard_scalar_gate_d281_conservative_update.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/teacher_off_eval_after_conservative_update/teacher_off_policy_eval_summary_d281_after_conservative_update.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/actor_teacher_trace_after_conservative_update/actor_teacher_trace_summary_d279.json`
