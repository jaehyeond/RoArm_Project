# Session 2026-06-10 - Transition X-Band Constant Baseline

## Scope

- Branch: professor 10cm / 0.72kg cube tap RL robustness/action-space branch.
- Goal: test whether the D227 fixed transition-bin PPO result generalizes to a
  narrow pose band and beats simple constant residual baselines.
- Runtime: local RTX4090/cuda:0 via
  `conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke`.
- No SSH/B200, pull, Track A, dataset, VLA, action-teacher, RoArm deployment,
  code change, reward change, action-space change, or gate change.

## Contract

- `rl_action_mode=candidate8_diffik_target_residual`
- `policy_action_space=3`
- `policy_target_disp_m=0.006`
- `tap_target_disp_tolerance_m=0.003`
- `tap_contact_proxy_mode=link5_collision_aabb`
- `candidate6_diffik_target_base_mode=previous_joint_target`
- `candidate6_diffik_target_path_mode=near_face_goal`
- `candidate6_diffik_cube_reference_mode=current_pose`
- `tap_success_terminate=True`
- `candidate8_diffik_target_residual_forward_m=0.004`
- `candidate8_diffik_target_residual_lateral_m=0.012`
- `candidate8_diffik_target_residual_height_m=0.004`

## Pose Band

- Center: `(x=0.1625, y=0.15)`
- Randomization: `cube_randomization_half_extent_x_m=0.0025`,
  `cube_randomization_half_extent_y_m=0.0`
- Effective x range: `[0.160, 0.165]`
- Effective y range: `[0.15, 0.15]`

## Step 1 - Zero/Base Band Eval

Source:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_transition_xband0160_0165_y015_base_seed1027_n64_summary.out`

Seed1027, n64, `max_iterations=0`, zero policy:

- `success_event_rate_per_env=0.34375`
- `target_band_max=0.046875`
- `overshoot_max=0.0`
- `tap_contact_seen_max=1.0`
- `reaction_seen_max=1.0`
- `ik_reset_rate_min=1.0`
- `candidate6_numeric_ok_rate_min=1.0`

Interpretation:

- The x-band is a useful event-success transition region.
- It is not a ceiling and not the D226/D227 deep infeasible bin.
- Promotion cannot use `success_episode_rate` here because success termination
  makes it less informative; use event rate and target-band.

## Step 2 - Constant Baselines

All runs used seed1027, n64, `max_iterations=0`, same x-band.

| Action | Event Rate | Target Band | Overshoot | Source |
|---|---:|---:|---:|---|
| `[0,0,0]` | `0.296875` | `0.0625` | `0.0` | `cube10cm_tap_transition_xband0160_0165_y015_const_zero_seed1027_n64_summary.out` |
| `[0.25,0,0]` | `0.546875` | `0.0625` | `0.0` | `cube10cm_tap_transition_xband0160_0165_y015_const_fwdp025_seed1027_n64_summary.out` |
| `[0.5,0,0]` | `1.0` | `0.09375` | `0.0` | `cube10cm_tap_transition_xband0160_0165_y015_const_fwdp05_seed1027_n64_summary.out` |
| `[1,0,0]` | `1.0` | `0.078125` | `0.0` | `cube10cm_tap_transition_xband0160_0165_y015_const_fwdp1_seed1027_n64_summary.out` |
| `[0,0.25,0]` | `0.21875` | `0.0625` | `0.0` | `cube10cm_tap_transition_xband0160_0165_y015_const_latp025_seed1027_n64_summary.out` |
| `[0,-0.25,0]` | `0.375` | `0.0625` | `0.0` | `cube10cm_tap_transition_xband0160_0165_y015_const_latneg025_seed1027_n64_summary.out` |

Interpretation:

- Constant forward residual can make event success reach `1.0`.
- Constant baselines do not solve the 6mm target-band quality. Best target-band
  in this screen is `[0.5,0,0]` at `0.09375`.
- Therefore event success alone is not evidence that residual RL is learning the
  task objective. Target-band and target-error must be the promotion gate.

## Step 3 - X-Band PPO L1

Source:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_transition_xband0160_0165_y015_3daction_l1_82k_seed1028_summary.out`

Setup:

- seed1028
- n64
- `num_steps_per_env=64`
- `max_iterations=20`
- `ppo_init_noise_std=0.2`
- no gates
- no L2/Large

Same-run zero/base pre-eval:

- `success_event_rate_per_env=0.34375`
- `target_band_max=0.046875`
- `overshoot_max=0.0`
- IK/numeric OK.

Post PPO:

- `success_event_rate_per_env=0.234375`
- `target_band_max=0.0625`
- `overshoot_max=0.0`
- `tap_target_disp_error_min=0.004991902969777584`
- `tap_target_excess_max=0.0`
- contact/reaction `1.0`
- IK/numeric OK.

Residual signal:

- `candidate8_target_residual_abs_max_max=0.0028817979618906975`
- `candidate8_forward_abs_max=0.0010318523272871971`
- `candidate8_lateral_abs_max=0.0028817979618906975`
- `candidate8_height_abs_max=0.0007148905424401164`

Checkpoint:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_runs/cube10cm_tap_transition_xband0160_0165_y015_3daction_l1_82k/seed1028_env64_it20/model_19.pt`

## Verdict

- Verdict: `TRANSITION_XBAND_CONSTANT_BASELINE_AND_L1_FAIL_NO_L2`.
- The x-band PPO L1 did not beat the best constant target-band (`0.0625` vs
  `0.09375`).
- The x-band PPO L1 also reduced event rate below same-run zero/base
  (`0.234375` vs `0.34375`).
- Overshoot stayed clean at `0.0`, so this was not an overshoot failure.
- The failure is target-quality / learning-stage failure under the x-band
  distribution, not geometry, IK, or numeric instability.

## Decision

- D227 fixed-bin PPO success remains real but fixed-only.
- D228 blocks x-band L2/Large PPO.
- Do not promote from event success alone in this branch.
- Promotion gate before any L2:
  - PPO target-band must beat best constant target-band on the same distribution.
  - PPO event rate must not degrade below same-run base unless target-band
    improvement is decisive.
  - overshoot must stay at base level, preferably `0.0`.
  - IK/numeric must remain OK.
  - contract violations must remain `0`.

## Next Valid Work

- Do not run L2/Large PPO from D227/D228.
- Either:
  - find a pose curriculum where target-band has a smoother learnable signal
    and constant baselines do not dominate event success; or
  - audit why success-event and 6mm target-band quality are decoupled in the
    current x-band before more PPO.
