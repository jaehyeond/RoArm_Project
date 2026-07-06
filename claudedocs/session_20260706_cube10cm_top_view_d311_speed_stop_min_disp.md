# D311 Cube10cm Push Primitive Speed-Stop Minimum Displacement

Date: 2026-07-06 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch after D310. This session did not run PPO, tiny PPO trace gates, TensorBoard training, torchrun, learned-policy updates, RoArm deployment, Track A, VLA/SmolVLA fine-tuning, B200/SSH, pull, or `.ssh` copy.

## Starting Point

D310 moved the working D309 non-PPO tool/object push primitive into the env runtime contract as default-off `rl_action_mode="tap_push_primitive"`.

The next valid work was broader non-PPO primitive diagnostics and contact/control contract hardening before any PPO gate.

## Code Changes

- `roarm_rl/roarm_cube_push_env.py`
  - Added default-preserving config `tap_push_primitive_speed_stop_min_disp_m = 0.0`.
  - The existing speed stop now triggers only when both:
    - cube speed is at or above `tap_push_primitive_speed_stop_mps`; and
    - current XY displacement is at or above `tap_push_primitive_speed_stop_min_disp_m`.
  - The default `0.0` preserves the D310 behavior unless a diagnostic explicitly opts in.

- `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
  - Added `--primitive_speed_stop_min_disp_m`.
  - Wires the CLI value into the env primitive config.
  - Records `primitive_speed_stop_min_disp_m` in the summary JSON.

## Non-PPO Diagnostics

Common settings:

- `--reset_pose_source env_hook`
- corrected `--no-env_hook_force_second_reset`
- `--d256_reset_sample_mode random`
- `--exec_source env_tap_push_primitive`
- `--primitive_target_path_mode legacy_far_face_through`
- `--primitive_target_base_mode previous_joint_target`
- `--primitive_cube_reference_mode start_pose`
- `--primitive_goal_disp_m 0.003`
- `--primitive_push_steps 220`
- `--primitive_speed_stop_mps 0.200`
- `--primitive_diffik_step_clip_rad 0.010`
- `--tap_stop_after_disp_m 0.003`
- `--no-tap_stop_after_useful_seen`
- actor checkpoint matched to D310:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`

### Baseline Extension

Runs:

- `fresh32_random_env_tap_push_primitive_matched_seed30703`
- `fresh32_random_env_tap_push_primitive_matched_seed30704`

Seed `30703`:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- primitive stop-latched: `32/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max/min: `3.819/6.013/3.002mm`
- XY `>=1mm`: `32/32`
- XY `>=3mm`: `32/32`
- XY `>=7mm`: `0/32`
- XY `>=20mm`: `0/32`

Seed `30704`:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- primitive stop-latched: `32/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max/min: `3.713/5.493/0.701mm`
- XY `>=1mm`: `31/32`
- XY `>=3mm`: `31/32`
- XY `>=7mm`: `0/32`
- XY `>=20mm`: `0/32`

The single low-displacement env was seed `30704`, env `19`, D256 episode `700`:

- max XY: `0.7008mm`
- max along: `0.0021mm`
- contact/reaction/useful/final proxy: true
- overshoot: false
- primitive stop step: `1`

Interpretation: this was not no-contact and not overshoot. It was an early speed-stop primitive termination before the intended displacement floor.

### Speed-Stop Minimum Displacement Gate

Runs:

- `fresh32_random_env_tap_push_primitive_matched_seed30703_speedmin001`
- `fresh32_random_env_tap_push_primitive_matched_seed30704_speedmin001`

Changed setting:

- `--primitive_speed_stop_min_disp_m 0.001`

Seed `30703` with speed-min:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- primitive stop-latched: `32/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max/min: `3.817/5.945/3.002mm`
- XY `>=1mm`: `32/32`
- XY `>=3mm`: `32/32`
- XY `>=7mm`: `0/32`
- XY `>=20mm`: `0/32`

Seed `30704` with speed-min:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- primitive stop-latched: `32/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max/min: `3.777/5.493/3.024mm`
- XY `>=1mm`: `32/32`
- XY `>=3mm`: `32/32`
- XY `>=7mm`: `0/32`
- XY `>=20mm`: `0/32`

Combined speed-min `64` envs:

- contact/reaction/useful/final proxy: `64/64/64/64`
- overshoot: `0/64`
- primitive stop-latched: `64/64`
- cap: `0.0`
- max XY mean/max/min: `3.797/5.945/3.002mm`
- XY `>=1mm`: `64/64`
- XY `>=3mm`: `64/64`
- XY `>=7mm`: `0/64`
- XY `>=20mm`: `0/64`

## Interpretation

D311 confirms that D310's env primitive remains strong under two additional random fresh32 seeds, but the original speed-stop condition can terminate too early in rare cases when speed crosses the stop threshold before useful displacement accumulates.

Adding a minimum displacement gate to speed-stop is the correct control-contract hardening. It is not PPO, not learned-policy promotion, and not threshold tuning of the reward. It changes the primitive termination semantics so speed alone cannot end a push before the minimum useful displacement floor.

The default remains `0.0` to preserve D310 behavior until more diagnostics justify promoting `0.001m` as the recommended runtime setting.

## Verification

- `python -m py_compile sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py roarm_rl/roarm_cube_push_env.py`: pass
- `git diff --check`: pass
- Fresh non-PPO Isaac diagnostics: pass for seeds `30703` and `30704`, plus speed-min re-runs.

## Verdict

`D311_SPEED_STOP_MIN_DISP_CONTROL_HARDENING_PASS_NO_PPO_PROMOTION`

Next work: broaden speed-min primitive diagnostics across more random seeds and contact-proxy/approach-face edge cases before deciding whether `primitive_speed_stop_min_disp_m=0.001` should become the recommended default for this branch. No PPO gate yet.
