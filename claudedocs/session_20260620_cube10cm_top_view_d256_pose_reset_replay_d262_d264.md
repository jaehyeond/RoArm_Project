# D262-D264 D256 Pose-distribution Reset and Replay Probe

Date: 2026-06-20 KST

Scope:

- Professor 10cm / 0.72kg cube top-view visual trajectory branch only.
- No PPO learning.
- No long PPO.
- No teacher-off evaluation.
- No learned-policy or RoArm readiness claim.
- No RunPod/B200/SSH/pull/cleanup.

## Question

The D261 IK reset reduced TCP distance but produced raw teacher delta explosion.
The working hypothesis was:

> TCP-only IK reset is insufficient. The reset must match the D256 joint/TCP
> feature distribution that the D257 teacher actually saw during training.

This session tested that hypothesis with visualization, D256 initial-pose reset,
and direct D256 action replay.

## D262 Visualization

Added:

- `sim_scripts/cube10cm_top_view_d256_feature_distribution_viz.py`

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/`

Key files:

- `d256_vs_d261_normalized_support_bars.png`
- `d256_arm_joint_state_distribution.png`
- `d256_tcp_target_relative_geometry_distribution.png`
- `d256_feature_distribution_summary_d262.md`

Counts:

- rows / episodes: `142978` / `737`
- label counts: `{'clean_useful_tap': 142978}`
- D256 joint delta abs over `0.04rad`: `0.14844941179761922`

Interpretation:

- D261 live env ranges sit outside D256 support for several critical features:
  arm joints, `tcp_local_z_m`, `target_to_tcp_*`, and `tcp_to_cube_*`.
- Therefore the D257 MLP teacher is extrapolating under D261 reset conditions.

## D263 D256 Initial-pose Reset Teacher-only Probe

Updated:

- `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`

New option:

- `--reset_pose_source d256_initial`

This injects D256 frame-0 joint/cube/target/push-dir state into the live env
before running D257 teacher-only.

Command:

```bash
conda run -n isaaclab env PYTHONPATH=. python sim_scripts/cube10cm_top_view_teacher_rollout_probe.py --env_kind tap10cm --out_dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d263_d256_initial_reset --num_envs 32 --steps 580 --sample_every 20 --artifact_tag d263_d256_initial_reset --fixed_push_dir_x 1 --fixed_push_dir_y 0 --bc_teacher_feature_target_mode env_target --reset_pose_source d256_initial --d256_reset_frame_index 0
```

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d263_d256_initial_reset/tap10cm/teacher_rollout_probe_summary_d263_d256_initial_reset.json`

Key results:

- reset source: `d256_initial`
- selected D256 frame: `0`
- reset episode unique count: `32`
- initial feature outside train min/max rate: `0.0`
- initial feature outside train p01/p99 rate: `0.19328703703703703`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max:
  `0.08348368108272552` / `0.06940185278654099` /
  `0.09543989598751068`
- max disp along mean/min/max:
  `0.0014523034915328026` / `0.000009059906005859375` /
  `0.01252603530883789`
- raw delta clip exceed rate: `0.20877155172413794`
- raw delta abs max: `0.6774565577507019`
- action cap rate: `0.13050466954022988`
- rollout feature outside train min/max rate: `0.16936063218390804`

Interpretation:

- D256 initial-pose reset is a real improvement:
  - D261 no-IK raw clip exceed was `0.7170689655172414`;
  - D263 raw clip exceed is `0.20877155172413794`;
  - D261 no-IK action cap was `0.37896012931034484`;
  - D263 action cap is `0.13050466954022988`.
- But teacher-only still does not reach contact.
- Matching the reset pose manifold is necessary but not sufficient.

## D264 Direct D256 Action Replay

Added:

- `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py`

This disables PPO and disables the teacher. It:

1. resets the env from D256 frame-0 state;
2. replays D256 `state + joint_delta` as direct joint targets;
3. holds each D256 row target for `3` env steps to map about `194` D256
   transitions to about `580` env steps.

Command:

```bash
conda run -n isaaclab env PYTHONPATH=. python sim_scripts/cube10cm_top_view_d256_action_replay_probe.py --num_envs 32 --steps 580 --hold_steps 3 --sample_every 20
```

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_action_replay_probe_d264/tap10cm/d256_action_replay_summary_d264.json`

Key results:

- teacher used: `False`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max:
  `0.07518836855888367` / `0.06179572641849518` /
  `0.09923214465379715`
- max disp along mean/min/max:
  `0.006767723709344864` / `0.000009298324584960938` /
  `0.017127275466918945`
- max disp xy mean/max:
  `0.007403433322906494` / `0.017222005873918533`
- max target jump abs mean/max:
  `0.06703907251358032` / `0.09352636337280273`

Interpretation:

- D256 action replay moves the cube more than D263 teacher-only and gets close
  to contact, but still does not cross the current env contact threshold:
  `tcp_cube_dist < 0.055m`.
- Therefore the blocker is not only D257 MLP teacher generalization.
- The D256 visual action/control/contact contract does not yet directly
  reproduce in the current env under this replay timing and current contact
  metric.

## Post-run Checks

- No matching Isaac/PPO/teacher-probe/action-replay/torchrun/rl_games process
  remained.
- GPU returned to the observed baseline: about `2509MiB` used /
  `13436MiB` free.

## Verdict

`D264_D256_POSE_RESET_IMPROVES_SUPPORT_ACTION_REPLAY_STILL_NO_CONTACT_NO_PPO`

Do not run PPO. The next valid work is a replay-contract diagnostic, not PPO.

## Next Work

Diagnose why D256 visual rows labelled clean/useful do not replay into current
env contact:

- frame-to-env-step timing (`hold_steps`, frame cadence, action hold);
- action target semantics (`state + joint_delta` vs target trajectory timing);
- current env contact proxy/threshold (`tcp_cube_dist < 0.055m`) versus the
  visual label/contact metric;
- TCP point versus tool-surface geometry;
- whether replay should use direct joint state sequence, direct joint targets,
  or a low-level trajectory-following controller.

Only after replay reproduces contact should D257 teacher retraining or tiny PPO
be considered.
