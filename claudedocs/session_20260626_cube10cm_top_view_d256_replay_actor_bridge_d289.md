# D289 Cube10cm D256 Replay-Actor Bridge Diagnostic

Date: 2026-06-26 KST

## Scope

Professor 10cm / 0.72kg cube top-view visual trajectory dataset branch only.
No long PPO, no tiny PPO promotion, no render, no RunPod/B200, no RoArm claim.

Goal: replace the unsafe D257 closed-loop MLP teacher path with a D256
recorded-action replay actor warm-start path, then check whether that path is
safe enough for teacher-off/bin diagnostics before PPO.

## What Passed

- Extended `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py` with
  `execution_mode=policy_actions` and the current action-contract knobs.
- D256 recorded actions are representable through the normal policy action path:
  - path:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_action_replay_probe_d289/tap10cm/d256_action_replay_summary_d289_policy_actions_32env_steps580.json`
  - contact/useful: `1.0 / 1.0`
  - max XY displacement: `0.015575507655739784m`
  - action contract: `action_scale=0.04`, `max_joint_delta_per_step_rad=0.04`,
    `action_smoothing_alpha=1.0`, `joint_delta_reference=joint_pos`.
- Added `sim_scripts/cube10cm_top_view_distill_actor_from_d256_replay.py`.
- A 32-episode supervised replay-action actor fit passed offline:
  - checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_distill_d289/tap10cm/model_actor_d256_replay_d289.pt`
  - final validation MSE/MAE/cosine:
    `0.0072013 / 0.0446665 / 0.951601`
- The 32-episode actor passed one teacher-off frozen eval:
  - path:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_distill_d289/tap10cm/teacher_off_eval/teacher_off_policy_eval_summary_d289.json`
  - contact/useful/reaction: `0.96875 / 0.96875 / 0.96875`
  - overshoot: `0.0`
  - max XY mean/max: `0.0001242639118572697 / 0.00327298603951931`

## What Failed

- The same 32-episode actor failed D256 reset-bin coverage:
  - path:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d289_d256_replay_actor_5bins/d256_reset_bin_actor_probe_summary_d286.json`
  - verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
  - useful max by 5 bins:
    `0.8125 / 0.21875 / 0.4375 / 0.25 / 0.375`
  - overshoot max by 5 bins:
    `0.15625 / 0.84375 / 0.71875 / 0.6875 / 0.625`
  - policy raw action max reached `186.31187438964844` in one bin.
- This means the first teacher-off pass was not enough. It was a narrow reset
  sample pass, not a broad D256 distribution pass.

## Control Experiments

- Fresh-process recorded-action replay controls passed for all five D256 episode
  bins:
  - bin0 `1..208`: useful `1.0`, max XY `0.009117063134908676m`
  - bin1 `209..370`: useful `1.0`, max XY `0.015542421489953995m`
  - bin2 `371..537`: useful `1.0`, max XY `0.016205472871661186m`
  - bin3 `538..715`: useful `1.0`, max XY `0.016418220475316048m`
  - bin4 `716..999`: useful `1.0`, max XY `0.011478593572974205m`
- Exact fresh replay of the global-160 batch2 episode list also passed:
  - selected episodes:
    `210,216,220,226,232,237,242,247,252,257,261,267,272,276,281,288,292,297,302,307,313,318,323,329,333,338,342,347,353,357,362,368`
  - contact/useful: `1.0 / 1.0`
  - max XY: `0.014092150144279003m`

## Multi-Batch Collection Finding

- Attempting to collect 160 episodes inside one Isaac process produced replay
  contamination after the first batch even after tap-buffer reset and scene sync:
  - batch1 `1..205`: useful/overshoot/maxXY `1.0 / 0.0 / 0.014227m`
  - batch2 `210..368`: `0.5 / 0.5 / 0.034543m`
  - batch3 `372..536`: `0.34375 / 0.65625 / 10.880388m`
  - batch4 `540..714`: `0.40625 / 0.59375 / 0.134980m`
  - batch5 `719..999`: `0.46875 / 0.53125 / 0.041255m`
- Fresh replay of the same episode lists is clean, so this is not a D256 label
  problem. It is an in-process multi-batch Isaac/manual-reset collection
  artifact.
- In-process fresh env recreation was also tested and hung after the first batch.
  The script now fails fast for `--fresh_env_per_batch` and for
  `collection_episode_count > num_envs`.

## Code Changes

- `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py`
  - added `execution_mode=policy_actions`
  - added action-contract knobs
  - added `episode_min`, `episode_max`, and `episode_indices`
- `sim_scripts/cube10cm_top_view_distill_actor_from_d256_replay.py`
  - added D256 replay-action actor distillation
  - added batch summaries
  - added guards that block unsafe same-process multi-batch collection
- `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`
  - `apply_d256_pose_reset()` now clears tap buffers and syncs scene state after
    manual pose writes.
- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
  and `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`
  now accept the same action-contract knobs used by the replay actor path.

## Decision

The D256 recorded-action path is the right direction, but PPO is still blocked.
The current 32-episode actor warm-start is too narrow, and the naive 160-episode
same-process collection path is invalid.

This is not being too strict. The broad bin probe exposed real closed-loop
overshoot that a single teacher-off average hid. Reward/TensorBoard PPO should
not be used until the broad teacher-off/bin gates pass.

## Next Work

1. Build separate-process single-batch dataset collection for D256 replay-action
   actor distillation.
2. Merge those clean batch datasets offline.
3. Train a broader replay-action actor from the merged dataset.
4. Rerun teacher-off frozen eval and D256 reset-bin diagnostics.
5. Only if those pass, run tiny PPO smoke plus TensorBoard gate.

No learned policy, teacher-off broad success, tiny PPO promotion, long PPO, or
RoArm readiness claim exists after D289.
