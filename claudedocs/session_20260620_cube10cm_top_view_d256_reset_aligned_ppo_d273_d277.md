# 2026-06-20 - cube10cm top-view D256 reset-aligned PPO data-prior checks D273-D277

## Scope

This session stayed on the professor 10cm / 0.72kg cube top-view visual
trajectory branch. It did not run long PPO, RunPod, B200, RoArm deployment,
Track A, SmolVLA/VLA fine-tuning, rendering scale-up, or cleanup.

Goal:

1. Keep long PPO blocked.
2. Wire or probe D256 train-clean frame-0/state-distribution reset selection.
3. Re-run teacher-only with AABB/useful/vertical/overshoot/action-cap checks.
4. If teacher-only is plausible, run only tiny PPO smoke plus TensorBoard gate.
5. Do not claim learned policy or RoArm readiness before teacher-off eval.

## Why This Was Needed

D272 proved the D257 data-prior teacher was loaded and observable in
TensorBoard, but behavior failed from default PPO resets:

- tap contact/useful/success stayed `0.0`;
- vertical offset was about `0.15m`;
- max useful displacement stayed below `1mm`;
- overshoot appeared;
- joint-delta cap was high.

The critical contrast was D269: the same D257 teacher from a D256 initial reset
reached AABB contact/useful `0.71875/0.71875`. That made reset/pose
distribution the next blocker, not PPO scale.

## Code Changes

- `roarm_rl/roarm_cube_push_env.py`
  - Added opt-in reset config:
    - `d256_reset_csv_path`;
    - `d256_reset_frame_index`;
    - `d256_reset_sample_mode`.
  - Added lazy CSV load/sample helpers for D256 reset rows.
  - `_reset_idx()` now applies D256 arm joints, gripper, cube pose, target
    pose, and push direction when the reset path is configured.
  - Added TensorBoard extras for D256 reset activity.
- `roarm_rl/train_cube_push_ppo.py`
  - Added D256 reset CLI flags.
  - Added `--no_init_at_random_ep_len`.
- `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`
  - Added `--reset_pose_source env_d256_initial`.
  - Added vertical offset, face gap, lateral, reaction, and overshoot metrics.
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - Added D256 reset tags and `--expect_d256_reset`.

The reset hook is default-off. Existing reset behavior is unchanged when
`d256_reset_csv_path` is empty.

## D274 Teacher-Only Gate

Run: D257 teacher-only, `tap10cm`, fixed +x, `env_target`,
`link5_collision_aabb`, D256 env reset hook.

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d274_env_d256_reset_metrics/tap10cm/teacher_rollout_probe_summary_d274_env_d256_reset_teacher_only_metrics.json`

Key metrics:

- reset hook active rate: `1.0`
- initial feature outside train min/max: `0.0`
- AABB contact/useful/reaction: `0.71875` / `0.71875` / `0.71875`
- TCP-threshold contact: `0.0`
- tap overshoot seen: `0.03125`
- min tap contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last tap contact vertical offset mean/max: `0.028668411076068878` / `0.09710794687271118`
- max displacement along mean/max: `0.0014097457751631737` / `0.01252603530883789`
- raw delta clip exceed: `0.22213362068965517`
- action cap rate: `0.14152298850574713`

Interpretation:

- This passes the reasonableness gate for a tiny PPO smoke.
- It does not prove final policy quality.
- It confirms the D272 default-reset failure was not a reason to scale PPO; it
  was a reset/pose-distribution mismatch.

## D275 Random Episode Offset Failure

Run: tiny PPO, D256 reset, D257 teacher prior, but default
`init_at_random_ep_len=True`.

Gate:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d275_logs/cube10cm_d275_tap10cm_aabb_d256reset_bc_smoke/tensorboard_scalar_gate_d275.json`

Result:

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- issue: tap overshoot seen rate too high, max `0.125`
- D256 reset active: `1.0`
- BC teacher blend: `1.0`

Interpretation:

`init_at_random_ep_len=True` is incompatible with D256 frame-0 reset plus
`bc_teacher_phase_timing=direct_steps`. It randomizes the episode phase while
the reset state is still frame zero.

## D276 No-Random-Offset Short Smoke

Run: same tiny PPO but with `--no_init_at_random_ep_len`.

Gate:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d276_logs/cube10cm_d276_tap10cm_aabb_d256reset_bc_no_randlen_smoke/tensorboard_scalar_gate_d276.json`

Result:

- contact/useful improved relative to D272 default reset:
  - contact seen `0.45703125 -> 0.46875`
  - useful seen `0.45703125 -> 0.46875`
  - success `0.39453125 -> 0.42578125`
- overshoot stayed `0.0`
- D256 reset active `1.0`
- BC teacher blend `1.0`
- gate still failed because no full episode completed, so
  `Train/mean_reward` and `Train/mean_episode_length` were missing.

Interpretation:

No-random-offset is the right direction, but a 24-step smoke cannot verify the
reward dashboard when the task horizon is about 600 steps.

## D277 Episode-Complete Tiny Smoke

Run: one episode-complete tiny PPO smoke, still teacher-on:

- `tap10cm`
- `link5_collision_aabb`
- fixed +x
- D256 reset frame `0`
- `--no_init_at_random_ep_len`
- D257 checkpoint
- `bc_teacher_blend=1.0`
- `bc_teacher_imitation_reward_scale=5.0`
- `bc_teacher_feature_target_mode=env_target`
- `bc_teacher_phase_timing=direct_steps`
- `num_envs=32`
- `max_iterations=1`
- `num_steps_per_env=600`
- total timesteps `19,200`

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/tensorboard_scalar_gate_d277.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/tensorboard_scalar_gate_d277.md`

Gate:

- verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- issues: none
- warnings:
  - short run: `Train/mean_reward` has `1` point, promotion gate expects at
    least `10`;
  - raw TCP-cube distance high for tap/AABB diagnostic:
    `0.20408329367637634`.

Selected metrics:

- `Train/mean_reward`: `-3957.08154296875`
- `Train/mean_episode_length`: `599.0`
- `Loss/value_function`: `59124.1484375`
- `Policy/mean_noise_std`: `0.8021852970123291`
- `cube_push_d256_reset_active_rate`: `1.0`
- `cube_tap_d256_reset_active_rate`: `1.0`
- `cube_push_bc_teacher_blend_mean`: `1.0`
- `cube_tap_bc_teacher_blend_mean`: `1.0`
- `cube_push_bc_teacher_imitation_mse`: `0.66529381275177`
- `cube_tap_contact_seen_rate`: `0.6662499904632568`
- `cube_tap_reaction_seen_rate`: `0.6662499904632568`
- `cube_tap_useful_seen_rate`: `0.6469791531562805`
- `cube_tap_success_rate`: `0.6652604341506958`
- `cube_tap_overshoot_seen_rate`: `0.019687499850988388`
- `cube_tap_max_disp_along_m`: `0.0018036302644759417`
- `cube_tap_contact_vertical_offset_m`: `0.015306632034480572`
- `cube_push_joint_delta_cap_rate`: `0.15915799140930176`

Interpretation:

- D277 fixes the reward dashboard visibility gap from D276.
- AABB/contact/useful/reaction, vertical offset, overshoot, and action cap do
  not fail the gate.
- The raw TCP warning is diagnostic under the D270 AABB/tool-surface contract.
- The short-run warning is real and blocks promotion to long PPO.
- This is not a learned-policy result because the run is teacher-on with
  `bc_teacher_blend=1.0`.

## Verification

- `python -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_push_ppo.py sim_scripts/cube10cm_top_view_teacher_rollout_probe.py sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  passed.
- `git diff --check` passed.
- `ps -C python -C python3 -o pid,cmd` showed no active local Python process.
- `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv`
  showed the observed local baseline class state:
  RTX 4090 Laptop GPU, `16376MiB` total, `2509MiB` used, `13436MiB` free.

## Verdict

`D277_D256_RESET_ALIGNED_DATA_PRIOR_TINY_SMOKE_WARN_NO_LONG_PPO`

Do not run long PPO from this result.

D277 proves:

- D256 reset/pose selection can be wired into the env;
- teacher-only behavior becomes plausible under the dataset AABB contract;
- random episode offsets must be disabled for frame-0/direct-step teacher use;
- TensorBoard reward/loss/policy and reset/teacher/task metrics are observable.

D277 does not prove:

- learned policy success;
- teacher-off policy quality;
- RoArm readiness;
- long PPO readiness.

## Next Step

Run teacher-off frozen eval before any PPO ladder:

- same `tap10cm`;
- same `link5_collision_aabb`;
- same fixed +x / `env_target` feature contract;
- same D256 reset hook;
- same `--no_init_at_random_ep_len`;
- `bc_teacher_blend=0.0`;
- TensorBoard gate with `--expect_d256_reset`;
- require AABB contact/useful/reaction, controlled vertical offset, low
  overshoot, acceptable action cap, and reward/loss/policy scalar visibility.

Only if teacher-off frozen eval passes should a short controlled PPO ladder be
considered. No learned policy or RoArm readiness claim exists before that.
