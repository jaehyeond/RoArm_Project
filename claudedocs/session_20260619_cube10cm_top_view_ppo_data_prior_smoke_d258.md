# Session 2026-06-19 - Cube10cm Top-View PPO Data-Prior Smoke D258

## Scope

Active branch:

- Professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.

Goal:

- Continue the method pipeline from camera contract -> Isaac Lab visual trajectory
  -> post-render label validation -> LeRobot mp4/parquet -> split curation ->
  training/eval connection by running the first tiny Isaac Lab PPO data-prior
  smoke from the D257 teacher checkpoint.

This session did not run SmolVLA/VLA fine-tuning, RunPod, B200/SSH/pull, raw
cleanup, RoArm control, teacher-off eval, or any long PPO.

## Current-State Checks

- `git status --short --untracked-files=all --branch` was clean at session start.
- `CLAUDE.md` Current-State Protocol was re-read.
- `START_HERE.md` D257 current truth was re-read:
  - D257 has LeRobot pair -> RL transition/reward table -> PPO-compatible
    state-action teacher checkpoint.
  - No Isaac Lab PPO runtime had been run before D258.
- `claudedocs/DECISIONS.md` D257 was re-read.
- D257 checkpoint preflight passed:
  - sha256 `f81df20278ec9ceddef141729f717abbba2412a4a2f9f3a366d88b387caa76b8`;
  - required keys present;
  - feature count `27`;
  - target count `5`;
  - `x_mean` shape `(27,)`;
  - `y_mean` shape `(5,)`.
- Pre-run process check found no active Isaac/PPO/torchrun process.
- Pre-run GPU check: RTX 4090 Laptop GPU, `16376MiB` total,
  `2509MiB` used, `13436MiB` free.
- Disk remained tight: repo and `/tmp` about `590G` total, `531G` used,
  `30G` available, `95%` used.

## First Attempt

Command:

- Ran the D257 command file as written:
  `bash .../state_action_teacher_d257/ppo_data_prior_smoke_command_d257.txt`

Result:

- Failed before a valid smoke.
- Isaac/PhysX could not see a CUDA-capable device inside the sandbox.
- The command also failed to import `roarm_rl` because the script was executed as
  `python roarm_rl/train_cube_push_ppo.py` without `PYTHONPATH=.`.

Interpretation:

- This first attempt is a pre-runtime environment failure, not a PPO result.
- Future direct script execution needs `PYTHONPATH=.` from repo root, or the
  training entrypoint should be refactored to avoid that import-path trap.

## Host Runtime Smoke

Reran outside the sandbox with host GPU approval and `PYTHONPATH=.`.

Effective command:

```bash
PYTHONPATH=. conda run -n isaaclab python roarm_rl/train_cube_push_ppo.py \
  --num_envs 32 \
  --max_iterations 2 \
  --seed 1257 \
  --experiment_name cube10cm_d257_data_prior_smoke2 \
  --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs \
  --episode_length_s 6.0 \
  --action_scale 0.04 \
  --action_smoothing_alpha 1.0 \
  --max_joint_delta_per_step_rad 0.04 \
  --contact_joint_delta_scale 1.0 \
  --fast_cube_joint_delta_scale 1.0 \
  --joint_target_lead_limit_rad 0.06 \
  --joint_delta_reference joint_pos \
  --bc_teacher_checkpoint_path claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt \
  --bc_teacher_blend 1.0 \
  --bc_teacher_imitation_reward_scale 5.0 \
  --bc_teacher_policy_delta_clip_rad 0.04 \
  --bc_teacher_policy_delta_scale 1.0 \
  --bc_teacher_lowx_policy_delta_scale 1.0 \
  --bc_teacher_highx_policy_delta_scale 0.8 \
  --bc_teacher_delta_smoothing_alpha 0.85 \
  --bc_teacher_phase_timing direct_steps \
  --num_steps_per_env 24 \
  --save_interval 1
```

Output directory:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2`

Generated runtime files:

- `events.out.tfevents.1781806911.cgxr-Legion-Pro-7-16IRX9H.915799.0`
- `model_0.pt`
- `model_1.pt`

Git policy:

- The event file and `model_*.pt` are generated runtime artifacts and were added
  to `.gitignore`.
- Tracked evidence is the summary JSON/markdown plus this session log.

## Smoke Metrics

TensorBoard scalar extraction from the event file:

- `cube_push_bc_teacher_blend_mean`: `1.0`, `1.0`
- `cube_push_bc_teacher_imitation_mse`: `1.210442`, `1.253437`
- `bc_teacher_imitation_penalty`: `-6.052209`, `-6.267184`
- `cube_push_disp_along_m`: `-0.0`, `0.000151`
- `cube_push_disp_xy_m`: `0.000008`, `0.000615`
- `cube_push_tcp_cube_dist_m`: `0.338168`, `0.32687`
- `cube_push_controlled_rate`: `0.0`, `0.018229`
- `cube_push_low_motion_rate`: `1.0`, `0.977865`
- `cube_push_success_rate`: `0.0`, `0.0`
- `Train/mean_reward`: `-392.534027`
- `Train/mean_episode_length`: `42.333332`

Post-run checks:

- No active Isaac/PPO/torchrun process remained.
- GPU memory returned to the pre-run baseline of about `2509MiB` used /
  `13436MiB` free.
- Disk remained about `30G` available.

## Interpretation

Wiring result:

- PASS. The D257 checkpoint loaded through `bc_teacher_checkpoint_path`.
- PASS. `cube_push_bc_teacher_blend_mean` was nonzero and equal to `1.0`.
- PASS. `cube_push_bc_teacher_imitation_mse` and imitation penalty were logged.
- PASS. The tiny smoke exited cleanly and released the process/GPU.

Behavior result:

- Not proven. The two-iteration smoke still had near-zero displacement and zero
  success.
- This cannot be treated as a learned policy, teacher-off success, or RoArm
  readiness.
- It also should not justify a longer PPO run by itself.

Critical hypothesis:

- The data-prior path is connected, but the teacher-on behavior may be limited by
  reset/feature distribution mismatch, phase timing, target trajectory feature
  mismatch, action saturation, or insufficient horizon.
- Since `bc_teacher_blend=1.0`, the environment action path is teacher-dominant;
  near-zero motion deserves a policy-free teacher rollout/feature-alignment
  probe before scaling PPO.

## Verdict

`D258_PPO_DATA_PRIOR_SMOKE_WIRING_PASS_BEHAVIOR_UNPROVEN`

## Next Step

Recommended next concrete action:

1. Do not launch longer PPO yet.
2. Add or run a teacher-only rollout/feature-alignment probe using the D257
   checkpoint with no PPO learning.
3. Verify:
   - teacher feature ranges versus D256 train-clean feature ranges;
   - phase alpha timing under `direct_steps`;
   - predicted joint deltas before/after env clamp;
   - TCP-to-cube distance evolution;
   - whether the teacher reaches contact before PPO learning is involved.
4. Only if teacher-on rollout produces plausible contact/reaction should a longer
   data-prior PPO run be considered.
