# D287 - Cube10cm Teacher/Action Contract Diagnostic

Date: 2026-06-26 KST

## Scope

- Track: professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.
- No Track A, SmolVLA/VLA fine-tuning, RoArm deployment, RunPod/B200/SSH, render, cleanup, long PPO, or PPO promotion.
- Purpose: repair the D286 diagnostic and decide whether the next step can be tiny PPO + TensorBoard, or must remain teacher/action-contract work.

## Code Changes

- `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`
  - reads env log scalars before terminate/reset can clear buffers;
  - supports `--exec_source actor|teacher|blend`;
  - uses zero-action warmup;
  - exposes `--bc_teacher_policy_delta_scale`;
  - exposes `--tap_stop_after_disp_m`;
  - exposes `--tap_contact_slowdown_use_proxy`.
- `roarm_rl/roarm_cube_push_env.py`
  - adds default-off `tap_stop_after_disp_m`;
  - adds default-off `tap_contact_slowdown_use_proxy`;
  - logs the new tap guard state.
- `roarm_rl/train_cube_push_ppo.py`
  - exposes the new default-off tap action-constraint flags for future tiny PPO smoke only after gates pass.

## Key Results

- D285 actor, deterministic, `max_joint_delta_per_step_rad=0.040`, 5 bins:
  - useful max range: `0.28125..0.5625`;
  - overshoot max range: `0.125..0.75`;
  - cap max: `0.0`.
- Teacher-only, `max_delta=0.040`, 5 bins:
  - useful max range: `0.21875..0.5625`;
  - overshoot max range: `0.15625..0.6875`;
  - cap max: `0.0`.
- Teacher-only, `20` bins / `8` envs, `max_delta=0.040`:
  - useful max range: `0.125..0.875`;
  - overshoot max range: `0.125..0.875`;
  - safe bins: `[]`.

## Failed Quick Fixes

- `bc_teacher_policy_delta_scale=0.5` did not create safe bins:
  useful `0.28125..0.5625`, overshoot `0.125..0.6875`.
- `tap_stop_after_disp_m=0.005` did not create safe bins:
  useful `0.25..0.5625`, overshoot `0.125..0.625`.
- `tap_contact_slowdown_use_proxy=True` did not create safe bins:
  useful `0.21875..0.5625`, overshoot `0.15625..0.6875`.

## Interpretation

- The old D286 useful-zero result was partly a diagnostic measurement artifact: terminate/reset could clear buffers before the script read them.
- After correction, actor and teacher both produce useful contact in some cases.
- However, teacher-only broader D256 reset coverage is not overshoot-safe.
- Therefore this is not only an actor/teacher bridge problem. The D257 teacher trajectory/action contract must be redesigned or constrained before actor distillation or PPO.

## Decision

- No long PPO.
- No tiny PPO + TensorBoard gate yet.
- No learned-policy, teacher-off success, or RoArm-readiness claim.
- Next concrete work:
  1. Compare D256 visual `clean_useful_tap` labels with current env overshoot semantics.
  2. Rebuild or constrain the teacher trajectory/action targets so teacher-only 20-bin coverage has safe bins.
  3. Only then rerun actor distillation / teacher-off bin diagnostics.
  4. Tiny PPO + TensorBoard gate comes after those gates pass.

## Validation

- `python -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_push_ppo.py sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`
- `git diff --check -- roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_push_ppo.py sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`
- Post-run GPU: RTX 4090 Laptop, `0%` util, about `14703 MiB` free.
- No remaining local Python/torchrun process.

## Primary Artifacts

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d287_maxdelta0040_deterministic_corrected/tap10cm/d256_reset_bin_actor_probe_summary_d286.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_teacher_probe_d287_20bins_maxdelta0040_corrected/tap10cm/d256_reset_bin_actor_probe_summary_d286.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_teacher_probe_d287_scale050_maxdelta0040_corrected/tap10cm/d256_reset_bin_actor_probe_summary_d286.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_teacher_probe_d287_stopdisp005_maxdelta0040_corrected/tap10cm/d256_reset_bin_actor_probe_summary_d286.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_teacher_probe_d287_proxy_slowdown_maxdelta0040_corrected/tap10cm/d256_reset_bin_actor_probe_summary_d286.json`
