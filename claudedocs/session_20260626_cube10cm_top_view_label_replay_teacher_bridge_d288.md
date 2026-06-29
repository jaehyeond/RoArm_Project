# D288 Cube10cm Top-View Label/Replay/Teacher Bridge Audit

Date: 2026-06-26 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory dataset branch only. No long PPO, no PPO training, no render, no RunPod/B200, no RoArm control, no Track A.

## Question

D287 showed that the corrected teacher/action contract was not overshoot-safe. D288 checked whether that was because the D256 visual labels were too loose, because the live env cannot replay the data, or because the D257 MLP teacher is not a safe closed-loop teacher.

## Results

### 1. D256 label/env contract audit

Script:

- `sim_scripts/cube10cm_top_view_d256_label_env_contract_audit.py`

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_label_env_contract_audit_d288/d256_label_env_contract_audit_d288.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_label_env_contract_audit_d288/d256_label_env_contract_audit_d288.md`

Key numbers:

- `train_clean_positive`: 737 episodes.
- train clean contact/reaction: `737 / 737`.
- train clean overshoot episodes: `0`.
- train clean `max_tap_disp_xy_m` min/p50/p90/p95/p99/max:
  `0.000671 / 0.005821 / 0.013904 / 0.016036 / 0.018031 / 0.019745`.
- current env overshoot threshold: `0.020 m`.
- train clean episodes with `max_xy >= 0.020m`: `0`.
- `eval_overshoot_diagnostic`: 167 episodes, all overshoot.
- eval overshoot `max_tap_disp_xy_m` min/p50/p90/p95/p99/max:
  `0.020069 / 0.070663 / 0.162419 / 0.203046 / 0.258294 / 0.264307`.

Verdict:

- `D288_LABEL_CLEAN_TEACHER_ONLINE_CONTRACT_MISMATCH_CONFIRMED`.

Interpretation:

- The D256 train-clean labels are not permissive about overshoot. They are clean under the same 0.020m XY displacement threshold.
- The D287 teacher/actor overshoot is an online action/teacher execution problem, not a loose-label problem.

### 2. D256 recorded-action replay

Script:

- `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py`

Outputs:

- `d256_action_replay_summary_d288_link5_8env_steps580.json`
- `d256_action_replay_summary_d288_link5_32env_steps580.json`

32-env result:

- contact rate: `1.0`.
- useful rate: `1.0`.
- TCP-threshold contact rate: `0.0` because the old point TCP metric is not the current AABB proxy.
- max XY displacement mean/min/max:
  `0.007403433322906494 / 0.000011587240805965848 / 0.017222005873918533`.
- max along displacement mean/min/max:
  `0.006767723709344864 / 0.000009298324584960938 / 0.017127275466918945`.
- max target jump abs mean/max:
  `0.06703907251358032 / 0.09352636337280273`.

Interpretation:

- The live env can replay D256 recorded joint targets cleanly under `link5_collision_aabb`.
- The problem is not that D256 reset states or live physics are inherently impossible.

### 3. Teacher/action contract fixes tested

Code changes:

- Added `bc_teacher_phase_timing=linear_episode|linear_steps`.
- Added `bc_teacher_linear_phase_steps`.
- Added `joint_delta_reference` to the D256 reset-bin probe and defaulted that probe to `joint_pos`, matching D256 target semantics.
- Added optional teacher comparison to the D256 action replay probe.

Tested no-PPO teacher-only bin probes:

1. `linear_steps`, old default `target` reference:
   - safe bins: `[]`.
   - overshoot max by 5 bins: `0.1875 / 0.75 / 0.65625 / 0.65625 / 0.5`.

2. `joint_pos`, `direct_steps`:
   - safe bins: `[]`.
   - overshoot max by 5 bins: `0.15625 / 0.65625 / 0.59375 / 0.65625 / 0.4375`.

3. `joint_pos`, `linear_steps`:
   - safe bins: `[]`.
   - overshoot max by 5 bins: `0.125 / 0.71875 / 0.6875 / 0.5625 / 0.53125`.

4. `joint_pos`, `linear_steps`, `bc_teacher_policy_delta_scale=0.5`:
   - safe bins: `[]`.
   - overshoot max by 5 bins: `0.125 / 0.5 / 0.6875 / 0.625 / 0.46875`.

Interpretation:

- `joint_pos` is the correct D256 action reference and should remain in later diagnostics.
- Linear phase alignment and lower teacher scale are not enough.
- The D257 MLP teacher is not a safe closed-loop teacher over D256 reset states.

### 4. Teacher comparison during clean recorded replay

Output:

- `d256_action_replay_summary_d288_link5_32env_steps580_teacher_compare.json`
- `d256_action_replay_summary_d288_link5_32env_steps580_teacher_compare_smooth100.json`

With `linear_steps`, `joint_pos`, scale `1.0`, smoothing `0.85`:

- replay useful/max XY: `1.0 / 0.017222005873918533`.
- teacher vs recorded delta MSE/MAE/cosine:
  `0.00013523723706308327 / 0.005776729541911005 / 0.8618926034148398`.
- teacher vs live-needed delta MSE/MAE/cosine:
  `0.00012767873194851793 / 0.005706412376482682 / 0.8641119014343311`.
- teacher action abs mean/max:
  `0.23752034487973514 / 1.0`.

With smoothing disabled (`bc_teacher_delta_smoothing_alpha=1.0`):

- metrics were essentially unchanged.

Interpretation:

- The MLP teacher is close enough offline but not exact enough for closed-loop use.
- A mean action error around `0.0057 rad` can accumulate across hundreds of steps.
- Smoothing is not the main cause.

## Decision

Do not run long PPO.

Do not run tiny PPO + TensorBoard gate yet from D257/D288 MLP-teacher execution.

The next valid work is to replace the teacher/action bridge before PPO:

1. Use D256 recorded-action replay as the safe teacher baseline.
2. Build actor warm-start/distillation from recorded D256 actions or a replay teacher, not from the current closed-loop MLP teacher alone.
3. Re-run teacher-off/bin diagnostics under:
   - `joint_delta_reference=joint_pos`;
   - `tap_contact_proxy_mode=link5_collision_aabb`;
   - D256 frame-0 reset;
   - no teacher blend at eval.
4. Only if teacher-off/bin diagnostics pass, run tiny PPO smoke plus TensorBoard scalar gate.

Verdict:

`D288_RECORDED_REPLAY_CLEAN_MLP_TEACHER_CLOSED_LOOP_UNSAFE_NO_PPO`

