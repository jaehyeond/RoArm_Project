# D272 cube10cm top-view tap10cm AABB data-prior PPO gate

Date: 2026-06-20 KST

Scope:

- Professor 10cm / 0.72kg cube top-view visual trajectory dataset branch only.
- No Track A, SmolVLA/VLA fine-tuning, RoArm deployment, B200, RunPod, or long
  PPO.
- Goal: verify the corrected `tap10cm + link5_collision_aabb` PPO data-prior
  wiring and TensorBoard observability after D270 restored the dataset contact
  contract.

## Code changes

- `roarm_rl/roarm_cube_push_env.py`
  - Tap reward now applies
    `bc_teacher_imitation_reward_scale * bc_teacher_imitation_mse` through
    `bc_imitation_penalty`.
  - Tap TensorBoard extras now include:
    - `cube_push_bc_teacher_blend_mean`;
    - `cube_push_bc_teacher_imitation_mse`;
    - `cube_push_bc_teacher_action_abs_mean`;
    - `cube_tap_bc_teacher_blend_mean`;
    - `cube_tap_bc_teacher_imitation_mse`;
    - `cube_tap_bc_teacher_action_abs_mean`;
    - `bc_teacher_imitation_penalty`.
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - Added `--env_kind {auto,push3cm,tap10cm}`.
  - Added `--expect_bc_teacher`.
  - Tap/AABB gate now treats raw TCP distance as diagnostic and uses tap contact,
    reaction, useful, displacement, vertical offset, overshoot, action cap, and
    BC teacher scalars as primary promotion signals.

## D271 regate

Existing D271 TensorBoard logs were re-read with the corrected gate:

- Output:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d270_logs/cube10cm_d270_tap10cm_aabb_data_prior_smoke/tensorboard_scalar_gate_d271_regated_after_d272_patch.json`
- Verdict:
  `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
- New gate correctly identified the old tap log visibility gap:
  - BC teacher blend scalar missing;
  - BC teacher imitation MSE scalar missing;
  - BC teacher action magnitude scalar missing.

This means D271 could not prove teacher-prior observability in TensorBoard even
though the command line requested the teacher prior.

## D272 tiny PPO smoke

Command shape:

- `env_kind=tap10cm`;
- `tap_contact_proxy_mode=link5_collision_aabb`;
- fixed push direction `+x`;
- `num_envs=32`;
- `max_iterations=2`;
- `num_steps_per_env=24`;
- `bc_teacher_checkpoint_path` set to the D257 checkpoint;
- `bc_teacher_blend=1.0`;
- `bc_teacher_imitation_reward_scale=5.0`;
- `bc_teacher_feature_target_mode=env_target`;
- `bc_teacher_phase_timing=direct_steps`.

Output root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke/`

Generated files:

- TensorBoard event file, `13216` bytes.
- `model_0.pt`, `1194495` bytes.
- `model_1.pt`, `1194495` bytes.
- `tensorboard_scalar_gate_d272.json`.
- `tensorboard_scalar_gate_d272.md`.

## Key scalars

Teacher-prior wiring/observability:

- `cube_push_bc_teacher_blend_mean`: `1.0 -> 1.0`.
- `cube_tap_bc_teacher_blend_mean`: `1.0 -> 1.0`.
- `cube_push_bc_teacher_imitation_mse`:
  `0.9571873545646667 -> 0.9571402072906494`.
- `cube_tap_bc_teacher_imitation_mse`:
  `0.9571873545646667 -> 0.9571402072906494`.
- `cube_push_bc_teacher_action_abs_mean`:
  `0.7122547030448914 -> 0.7180979251861572`.
- `cube_tap_bc_teacher_action_abs_mean`:
  `0.7122547030448914 -> 0.7180979251861572`.
- `bc_teacher_imitation_penalty`:
  `-4.7859368324279785 -> -4.785701751708984`.

Behavior:

- `cube_tap_contact_proxy_rate`: `0.0 -> 0.0`.
- `cube_tap_contact_seen_rate`: `0.0 -> 0.0`.
- `cube_tap_reaction_now_rate`: `0.0 -> 0.0`.
- `cube_tap_useful_seen_rate`: `0.0 -> 0.0`.
- `cube_tap_success_rate`: `0.0 -> 0.0`.
- `cube_tap_max_disp_along_m`: max `0.0005519219557754695`.
- `cube_tap_contact_vertical_offset_m`: last `0.1504015177488327`.
- `cube_tap_overshoot_seen_rate`: max `0.0651041716337204`.
- `cube_push_joint_delta_cap_rate`: max `0.330078125`.
- `Train/mean_reward`: `-24.44062042236328 -> -70.10071563720703`.
- `Loss/value_function`: `33301.984375 -> 62765.28515625`.

Gate result:

- `tensorboard_scalar_gate_d272` verdict:
  `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
- Issues:
  - no tap contact/reaction/useful signal;
  - tap useful/success absent;
  - overshoot seen rate too high;
  - joint-delta cap rate too high.

## Interpretation

D272 is a wiring/observability pass and a behavior fail.

It is not useful to run long PPO from this state. The teacher is definitely
loaded and blended, but default PPO resets place the tool far from the D256/AABB
contact-support distribution. This is visible in the `0.1504m` vertical offset,
zero contact/useful rates, and sub-1mm useful displacement.

This is not only an over-strict gate problem:

- Raw TCP distance is now only a warning on the tap/AABB branch.
- The failing primary metrics are the dataset-contract metrics: AABB contact,
  useful/reaction, displacement, overshoot, and action saturation.

The strongest causal clue is the contrast with D269:

- D269 D257 teacher-only from D256 initial reset had AABB contact/useful
  `0.71875/0.71875`.
- D272 default PPO reset with the same teacher prior had contact/useful
  `0.0/0.0`.

Therefore the next blocker is reset/pose-distribution alignment, not PPO scale.

## Verdict

`D272_TAP10CM_AABB_DATA_PRIOR_WIRING_VISIBLE_BEHAVIOR_FAIL_NO_PPO_PROMOTION`

## Next work

1. Add or probe a PPO reset path that samples D256 train-clean frame-0/state
   distribution rather than the current default reset.
2. Re-run teacher-only rollout with the same AABB/useful metrics from that reset.
3. Check feature support, phase alpha, clamp/action cap, vertical offset,
   displacement, overshoot, and TensorBoard BC teacher metrics.
4. Only after teacher-on contact/useful passes from the intended PPO reset
   distribution, run another tiny PPO smoke.
5. Do not claim learned policy, teacher-off success, or RoArm readiness until
   teacher-off eval passes.

## Verification

- `python -m py_compile` passed for the edited env, PPO entrypoint, TensorBoard
  gate, and probe scripts.
- `git diff --check` passed.
- D272 PPO smoke exited with code `0`.
- `ps -C python -C python3` showed no active local Python process after the run.
- `nvidia-smi` returned to the observed baseline of about `2509MiB` used.
