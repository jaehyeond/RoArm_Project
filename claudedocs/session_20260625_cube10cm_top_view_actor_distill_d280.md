# D280 Cube10cm Top-view Actor Distillation / Warm-start PPO Smoke

Date: 2026-06-25 KST

## Scope

- Track: professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.
- Contract: `tap10cm + link5_collision_aabb + D256 frame-0 reset + fixed +x + episode_length_s=6.0`.
- Goal: fix the actor/teacher bridge before any longer PPO.
- Not done: long PPO, RoArm readiness claim, RunPod/B200 work, cleanup, Track A.

## Code Changes

- Added supervised actor distillation:
  `sim_scripts/cube10cm_top_view_distill_actor_from_teacher.py`.
- Added PPO actor warm-start CLI:
  `roarm_rl/train_cube_push_ppo.py --warm_start_checkpoint_path`.
- Added PPO tap termination flags:
  `--tap_success_terminate`, `--tap_overshoot_terminate`.
- Added teacher-off diagnostic knobs:
  `--zero_actions_after_useful_seen`, `--vertical_gate_mode`,
  `--action_scale`, `--max_joint_delta_per_step_rad`.

## Distillation Result

- Output checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
- sha256:
  `4c12862320883ebaab14c97043999e235224a5d892916d6a23f16189358639dd`
- Train/val samples: `16704` / `1856`
- Initial val MSE/MAE/cosine:
  `0.38865897059440613` / `0.5184221863746643` / `0.32961708307266235`
- Final val MSE/MAE/cosine:
  `0.01078740879893303` / `0.0625312477350235` / `0.9815400838851929`
- Verdict:
  `D280_ACTOR_DISTILL_SUPERVISED_FIT_WARN_NEEDS_ROLLOUT_EVAL`
- Warning reason:
  teacher rollout used for collection still had overshoot `0.21875`.

## Rollout Checks

- Distilled actor D279-style trace:
  - MSE/cosine: `0.0765833854675293` / `0.8944697976112366`
  - useful: `0.59375`
  - overshoot: `0.125`
  - vertical max: `0.22511835396289825`
  - joint cap max: `0.7604166865348816`
  - verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`

- Default teacher-off eval:
  - useful: `0.59375`
  - overshoot: `0.125`
  - joint cap max: `0.7604166865348816`
  - verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`

- Action-scale probes:
  - `action_scale=0.020`: useful `0.5625`, overshoot `0.125`
  - `action_scale=0.010`: useful `0.46875`, overshoot `0.15625`
  - conclusion: action scale alone is not the fix.

## Stop-after-useful Diagnostic

- Probe:
  `--zero_actions_after_useful_seen --vertical_gate_mode min_contact`
- Result:
  - verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
  - useful: `0.71875`
  - overshoot: `0.0`
  - joint cap max: `0.2135416716337204`
  - vertical gate value: `0.0`
- Interpretation:
  the remaining blocker is mainly stop-after-useful semantics and final-frame
  vertical-gate strictness, not only actor/teacher mismatch.

## Tiny PPO Smoke

- Ran one 1-iteration warm-start PPO smoke, not a training ladder.
- Runtime:
  - D280 actor warm-start loaded.
  - D256 reset active.
  - AABB contact proxy active.
  - `bc_teacher_blend=0.0`.
  - BC imitation reward scale `0.05`.
  - `tap_success_terminate=True`.
  - `tap_overshoot_terminate=True`.
- Output:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/ppo_warmstart_smoke/cube10cm_d280_warmstart_success_terminate_smoke/`
- TensorBoard gate:
  - verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
  - issue: `joint-delta cap rate too high: max=0.3993055820465088`
  - useful: `0.12109375`
  - overshoot: `0.0221354179084301`
  - BC imitation MSE logged: `0.46044886112213135`
  - D256 reset active: `1.0`

- Trace after PPO smoke:
  - MSE/cosine: `0.086099773645401` / `0.8869514465332031`
  - useful: `0.5`
  - overshoot: `0.1875`
  - joint cap max: `0.78125`
  - verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`

## Decision

- D280 improved actor/teacher alignment but did not produce a promotable policy.
- The tiny PPO runtime is wired, but the gate fails and the 1-iteration PPO update worsens rollout trace.
- Do not run long PPO or a PPO ladder from D280.
- Do not claim learned policy, teacher-off success, or RoArm readiness.
- Next work:
  encode stop-after-useful semantics in the env/reward/termination contract,
  use contact-time vertical gate for tap/reaction evaluation, then rerun
  teacher-off eval, actor-vs-teacher trace, and only then another tiny PPO
  TensorBoard gate.
