# D297 Cube10cm Teacher-Off Reset-Protocol Re-Audit

Date: 2026-06-29 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No PPO training, long PPO, render, cleanup, RunPod/B200/SSH, Track A,
SmolVLA/VLA fine-tuning, or RoArm deployment was performed.

## Question

D296 said random D256 reset sampling failed under several action constraints.
Before treating that as a real actor/action failure, D297 asks whether the
teacher-off eval reset protocol itself caused the overshoot.

The key distinction:

- `direct_reset`: call `env.reset()` and evaluate from the D256 reset state.
- forced second reset: call `env.reset()`, force `episode_length_buf` to max,
  then step once to trigger another reset path before evaluation.

Only `direct_reset` is now the default teacher-off promotion gate.

## Code Changes

Updated `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py` with:

- per-env CSV output;
- per-step/env action CSV output;
- reset-alignment CSV output;
- recorded actions, recovery actions, actor actions, row/frame/episode indices
  in the saved diagnostic dataset;
- actor-vs-recorded and actor-vs-recovery metrics;
- `--env_hook_force_second_reset` so the old reset path can be isolated.

Updated `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` with:

- `--d256_reset_warmup_mode direct_reset|force_step_zero|force_step_policy`;
- default `direct_reset`;
- old forced-step behavior preserved only as explicit diagnostics.

## Diagnostic Results

Exact D296 failing episodes under manual reset:

- episodes: `339,154,198,668,736,656,195,606`;
- useful `1.0`;
- overshoot `0.0`;
- mean/max XY `0.0012649/0.0032778m`;
- actor-vs-recorded MSE/cosine `0.105531/0.681824`.

Random env-hook seed `29604` with the forced second reset path:

- useful `0.8125`;
- overshoot `0.15625`;
- mean/max XY `0.0067425/0.0531604m`.

Manual replay of the same env-hook-selected 32 episodes:

- useful `0.96875`;
- overshoot `0.0`;
- mean/max XY `0.0020878/0.0167372m`.

Env-hook seed `29604` with direct reset and no forced second reset:

- useful `1.0`;
- overshoot `0.0`;
- mean/max XY `0.0010001/0.0034951m`.

Reset-alignment audit for the failing forced-second-reset path:

- cube actual-vs-expected XY error mean/max `0.0/0.0m`;
- cube start-vs-expected XY error mean/max `0.0/0.0m`;
- arm actual-vs-expected max error mean/max `0.0/0.0rad`;
- arm target-vs-expected max error mean/max `0.0/0.0rad`;
- arm joint velocity mean/max `0.0/0.0rad/s`;
- cube linear velocity mean/max `0.0/0.0m/s`;
- cube angular velocity mean/max `0.0/0.0rad/s`.

Interpretation:

- D256 labels were not the immediate cause.
- Actor actions were not different between manual and env-hook variants.
- Reset pose, target, and velocity alignment were not different.
- The observed failure is tied to the forced second reset path/contact-cache
  behavior in the eval protocol.

## Corrected Teacher-Off Gate

Common contract:

- checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- `num_envs=32`, `eval_steps=580`;
- D256 reset active, `d256_reset_sample_mode=random`, frame `0`;
- `d256_reset_warmup_mode=direct_reset`;
- BC teacher off: `bc_teacher_blend=0.0`;
- contact proxy: `link5_collision_aabb`;
- `tap_stop_after_disp_m=0.003`;
- gate: useful `>=0.90`, overshoot `<=0.05`, mean XY `>=0.0005m`,
  max XY `>=0.001m`, XY `>=1mm` rate `>=0.25`.

Seed `29603`:

- verdict `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
- useful `1.0`;
- overshoot `0.0`;
- mean/max XY `0.0020230/0.0116959m`;
- XY `>=1mm` / `>=3mm` rates `0.53125/0.4375`;
- mean/max along `0.0018544/0.0116892m`;
- along `>=1mm` / `>=3mm` rates `0.5/0.3125`;
- joint cap max trace `0.0`;
- D256 reset active `1.0`;
- BC teacher blend last `0.0`;
- issues `[]`.

Seed `29604`:

- verdict `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
- useful `1.0`;
- overshoot `0.0`;
- mean/max XY `0.0011871/0.0040057m`;
- XY `>=1mm` / `>=3mm` rates `0.375/0.3125`;
- mean/max along `0.0011399/0.0040054m`;
- along `>=1mm` / `>=3mm` rates `0.34375/0.25`;
- joint cap max trace `0.0`;
- D256 reset active `1.0`;
- BC teacher blend last `0.0`;
- issues `[]`.

## Decision

Verdict:

`D297_TEACHER_OFF_DIRECT_RESET_GATE_PASS_NO_PPO`

Meaning:

- D296's old random-reset overshoot is superseded as a teacher-off eval
  reset-protocol artifact.
- This is not a learned-policy claim.
- This is not RoArm readiness.
- Long PPO remains blocked.
- A PPO ladder, partial actor preservation, and real PPO actor updates remain
  blocked.
- The next valid runtime is one explicitly approved tiny PPO + TensorBoard gate
  using the corrected direct-reset teacher-off contract.

## Next Runtime Contract

Only after explicit approval:

1. Run one tiny PPO + TensorBoard gate, not a long PPO.
2. Keep D256 reset active and random reset rechecks.
3. Keep BC teacher blend off.
4. Keep actor preservation on.
5. Keep `link5_collision_aabb` as the contact proxy.
6. Enforce D293/D294 max/mean/rate displacement gates.
7. After the saved checkpoint is produced, run teacher-off eval with
   `--d256_reset_warmup_mode direct_reset`.
8. Do not promote learned-policy or RoArm readiness unless teacher-off frozen
   eval passes and TensorBoard gates pass.

## Artifacts

- D297 root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/`
- Exact D296 fail episode diagnostic:
  `closed_loop_recovery_summary_d297_fail8_actor_action_diagnostic.json`
- Forced-second-reset reproduction:
  `random_envhook_seed29604/closed_loop_recovery_summary_d297_random_envhook_seed29604_actor_action_diagnostic.json`
- Direct-reset env-hook pass:
  `random_envhook_direct_seed29604/closed_loop_recovery_summary_d297_random_envhook_direct_seed29604_actor_action_diagnostic.json`
- Reset alignment audit:
  `reset_alignment_envhook_seed29604_vel/reset_alignment_envhook_seed29604_vel_d297.csv`
- Corrected teacher-off seed `29603`:
  `teacher_off_direct_seed29603/teacher_off_policy_eval_summary_d297_direct_seed29603.json`
- Corrected teacher-off seed `29604`:
  `teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d297_direct_seed29604.json`
- Updated scripts:
  `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
  `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
