# Cube10cm Top-View D256 Reset Bin Actor Probe D286

Date: 2026-06-26 KST

## Scope

- Branch context: professor 10cm / 0.72kg cube top-view visual trajectory
  dataset branch.
- No long PPO, PPO training, render, cleanup, B200/SSH, RunPod runtime, RoArm
  deployment, or Track A work was run.
- Goal: diagnose the D285 collection failure before any additional PPO smoke.

## What Was Tested

D285 showed that even full actor freeze did not make PPO collection safe. The
failure mode was high `cube_push_joint_delta_cap_rate` during collection.

D286 tested two hypotheses:

1. The failure is mostly from specific bad D256 reset episode ranges.
2. The failure can be fixed by reducing action scale.

## Code Changes

- Added opt-in D256 reset episode filters:
  - `d256_reset_episode_min`;
  - `d256_reset_episode_max`.
- Exposed those filters in `roarm_rl/train_cube_push_ppo.py`.
- Added diagnostic script:
  `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`.

The diagnostic script is not PPO training. It reuses one Isaac Lab app, changes
the D256 episode filter per bin, and records frozen actor action magnitude,
joint-cap pressure, useful/contact trace max, overshoot trace max, and
actor-teacher sidecar agreement.

## Results

Actor checkpoint:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`

Default `action_scale=0.04`, `action_noise_std=0.02`, `580` steps:

- cap max by D256 episode bin:
  `0.6302083730697632 / 0.7604166865348816 /
  0.8229166865348816 / 0.703125 / 0.78125`
- useful max across all bins: `0.0`
- verdict: fail

Reduced `action_scale=0.01`, `action_noise_std=0.02`, `580` steps:

- cap max by D256 episode bin:
  `0.010416666977107525 / 0.015625 / 0.0052083334885537624 /
  0.0781250074505806 / 0.0833333358168602`
- useful max across all bins: `0.0`
- verdict: cap improves, behavior still fails

Comparison artifact:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d286_comparison/tap10cm/d256_reset_bin_actor_probe_comparison_d286.md`

## Interpretation

- Reset-bin filtering alone is not enough.
- Action-scale reduction alone is not enough.
- Default action scale keeps severe cap pressure and no useful signal.
- `action_scale=0.01` removes most cap pressure but still produces no useful or
  contact behavior.
- Therefore another D285 PPO smoke is not justified.

## TensorBoard

TensorBoard is still required for the next PPO smoke. It should be started live
against the PPO log directory so collection-time scalars are visible:

- reward;
- joint cap rate;
- action magnitude;
- useful/contact/reaction;
- overshoot;
- BC teacher MSE/blend/action magnitude.

But D286 did not run PPO, so no new TensorBoard dashboard was needed for this
diagnostic. The next TensorBoard run should happen only after a new PPO smoke
candidate exists.

## Verdict

```text
D286_NO_RESET_BIN_OR_ACTION_SCALE_FIX_READY_FOR_PPO
```

## Next Work

Do not run long PPO.

Next work should be a non-PPO fix/diagnostic pass:

1. Repair the actor/teacher bridge, or add an explicit action projection or
   action-cap/teacher constraint.
2. Rerun teacher-off frozen eval and D256 reset-bin diagnostics.
3. Only if those pass, run a tiny PPO smoke.
4. Start TensorBoard live for that PPO smoke and gate it before any longer PPO.

No learned-policy, teacher-off success, or RoArm-readiness claim exists.
