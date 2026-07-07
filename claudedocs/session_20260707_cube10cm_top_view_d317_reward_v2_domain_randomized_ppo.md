# 2026-07-07 - Cube10cm Top-View D317 Reward-v2 Domain-Randomized PPO

## Scope

- Branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset work.
- Purpose: define positive promotion criteria, cross-evaluate the D316 30-iteration checkpoint, document and apply reward v2, then run failure-capable primitive-parameter PPO on a low-friction randomized distribution.
- Exclusions: no B200/JHPark SSH, no pull, no `.ssh` copy, no RoArm deployment, no Track A, no VLA/SmolVLA fine-tuning, no raw joint-delta revival, no hand-written controller condition addition, and no high-friction training target.

## Repo Current Truth Checked First

- `CLAUDE.md` Current-State Protocol requires `START_HERE.md`, `claudedocs/DECISIONS.md`, `claudedocs/EXPERIMENT_LEDGER.md`, referenced session docs, `git status --short`, and metric verification from logs before citing.
- `CLAUDE.md` also requires at least one failure-capable experiment per research session, and says validation that cannot change a decision must not be run.
- `START_HERE.md` before this update said D316 was the latest current truth: D316 reward-v1 short PPO improved useful/overshoot but longer PPO reintroduced over-push.
- `claudedocs/DECISIONS.md` D314 records friction as the first baseline-breaking axis.
- `claudedocs/DECISIONS.md` D315 records the first real learnable primitive-residual PPO and its overshoot-heavy failure.
- `claudedocs/DECISIONS.md` D316 records reward-v1's short-run improvement and longer-run instability.
- Initial worktree check was clean on tracked files before D317 edits: `## master...origin/master`.

## Protocol Update

`claudedocs/cube10cm_top_view_d312_perturbation_protocol.md` now has an affirmative `Promotion Criteria V1`:

- Friction `0.8/0.6`: strict useful `>=65%`, overshoot `<=5%`, and low-motion `<1mm <=10%`.
- Nominal row: strict useful `>=90%`.
- Reproducibility: all criteria hold across three different seeds.
- A checkpoint satisfying all criteria is a `PROMOTION_CANDIDATE`; do not use a negative verdict as the default.

The same protocol now documents `Reward V2 Spec`:

- Reduce transient displacement pressure with `tap_transient_disp_reward_scale=8.0` instead of `80.0`.
- Keep strict useful rewards and overshoot penalties enabled.
- Make control-band reward a dominant plateau over actual XY displacement `1..15mm`.
- Keep high-friction `2.2/1.8` isolated until render/trace audit resolves the 12m runaway row.

## Code Changes

Default-off runtime and training knobs were added to support D317 without changing nominal defaults:

- Env runtime:
  - `policy_cube_pose_noise_xy_m`
  - `tap_control_band_max_disp_m`
  - `cube_friction_randomize_min`
  - `cube_friction_randomize_max`
  - `cube_dynamic_friction_ratio`
- `roarm_rl/train_cube_push_ppo.py` exposes matching CLI flags and validates non-negative/min-max contracts.
- `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py` can now evaluate `candidate8_diffik_target_residual` checkpoints with `policy_action_space=3` and policy-side cube pose noise.

No hand-written controller stop condition was added.

## Step 2: D316 model_29 Cross-Eval

Checkpoint:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d316/tap10cm/d316_candidate8_friction_low_reward_v1_30it/model_29.pt`

All rows used D290 fresh32 closed-loop recovery with `candidate8_diffik_target_residual`, `policy_action_space=3`, no preserve-blend shortcut, no useful/overshoot termination, and 580 steps.

| Row | Baseline v1 | D316 model_29 policy | Promotion Criteria V1 reading |
|---|---:|---:|---|
| Nominal strict useful | `32/32` | `32/32` | passes nominal no-regression row |
| Nominal overshoot | `0/32` | `0/32` | passes |
| Friction `0.8/0.6` strict useful | `10/32` | `4/32` | fails `>=65%` |
| Friction `0.8/0.6` overshoot | `1/32` | `28/32` | fails `<=5%` |
| Obs-noise `0.015m` strict useful | `32/32` | `32/32` | passes this stress row |
| Obs-noise `0.015m` overshoot | `0/32` | `0/32` | passes |

Key result:

- The D316 30-iteration policy looked good in its 64-step collection-final trace, but long-horizon D290 cross-eval exposes continued low-friction pushing: useful `4/32`, overshoot `28/32`, mean/max XY `30.706/41.887mm`.
- Therefore `model_29.pt` is not a promotion candidate under the positive criteria.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_promotion_cross_eval/tap10cm/nominal/closed_loop_recovery_summary_d317_cross_eval_model29_nominal.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_promotion_cross_eval/tap10cm/friction_0p8_0p6/closed_loop_recovery_summary_d317_cross_eval_model29_friction_0p8_0p6.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_promotion_cross_eval/tap10cm/obs_noise_0p015/closed_loop_recovery_summary_d317_cross_eval_model29_obs_noise_0p015.json`

## Step 4: Reward-v2 Domain-Randomized PPO

Runs:

- `d317_reward_v2_friction_uniform_seed31711`
- `d317_reward_v2_friction_uniform_seed31712`
- `d317_reward_v2_friction_uniform_seed31713`

Common training setup:

- Local GPU only.
- `rl_action_mode=candidate8_diffik_target_residual`
- `policy_action_space=3`
- `num_envs=64`
- `num_steps_per_env=64`
- `max_iterations=300`
- `save_interval=25`
- static friction randomized uniformly in `[0.7, 1.6]`
- dynamic friction ratio `0.8`
- no preserve-blend actor shortcut.

Reward v2 flags:

- `tap_contact_reward_scale=0.25`
- `tap_contact_proximity_reward_scale=0.2`
- `tap_reaction_reward_scale=2.0`
- `tap_transient_disp_reward_scale=8.0`
- `tap_overshoot_penalty_scale=24.0`
- `tap_overshoot_seen_penalty_scale=2.0`
- `tap_target_excess_quadratic_penalty_scale=1.0`
- `tap_strict_useful_reward_scale=8.0`
- `tap_strict_useful_seen_reward_scale=0.25`
- `tap_control_band_reward_scale=16.0`
- `tap_control_band_max_disp_m=0.015`
- `ppo_entropy_coef=0.002`
- `ppo_desired_kl=0.005`

### Collection-Final Results

| Seed | Contact/reaction | Strict useful | Overshoot | Low-motion `<1mm` | XY `>=20mm` | Max XY mean/max |
|---:|---:|---:|---:|---:|---:|---:|
| 31711 | `35/64` | `24/64` | `19/64` | `7/64` | `19/64` | `15.607/42.299mm` |
| 31712 | `34/64` | `19/64` | `12/64` | `12/64` | `12/64` | `12.190/33.771mm` |
| 31713 | `41/64` | `27/64` | `12/64` | `13/64` | `12/64` | `12.339/34.784mm` |

Reward v2 removed some low-motion compared with D316 short reward-v1, but contact/reaction and useful fell and overshoot remained too high. None of the three seeds approaches the friction-row promotion criteria.

### Curve Summary

TensorBoard scalar source: `Episode/cube_tap_useful_seen_rate`, with overshoot and low-motion read at the same peak-useful step.

| Seed | Peak iter | Peak useful | Peak overshoot | Peak low-motion | Final useful | Final overshoot | Final low-motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 31711 | `27` | `49.95%` | `12.23%` | `9.69%` | `35.35%` | `27.39%` | `11.18%` |
| 31712 | `239` | `50.07%` | `12.99%` | `8.76%` | `42.33%` | `19.65%` | `16.26%` |
| 31713 | `234` | `50.90%` | `12.55%` | `6.01%` | `38.38%` | `17.36%` | `17.75%` |

Even at the best useful point, peak useful is about `50%` and peak overshoot is about `12..13%`, so the positive promotion criteria are not close.

### Peak-Checkpoint D290 Cross-Eval

Representative peak checkpoints were evaluated with D290 fresh32 on friction `0.8/0.6`:

| Seed checkpoint | Strict useful | Contact/reaction | Overshoot | Mean/max XY |
|---|---:|---:|---:|---:|
| seed31711 `model_25.pt` | `4/32` | `32/32` | `28/32` | `32.599/41.639mm` |
| seed31712 `model_250.pt` | `4/32` | `32/32` | `28/32` | `29.253/38.881mm` |
| seed31713 `model_225.pt` | `2/32` | `32/32` | `30/32` | `30.776/35.921mm` |

This confirms the collection-horizon issue: policies can touch/react, but without learned stop/termination semantics they keep pushing in the 580-step probe.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d317/tap10cm/d317_reward_v2_friction_uniform_seed31711/collection_final_env_trace_iter_299.jsonl`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d317/tap10cm/d317_reward_v2_friction_uniform_seed31712/collection_final_env_trace_iter_299.jsonl`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d317/tap10cm/d317_reward_v2_friction_uniform_seed31713/collection_final_env_trace_iter_299.jsonl`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_reward_v2_peak_cross_eval/tap10cm/seed31711_model25_friction_0p8_0p6/closed_loop_recovery_summary_d317_v2_seed31711_model25_friction_0p8_0p6.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_reward_v2_peak_cross_eval/tap10cm/seed31712_model250_friction_0p8_0p6/closed_loop_recovery_summary_d317_v2_seed31712_model250_friction_0p8_0p6.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_reward_v2_peak_cross_eval/tap10cm/seed31713_model225_friction_0p8_0p6/closed_loop_recovery_summary_d317_v2_seed31713_model225_friction_0p8_0p6.json`

## Decision

- D317 satisfies the session progress rule with real failure-capable PPO updates on three seeds.
- Promotion Criteria V1 is now positive and explicit; D316 `model_29.pt` and D317 reward-v2 checkpoints fail it.
- Reward v2 did not solve the core low-friction long-horizon problem. It reduced the reward-v1 cliff pressure, but the learned residual still lacks a stable stop/termination behavior after contact.
- The next learning repair should add a learnable primitive termination/stop parameter or finite action chunk semantics, not another hand-coded controller condition and not a raw joint-delta rollback.
- The full every-25-checkpoint fresh32 sweep was not run after the peak checkpoints and collection curves already disqualified promotion; all 25-iteration checkpoints were saved and can be swept later if a decision-relevant reason appears.

## Verdict

`D317_REWARD_V2_DOMAIN_RANDOMIZED_PPO_OVERSHOOT_FAIL`

