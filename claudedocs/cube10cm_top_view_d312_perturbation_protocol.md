# D312 Cube10cm Baseline Perturbation Protocol

Date: 2026-07-06 KST

Purpose: test whether baseline controller v1 generalizes beyond the nominal corrected reset distribution. This is non-PPO. It is the required next failure-capable experiment before primitive-parameter PPO.

## Baseline Controller V1

- `rl_action_mode="tap_push_primitive"`
- `primitive_speed_stop_min_disp_m=0.001`
- `tap_useful_min_disp_m=0.001`
- `exec_source=env_tap_push_primitive`
- corrected reset: `--no-env_hook_force_second_reset`
- objective: strict useful contact/reaction without overshoot and with max XY displacement `>=1mm`
- optional controller-side cube pose noise: `--primitive_cube_pose_noise_xy_m`

## Required Metrics

Every row must report:

- strict useful/contact/reaction/final proxy
- current proxy at final/current diagnostic rows
- overshoot rate, target `<=5%`
- joint-delta cap mean/max, expected low
- max XY mean/max/min
- XY `>=1mm`, `>=3mm`, `>=7mm`, `>=20mm`
- primitive stop-latched rate and stop step min/max
- cube size/mass/static friction/dynamic friction from summary JSON
- controller-side cube pose noise magnitude and sampled noise mean/max

Metric/success computation must continue to use ground-truth cube state. Pose
noise is injected only into the controller's cube reference, so it represents
vision/perception error rather than changing the referee.

## Rows

Run the 9-row matrix. Do not run a seed-only campaign.

1. Nominal reproducibility row
   - cube size `0.10m`
   - mass `0.72kg`
   - static/dynamic friction `1.5/1.2`
   - one fresh32 random corrected-reset run

2. Size rows
   - `--cube_size_m 0.09`
   - `--cube_size_m 0.11`

3. Mass rows
   - `--cube_mass_kg 0.50`
   - `--cube_mass_kg 1.00`

4. Friction rows
   - low: `--cube_static_friction 0.8 --cube_dynamic_friction 0.6`
   - high: `--cube_static_friction 2.2 --cube_dynamic_friction 1.8`

5. Observation-noise rows
   - mild visual pose noise: `--primitive_cube_pose_noise_xy_m 0.005`
   - severe visual pose noise: `--primitive_cube_pose_noise_xy_m 0.015`

Escalation is capped at one round. If all 9 rows pass, run at most one combined
or harsher severity round, then start primitive-parameter PPO. Do not replace
the old seed ladder with an unlimited severity ladder.

## Decision Rules

- If nominal fails, stop and debug contract regression.
- If mild perturbation fails, freeze the failure as an RL/control target. Do not immediately add another hand condition.
- If all 9 rows pass, one severity escalation round is allowed, then PPO starts.
- If a row fails because strict useful drops below target while contact/reaction remains high, that is an action/displacement policy target.
- If a row fails because current proxy drops while useful stays latched, that is a contact-geometry/proxy contract target.
- If a row fails through overshoot, that is a post-contact stop/control target.

## PPO Start Trigger

Primitive-parameter PPO starts immediately after the 9-row matrix completes,
regardless of result.

- If baseline breaks: the failing axis becomes the RL proof stage.
- If baseline does not break: train on a randomized distribution using the
  perturbation axes as domain randomization, and evaluate on combined/severe
  rows.
- The first PPO target is primitive-parameter learning, not a return to raw
  scalar joint-delta PPO.

## Promotion Criteria V1

Use an affirmative promotion definition. Do not use `NO_PROMOTION` as a default
verdict when these checks have not been run.

A learned primitive-parameter checkpoint is a `PROMOTION_CANDIDATE` only if all
of the following hold:

- Friction `0.8/0.6`: strict useful `>=65%`, overshoot `<=5%`, and low-motion
  `<1mm <=10%`.
- Nominal row: strict useful `>=90%`, i.e. no nominal-regression evidence.
- Reproducibility: the criteria above hold across three different seeds.

If a checkpoint satisfies the friction row but fails the nominal row, record it
as a friction repair with nominal regression, not as a promotion candidate. If a
checkpoint satisfies only one seed, record it as a single-seed candidate and
continue exactly to the two remaining seeds; do not open a larger seed ladder.

## Reward V2 Spec

Reward v1 showed a two-sided failure: short PPO suppressed overshoot but left
low-motion cases, while longer PPO reduced low-motion and moved back toward
over-push. Reward v2 should remove the cliff-edge optimum:

- Reduce target-band/transient displacement pressure by using
  `tap_transient_disp_reward_scale=8.0` instead of `80.0`.
- Keep strict useful reward and overshoot penalties enabled.
- Make the control-band reward a dominant plateau reward over actual XY
  displacement `1..15mm`, not merely ">=1mm and no overshoot".
- Keep high-friction `2.2/1.8` isolated until render/trace audit resolves the
  12m runaway row.

## Explicit Non-Goals

- No long PPO.
- No tiny PPO trace gate before the 9-row perturbation matrix is complete.
- No VLA/SmolVLA fine-tuning.
- No RoArm deployment claim.
- No POSCO/generalization claim from cube-only success.
- No additional nominal seed campaign whose result cannot change a decision.
