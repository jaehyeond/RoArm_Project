# D312 Cube10cm Baseline Perturbation Protocol

Date: 2026-07-06 KST

Purpose: test whether baseline controller v1 generalizes beyond the nominal corrected reset distribution. This is non-PPO. It is the required next failure-capable experiment before any PPO or tiny PPO trace gate.

## Baseline Controller V1

- `rl_action_mode="tap_push_primitive"`
- `primitive_speed_stop_min_disp_m=0.001`
- `tap_useful_min_disp_m=0.001`
- `exec_source=env_tap_push_primitive`
- corrected reset: `--no-env_hook_force_second_reset`
- objective: strict useful contact/reaction without overshoot and with max XY displacement `>=1mm`

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

## Rows

Run a small matrix first. Do not run a seed-only campaign.

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

Only after these rows should reset/contact geometry severity be increased.

## Decision Rules

- If nominal fails, stop and debug contract regression.
- If mild perturbation fails, freeze the failure as an RL/control target. Do not immediately add another hand condition.
- If all mild rows pass, increase severity or combine perturbations.
- If a row fails because strict useful drops below target while contact/reaction remains high, that is an action/displacement policy target.
- If a row fails because current proxy drops while useful stays latched, that is a contact-geometry/proxy contract target.
- If a row fails through overshoot, that is a post-contact stop/control target.

## Explicit Non-Goals

- No long PPO.
- No tiny PPO trace gate.
- No VLA/SmolVLA fine-tuning.
- No RoArm deployment claim.
- No POSCO/generalization claim from cube-only success.
- No additional nominal seed campaign whose result cannot change a decision.
