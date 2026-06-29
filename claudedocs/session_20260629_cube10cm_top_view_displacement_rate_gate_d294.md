# D294 Cube10cm Displacement Rate Gate

Date: 2026-06-29 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No Isaac Lab runtime, PPO training, render, cleanup, RunPod/B200/SSH, Track A,
SmolVLA/VLA fine-tuning, or RoArm deployment was performed in this decision.

## Question

D293 correctly made displacement a hard next-gate concept, but max displacement
alone can overstate progress if only one or two envs move. The next gate should
combine max, mean, and per-env displacement-rate evidence.

## Cross-Check

D256 train-clean positive distribution from
`label_package_d248/episode_split_manifest.csv`:

- rows: `737`;
- `max_tap_disp_xy_m >= 0.001`: `733/737`, rate `0.994572592`;
- `max_tap_disp_xy_m >= 0.003`: `727/737`, rate `0.986431479`;
- `max_tap_disp_along_m >= 0.001`: `729/737`, rate `0.989145183`;
- `max_tap_disp_along_m >= 0.003`: `723/737`, rate `0.981004071`.

Interpretation:

- A 1mm displacement-rate gate is not too strict relative to the D256 clean
  contract.
- Mean displacement remains necessary because rate can still hide very small
  margins above threshold.
- Max displacement remains useful only as a sanity check that no single-env
  outlier is being mistaken for distribution-level success.

## Code Guardrail

Updated runtime env logging:

- `roarm_rl/roarm_cube_push_env.py`
  - logs `cube_tap_max_disp_along_ge_1mm_rate`;
  - logs `cube_tap_max_disp_xy_ge_1mm_rate`;
  - logs `cube_tap_max_disp_along_ge_3mm_rate`;
  - logs `cube_tap_max_disp_xy_ge_3mm_rate`.

Updated TensorBoard gate:

- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - reads the new 1mm/3mm rate scalars;
  - added `--min_tap_disp_along_ge_1mm_rate`;
  - added `--min_tap_disp_xy_ge_1mm_rate`;
  - if either threshold is positive, missing scalar is an issue.

Updated teacher-off eval:

- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
  - records along/XY `>=1mm` and `>=3mm` rates;
  - added `--min_disp_along_ge_1mm_rate`;
  - added `--min_disp_xy_ge_1mm_rate`.

## Next Gate Contract

The next constrained short PPO gate should not pass from max displacement alone.
Use all three displacement views:

1. max displacement: confirms at least some physical movement occurred;
2. mean displacement: rejects single-env-only movement;
3. `>=1mm` rate: checks distribution-level movement.

Initial next-gate recommendation:

- TensorBoard:
  - `--require_tap_displacement_gate`;
  - `--min_tap_max_disp_along_m 0.001`;
  - `--min_tap_disp_xy_ge_1mm_rate 0.25` for the short collection gate.
- Teacher-off frozen eval:
  - `--min_mean_disp_xy_m 0.0005`;
  - `--min_max_disp_xy_m 0.001`;
  - `--min_disp_xy_ge_1mm_rate 0.25` as the first hard rate gate;
  - raise toward D256-like coverage only after short-gate stability is proven.

This `0.25` rate is intentionally conservative for the first short PPO gate
because the collection horizon may still be short. It is not the final research
target. D256 clean data supports much higher rates, so later gates should move
toward `0.90+` once the runtime is stable.

## Actor-Preservation Caveat

Actor preservation remains a plumbing/safety mechanism. A run with full or heavy
actor preservation can validate that PPO did not destroy the actor and that
metrics are observable, but it is not a learned-policy claim. Only after the
preserved-actor gate passes should partial preservation or real PPO actor
updates be discussed.

## Decision

Verdict:

`D294_DISPLACEMENT_RATE_GATE_ADDED_NO_RUNTIME_NO_LONG_PPO`

Next concrete order:

1. Do not run long PPO.
2. Do not claim learned-policy or RoArm readiness from D292/D293/D294.
3. Next runtime, only after explicit approval, is one constrained short PPO gate
   with max/mean/rate displacement gates active.
4. If the preserved-actor gate passes the max/mean/rate contract, then consider
   partial actor preservation or a very small real PPO actor-update gate.

## Verification

This decision should be verified with:

- `python -m py_compile roarm_rl/roarm_cube_push_env.py sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
- `git diff --check`
