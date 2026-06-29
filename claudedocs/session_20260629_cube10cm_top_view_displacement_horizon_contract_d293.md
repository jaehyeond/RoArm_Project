# D293 Cube10cm Displacement/Horizon Contract

Date: 2026-06-29 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No Isaac Lab runtime, PPO training, render, cleanup, RunPod/B200/SSH, Track A,
SmolVLA/VLA fine-tuning, or RoArm deployment was performed in this decision.

## Question

D292 passed PPO plumbing, actor preservation, TensorBoard scalar extraction, and
saved-checkpoint teacher-off eval, but both TensorBoard and teacher-off eval
showed that displacement was too small to claim meaningful pushing. This session
checks whether the next step should still be long PPO or a stricter
displacement/horizon gate.

## Cross-Check

D292 current truth:

- TensorBoard hard issues: `[]`.
- TensorBoard warnings:
  - raw TCP-cube distance high under AABB diagnostics:
    `0.09063738584518433`;
  - tap max displacement tiny:
    `1.3096122529532295e-05m`.
- Saved-checkpoint teacher-off eval:
  - useful/success/overshoot: `0.96875 / 0.96875 / 0.0`;
  - D256 reset active / BC teacher blend: `1.0 / 0.0`;
  - max displacement along/XY:
    `0.0031909942626953125 / 0.00327563239261508m`;
  - mean displacement along/XY:
    `0.0001180088147521019 / 0.0001241182180820033m`.

D256 train-clean contract:

- clean episodes: `737`;
- contact/reaction: `737 / 737`;
- overshoot: `0`;
- `max_tap_disp_xy_m` min/p50/p90/p95/p99/max:
  `0.000671 / 0.005821 / 0.013904 / 0.016036 / 0.018031 / 0.019745`;
- current overshoot threshold: `0.020m`.

Interpretation:

- Contact/reaction is a necessary early primitive gate, but not enough for
  mining/excavation automation claims.
- D292 should not be promoted to long PPO because the actor was fully preserved
  and the mean displacement is below even the professor weak-reaction tier.
- The next runtime needs a hard displacement gate rather than another warning.

## Task Framing

The 10cm cube is not the final mining/excavation task. It is a
tool-object-interaction primitive:

1. contact: the tool reaches the object;
2. reaction: the object physically responds;
3. controlled displacement: the object moves by a meaningful amount;
4. no overshoot: the object is not over-pushed;
5. visual trajectory: the state can be captured as LeRobot-compatible data.

The correct next target is not "move the cube by many centimeters." The correct
next target is to promote from contact/reaction to a small but explicit
displacement tier.

## Displacement Tiers

- Tier 0: contact/reaction only, no overshoot.
- Tier 1: at least `0.001m` displacement.
- Tier 2: at least `0.003m` stable displacement.
- Tier 3: `0.005..0.010m` strong push tier.
- Fail: `>=0.020m` overshoot.

For the next PPO gate, use Tier 1 first:

- useful/contact/reaction rate: `>=0.90`;
- overshoot rate: `<=0.05`;
- D256 reset active: `1.0`;
- BC teacher blend: `0.0`;
- joint delta cap rate: low, with existing ceiling `<=0.25`;
- TensorBoard tap max displacement along: `>=0.001m`;
- teacher-off mean max displacement along or XY: target `>=0.0005m`,
  preferred `>=0.001m`;
- teacher-off max displacement along or XY: `>=0.001m`.

## Code Guardrail

Updated scripts:

- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - added `--require_tap_displacement_gate`;
  - with that flag, `Episode/cube_tap_max_disp_along_m` below
    `--min_tap_max_disp_along_m` becomes an issue, not a warning.
- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
  - added `--min_mean_disp_along_m`;
  - added `--min_max_disp_along_m`;
  - added `--min_mean_disp_xy_m`;
  - added `--min_max_disp_xy_m`.

Defaults preserve older behavior. The next gate must explicitly enable these
thresholds.

Posthoc regate check on D292:

- input:
  `actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it`;
- command enabled `--require_tap_displacement_gate` and
  `--min_tap_max_disp_along_m 0.001`;
- verdict:
  `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`;
- issue:
  `tap max displacement remains small: max=1.3096122529532295e-05`;
- artifact:
  `tensorboard_scalar_gate_d293_contract_regate.json`.

## Mass/Physical Spec

The current env uses:

- size: `0.100m`;
- mass: `0.720kg`;
- static/dynamic friction: `1.5 / 1.2`;
- restitution: `0.0`.

The 0.72kg mass is a coherent density-preserving diagnostic from the 3cm/20g
object, but it should not be treated as the only real-world nominal if the
physical proxy object is lighter.

Recommended physical spec:

- keep the 10cm rigid-box shape for top-view visibility and controlled contact;
- measure the real proxy object's mass;
- use measured real mass as nominal;
- later add mass robustness around `0.7x..1.3x`;
- keep `0.72kg` as a hard tier/stress case.

Mass randomization is not currently implemented for the tap10cm env; it is a
future sim2real robustness step, not a prerequisite for the next short PPO gate.

## Decision

Verdict:

`D293_DISPLACEMENT_HORIZON_CONTRACT_SET_NO_LONG_PPO`

Next concrete order:

1. Do not run long PPO.
2. Do not claim learned-policy or RoArm readiness from D292.
3. Treat D292 as a plumbing/checkpoint gate only.
4. Before the next runtime, use the updated hard displacement gates.
5. Next runtime, only after explicit approval, is a constrained short PPO gate:
   actor preservation on, D256 reset active, BC teacher blend off, AABB contact
   proxy, TensorBoard hard displacement gate, and saved-checkpoint teacher-off
   displacement gate.

## Verification

This decision should be verified with:

- `python -m py_compile sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
- `git diff --check`
