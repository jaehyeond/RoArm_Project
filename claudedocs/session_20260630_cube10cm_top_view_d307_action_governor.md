# Session 2026-06-30 - Cube10cm D307 Action Governor Diagnostic

## Scope

- Track: professor 10cm / 0.72kg cube top-view visual trajectory branch.
- Purpose: follow D306 without PPO by testing a displacement/velocity-aware
  action governor on the D304 failed6 cases.
- Not run: long PPO, tiny PPO trace gate, PPO ladder, partial actor
  preservation, real actor update, render, cleanup, RunPod/B200/SSH, Track A,
  VLA fine-tuning, or RoArm deployment.

## Code Change

- Added default-off action governor options to
  `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`.
- Modes:
  - `off`: existing actor execution.
  - `predict_stop`: stop before projected displacement exceeds a target.
  - `predict_brake`: short opposite-action brake before zero hold.
- The governor uses current displacement, cube speed, contact state, and a
  projected displacement horizon. It is a non-PPO diagnostic/prototype only.

## Runtime Contract

- Actor base: D306 candidate-2
  `phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- Runtime action contract:
  - `max_joint_delta_per_step_rad=0.04`
  - `contact_joint_delta_scale=0.35`
  - `fast_cube_joint_delta_scale=0.2`
  - `action_smoothing_alpha=0.25`
  - `joint_delta_reference=joint_pos`
  - `tap_contact_proxy_mode=link5_collision_aabb`
  - `tap_stop_after_disp_m=0.003`
  - no useful/overshoot terminate

## Results

### Candidate-2 + Predict Stop

| Probe | Useful | Overshoot | Cap max | Mean XY | Max XY | Notes |
|---|---:|---:|---:|---:|---:|---|
| ep561 `h=0.060`, `v=0.060` | 1.0 | 0.0 | 0.0 | 0.000696m | 0.000696m | safe but below 1mm |
| ep561 `h=0.020`, `v=0.200` | 1.0 | 0.0 | 0.0 | 0.004996m | 0.004996m | fixed the D306 overshoot bracket for this episode |
| failed6 `h=0.020`, `v=0.200` | 1.0 | 0.0 | 0.0 | 0.002727m | 0.007170m | partial: 4/6 envs reached >=1mm, 2/6 stayed tiny |

Failed6 per-env displacement for the best governor setting:

| Episode | Max XY | Max Along | Stop Step | Interpretation |
|---:|---:|---:|---:|---|
| 561 | 7.170mm | 7.165mm | 474 | controlled push, no overshoot |
| 265 | 1.929mm | 1.929mm | 387 | Tier-1 pass |
| 341 | 2.159mm | 2.157mm | 335 | Tier-1 pass |
| 991 | 0.023mm | 0.019mm | -1 | action/direction geometry still fails |
| 536 | 5.051mm | 5.051mm | 455 | controlled push, no overshoot |
| 29 | 0.027mm | 0.023mm | -1 | action/direction geometry still fails |

Rates:

- XY >= 1mm: `0.666667`
- XY >= 3mm: `0.333333`
- Along >= 1mm: `0.666667`
- Along >= 3mm: `0.333333`

### Recorded-Target Repair Attempt

- Built a D307 recorded-target dataset from the governor failed6 closed-loop
  states:
  `recorded_target_dataset/phase_action_repair_dataset_d307_failed6_recorded_target_exact.pt`
- Fine-tuned from D306 candidate-2 for 80 epochs on D304 failed6 replay plus
  the D307 recorded-target dataset.
- Checkpoint:
  `recorded_repair_lr5e5_ep80/model_actor_d256_replay_batches_d290.pt`
- sha256:
  `2d2bc75c30c0fb2241bf7a6230cc2513abac6a9a3ccfe5a7fd769479f4a1fa60`
- Offline final val MSE/cosine:
  `0.0305119734` / `0.8834095001`
- Runtime failed6 + same governor collapsed displacement:
  - useful `1.0`
  - overshoot `0.0`
  - cap max `0.0`
  - mean/max XY `0.0000154146m` / `0.0000227652m`
  - XY >= 1mm: `0.0`

## Interpretation

- D307 improves the D306 bracket for overshoot-heavy cases: ep561 moved from
  candidate-2's `41.5mm` overshoot to `5.0mm` without overshoot.
- The governor is not sufficient for all failed cases. Episodes `991` and `29`
  receive full actions and still do not create meaningful displacement, so
  those are action-direction/contact-geometry failures, not just late stopping
  failures.
- Recorded-target supervised repair improved offline loss but collapsed runtime
  displacement. Offline actor-vs-recorded metrics are still not enough as a
  promotion criterion.

## Verdict

`D307_ACTION_GOVERNOR_PARTIAL_NO_PPO_PROMOTION`

## Decision

- D307 is not a PPO gate and not a learned-policy/RoArm-readiness claim.
- Do not run long PPO, tiny PPO trace gate, PPO ladder, partial actor
  preservation, or real actor update from D307.
- Next work should be non-PPO deployable action-space/control repair:
  - either put the displacement/velocity action governor into the env as a
    default-off runtime contract and test broader fresh resets;
  - or change the action representation so the actor outputs a tool/object
    push primitive instead of brittle scalar joint deltas.
- Promotion requires fresh multi-episode coverage with useful/contact high,
  overshoot <=5%, cap low, and >=1mm displacement rate meeting the D293/D294
  displacement contract.

