# D290 Cube10cm D256 Replay-Batch Actor Diagnostic

Date: 2026-06-27 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No long PPO, tiny PPO, TensorBoard PPO gate, render, RoArm deployment,
RunPod/B200, cleanup, or Track A work was run.

## Objective

Close the D289 next step:

1. collect D256 replay-action actor datasets in separate single-batch Isaac
   processes;
2. merge the batches and train a broader supervised actor warm-start;
3. run teacher-off and D256 reset-bin diagnostics before any PPO smoke.

## Code Changes

- `sim_scripts/cube10cm_top_view_distill_actor_from_d256_replay.py`
  now supports one-batch dataset export through `--dataset_out` and
  `--dataset_only`, plus explicit `--episode_indices`.
- Added `sim_scripts/cube10cm_top_view_train_actor_from_replay_batches.py`
  for offline supervised actor training from separately collected replay
  batches.
- Added `sim_scripts/cube10cm_top_view_d290_offline_actor_batch_diagnostic.py`
  to verify checkpoint load/inference against saved replay batches through the
  same RSL inference policy path used by rollout probes.
- Added `--exec_action_clip_abs` to:
  - `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`;
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`.

## Separate-Process Replay Batch Collection

Collected five separate-process single-batch datasets under:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_replay_actor_batches_d290/`

Each batch used 32 envs, 580 steps, `action_scale=0.04`,
`max_joint_delta_per_step_rad=0.04`, `joint_delta_reference=joint_pos`, and
`tap_contact_proxy_mode=link5_collision_aabb`.

| batch | episode range | samples | useful | overshoot | max XY m |
|---|---:|---:|---:|---:|---:|
| 00 | 1-208 | 18560 | 1.0 | 0.0 | 0.009117063 |
| 01 | 209-370 | 18560 | 1.0 | 0.0 | 0.015542421 |
| 02 | 371-537 | 18560 | 1.0 | 0.0 | 0.016205473 |
| 03 | 538-715 | 18560 | 1.0 | 0.0 | 0.016418220 |
| 04 | 716-999 | 18560 | 1.0 | 0.0 | 0.011478594 |

Interpretation: the D256 recorded-action path remains physically clean when
replayed through the normal policy action path. D256 labels/replay are not the
immediate blocker.

## Actor Training

Merged dataset size: 92,800 samples.

Primary checkpoint:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`

Training summary:

- 155 epochs from D285 actor-freeze source;
- train/val samples: 83,520 / 9,280;
- final val MSE/cosine: `0.004836` / `0.970317`;
- verdict: `D290_D256_REPLAY_BATCH_ACTOR_TRAIN_PASS_NEEDS_ROLLOUT_EVAL`.

Offline checkpoint-load diagnostic through the RSL inference path passed:

- aggregate samples: 92,800;
- aggregate MSE/MAE/cosine: `0.0047070058062672615` /
  `0.030751453712582588` / `0.9694145321846008`;
- verdict: `D290_OFFLINE_ACTOR_BATCH_DIAGNOSTIC_PASS`.

This confirms the checkpoint load/inference contract is not the cause of the
closed-loop failures.

## Teacher-Off Eval

Teacher-off frozen eval passed for one 32-env rollout:

- checkpoint: D290 `tap10cm_ep155`;
- useful: `0.96875`;
- overshoot: `0.0`;
- max XY max: `0.00327563239261508m`;
- joint cap max trace: `0.015625`;
- D256 reset active rate: `1.0`;
- verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`.

This is a narrow pass only. It is not a learned-policy success claim and does
not justify long PPO by itself.

## Reset-Bin Diagnostic

The same actor failed D256 reset-bin coverage:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d290_replay_batch_actor_5bins/d256_reset_bin_actor_probe_summary_d286.json`

| bin | episode range | useful max | overshoot max | cap max | max XY m | raw policy max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1-208 | 0.84375 | 0.15625 | 0.067708 | 0.005763 | 5.810399 |
| 1 | 209-370 | 0.21875 | 0.78125 | 0.25 | 0.027875 | 9.951535 |
| 2 | 371-537 | 0.5 | 0.6875 | 0.171875 | 0.365942 | 45.684681 |
| 3 | 538-715 | 0.4375 | 0.625 | 0.151042 | 0.371026 | 36.781395 |
| 4 | 716-999 | 0.25 | 0.625 | 0.192708 | 0.021318 | 8.439145 |

Verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`.

## Negative Controls

The following did not fix the reset-bin failure:

- `--exec_action_clip_abs 0.5`: cap max became `0.0` in all bins, but
  overshoot remained high: `0.15625 / 0.8125 / 0.78125 / 0.71875 / 0.8125`.
- `--exec_action_clip_abs 0.5 --tap_stop_after_disp_m 0.015`: no material
  improvement over clip-only.
- `--contact_joint_delta_scale 0.35 --tap_contact_slowdown_use_proxy`:
  cap pressure reduced but overshoot stayed high or worsened.
- Low-LR supervised refinement from the D290 actor improved offline val MSE to
  `0.004294`, but reset-bin still failed with overshoot
  `0.15625 / 0.875 / 0.5625 / 0.5 / 0.75`.

## Diagnosis

D290 resolves the D289 collection problem but does not solve closed-loop policy
stability.

The actor fits replay-batch observations well offline, but under closed-loop
execution it compounds small action errors, leaves the D256 replay state
manifold, and then produces saturated or misdirected actions. This is
imitation-error compounding, not a D256 replay-label failure and not a simple
joint-cap/action-scale problem.

## Decision

- Do not run long PPO.
- Do not run tiny PPO + TensorBoard gate yet.
- Do not claim learned policy success, broad teacher-off success, or RoArm
  readiness.
- Next work is DAgger-style closed-loop data aggregation:
  collect actor-rollout states under the D256 reset contract, label them with a
  recovery oracle tied to the recorded D256 episode/time action sequence or a
  safer constrained replay-teacher, retrain the actor, then rerun:
  1. offline actor-batch diagnostic;
  2. teacher-off frozen eval;
  3. D256 reset-bin diagnostic.

Verdict:

`D290_REPLAY_BATCH_ACTOR_OFFLINE_PASS_TEACHER_OFF_NARROW_PASS_RESET_BIN_FAIL_CLOSED_LOOP_DAGGER_NEXT_NO_PPO`
