# D299 - Cube10cm collection contract no-success-terminate diagnostic

Date: 2026-06-29 KST

## Scope

- Diagnosed D298's mismatch between PPO collection-time TensorBoard failure and
  saved-checkpoint teacher-off direct-reset pass.
- Added diagnostic options to the teacher-off evaluator:
  - `--tap_success_terminate`;
  - `--action_mode inference|ppo_stochastic`;
  - RSL-like pre-reset `extras["log"]` aggregation;
  - done/reset trace metrics.
- Ran non-PPO collection-time diagnostics, then one tiny actor-preserved PPO
  re-gate with `tap_success_terminate=False`.
- No long PPO, PPO ladder, partial actor preservation, render, cleanup,
  RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm deployment was performed.

## Diagnostic finding

D298 used `tap_success_terminate=True`, while the earlier teacher-off
direct-reset eval did not. D299 isolated that difference.

| diagnostic | action mode | success terminate | done count | final useful | final overshoot | RSL-like useful mean | RSL-like overshoot mean | max XY |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d299_inference_success_term_seed29801` | inference | true | 45 | 0.09375 | 0.875 | 0.0933728448275862 | 0.7933728448275862 | 0.05490714684128761 |
| `d299_ppo_like_stochastic_success_term_seed29801` | ppo_stochastic | true | 38 | 0.0 | 0.84375 | 0.0027478448275862067 | 0.7538793103448276 | 13.797537803649902 |
| `d299_ppo_like_stochastic_no_term_seed29801` | ppo_stochastic | false | 0 | 1.0 | 0.0 | 0.8638469827586207 | 0.0 | 0.007077273912727833 |
| `d299_ppo_like_stochastic_no_term_seed29604` | ppo_stochastic | false | 0 | 0.84375 | 0.0 | 0.6825969827586207 | 0.0 | 0.013731294311583042 |

Interpretation:

- `tap_success_terminate=True` is the immediate cause of D298's overshoot
  explosion under collection/recycle.
- Stochastic PPO-style action sampling is not sufficient by itself to cause the
  explosion: with success termination off, overshoot stayed `0.0` in both
  tested seeds.
- The no-termination path still has useful coverage limits. It is safer, but
  not a learned-policy success claim.
- TensorBoard useful mean is an all-step collection average. It includes early
  pre-contact approach frames, so `0.90` is too strict as a collection-average
  smoke threshold. Teacher-off final useful/overshoot remains the stricter
  checkpoint gate.

## PPO re-gate

- Command:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/ppo_command_d299.txt`
- Output root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it`
- PPO exit: clean, exit code `0`.
- Saved checkpoint:
  `model_0.pt`
- Checkpoint sha256:
  `753df107215e434a421da8eb029f2daf8c028c0f33ab4b4be55d945511e6d971`
- Actor preservation:
  `actor_preserve_blend=1.0`, `max_post_restore_delta=0.000000000`.
- Runtime contract:
  - D256 reset active, random sample mode;
  - `bc_teacher_blend=0.0`;
  - `bc_teacher_imitation_reward_scale=0.0`;
  - `tap_success_terminate=False`;
  - `tap_stop_after_disp_m=0.003`;
  - `link5_collision_aabb` contact proxy.

## TensorBoard gate

- Gate artifact:
  `tensorboard_scalar_gate_d299.json`
- Verdict:
  `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- Issues: none.
- Warnings:
  - no-termination gate allowed missing `Train/mean_reward` and
    `Train/mean_episode_length`;
  - raw TCP distance remained high, as expected for AABB/tap diagnostics.
- Key tap scalars:
  - contact/reaction seen: `0.7676724195480347`;
  - useful seen: `0.7658405303955078`;
  - success: `0.7676724195480347`;
  - overshoot seen: `0.0018318966031074524`;
  - max displacement along/XY:
    `0.001473818439990282/0.0016653359634801745m`;
  - along/XY `>=1mm`:
    `0.40420258045196533/0.4132543206214905`;
  - D256 reset active: `1.0`;
  - BC teacher blend: `0.0`;
  - joint cap: `0.0`.

Compared with D298 TensorBoard:

- useful improved from `0.04482758790254593` to `0.7658405303955078`;
- overshoot improved from `0.7133082151412964` to
  `0.0018318966031074524`;
- max XY decreased from `0.03478653356432915m` to
  `0.0016653359634801745m`.

## Saved-checkpoint teacher-off direct-reset eval

- Seed `29801`:
  - verdict:
    `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
  - useful: `0.96875`;
  - overshoot: `0.03125`;
  - mean/max XY displacement:
    `0.004003090318292379/0.06263629347085953m`;
  - XY `>=1mm`: `0.5625`;
  - joint cap max trace: `0.0`.
- Seed `29604`:
  - verdict:
    `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
  - useful: `1.0`;
  - overshoot: `0.0`;
  - mean/max XY displacement:
    `0.0011870721355080605/0.004005730152130127m`;
  - XY `>=1mm`: `0.375`;
  - joint cap max trace: `0.0`.

## Decision

- D299 fixes the D298 collection-time overshoot failure mode by removing
  `tap_success_terminate`.
- D299 is still not learned-policy success:
  - actor was fully preserved;
  - no-termination TensorBoard lacks completed-episode reward scalars;
  - collection useful mean is improved but not `0.90+`;
  - teacher-off eval is still deterministic/inference mode.
- Do not run long PPO.
- Do not use `tap_success_terminate=True` for this actor-preserved tap10cm
  collection gate.
- Next work should be a short controlled follow-up, not a PPO ladder:
  1. decide whether the next PPO smoke should keep no-termination and use a
     collection-average useful threshold around `0.65..0.75`;
  2. add or use a final-state TensorBoard metric if a `0.90+` useful gate is
     desired;
  3. only then consider a second tiny no-success-terminate gate with more seeds
     or slightly longer horizon.

## Verification

- `python -m py_compile` passed for modified scripts.
- `git diff --check` passed.
- No Isaac/PPO/TensorBoard/torchrun process remained under the narrowed process
  filter.
- GPU was at `0%` utilization with the same baseline-class Python compute
  contexts visible (`1649MiB` used).

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/collection_contract_d299/`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/`
