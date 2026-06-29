# D300 - Cube10cm collection-final TensorBoard gate

Date: 2026-06-29 KST

## Purpose

Cross-check D298/D299 against the TensorBoard gate semantics problem.
D299 showed that `tap_success_terminate=True` caused collection-time overshoot
through episode recycle. D300 tested the stricter follow-up: add final-state
TensorBoard scalars and gate on the final collection state instead of treating
the all-step collection average as a `0.90+` success metric.

## Code changes

- `roarm_rl/train_cube_push_ppo.py`
  - After `runner.learn(...)`, tap10cm runs now write
    `CollectionFinal/...` TensorBoard scalars from the current env buffers.
  - These scalars capture final collection state for contact, reaction, useful,
    success, overshoot, displacement, D256 reset activity, and joint cap.
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - Added `CollectionFinal/...` tags.
  - Added `--require_collection_final_tap_gate`.
  - Added final-state thresholds:
    `--min_collection_final_success_or_contact_rate`,
    `--min_collection_final_tap_useful_seen_rate`,
    `--min_collection_final_tap_success_rate`,
    `--max_collection_final_tap_overshoot_seen_rate`,
    `--min_collection_final_tap_disp_xy_ge_1mm_rate`.
  - Corrected final useful gating to use
    `CollectionFinal/cube_tap_useful_seen_rate` directly, not the max of useful
    and success.

## Runtime

Ran two explicitly approved tiny no-success-terminate actor-preserved PPO
re-gates:

- `seed=29801`
- `seed=29604`

Common contract:

- `tap10cm`
- `num_envs=32`
- `max_iterations=1`
- `num_steps_per_env=580`
- D256 random frame-0 reset active
- `tap_contact_proxy_mode=link5_collision_aabb`
- `tap_stop_after_disp_m=0.003`
- `tap_success_terminate=False`
- `bc_teacher_blend=0.0`
- `actor_preserve_blend=1.0`
- `init_noise_std=0.005`
- no long PPO, no PPO ladder, no partial actor preservation, no render, no
  cleanup, no RunPod/B200/SSH, no Track A/VLA/RoArm work

## Results

### Seed 29801

- PPO clean exit.
- Actor preservation:
  - `max_pre_restore_delta=0.228326231`
  - `max_post_restore_delta=0.000000000`
- checkpoint:
  - `model_0.pt`
  - sha256 `753df107215e434a421da8eb029f2daf8c028c0f33ab4b4be55d945511e6d971`
- collection-average TensorBoard:
  - useful `0.7658405303955078`
  - overshoot `0.0018318966031074524`
- collection-final TensorBoard:
  - contact/reaction `0.84375`
  - useful `0.8125`
  - success `0.84375`
  - overshoot `0.03125`
  - XY `>=1mm` `0.625`
  - mean/max XY `0.0037104845978319645/0.053734444081783295m`
- gate verdict:
  - `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
  - issues:
    - collection-final contact/reaction below `0.90`
    - collection-final useful below `0.90`

### Seed 29604

- PPO clean exit.
- Actor preservation:
  - `max_pre_restore_delta=0.241935968`
  - `max_post_restore_delta=0.000000000`
- checkpoint:
  - `model_0.pt`
  - sha256 `39e7080988517cab1ad017d9bc4f3ee69973eac351ae16ce6b583562d68eaf7b`
- collection-average TensorBoard:
  - useful `0.7738685607910156`
  - overshoot `0.0`
- collection-final TensorBoard:
  - contact/reaction `0.875`
  - useful `0.875`
  - success `0.875`
  - overshoot `0.0`
  - XY `>=1mm` `0.65625`
  - mean/max XY `0.0023780229967087507/0.006142078433185816m`
- gate verdict:
  - `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
  - issues:
    - collection-final contact/reaction below `0.90`
    - collection-final useful below `0.90`

## Interpretation

- D298 and D299 are consistent:
  - D298 failed mainly because success termination caused unsafe collection
    recycle.
  - D299/D300 no-success-terminate removes the major overshoot failure mode.
- The remaining blocker is not overshoot; it is final-state coverage.
  - Seed `29604` misses the final `0.90` useful gate by one env
    (`28/32 = 0.875`).
  - Seed `29801` misses by more and has one overshoot env
    (`1/32 = 0.03125`).
- Lowering the final useful gate to `0.85` would make seed `29604` look
  acceptable, but that would weaken the promotion standard. Keeping `0.90`
  is stricter and defensible for a promotion gate.
- This is still not learned-policy success:
  - actor was fully preserved;
  - Train reward scalars are missing under no-success-terminate;
  - no real actor update or partial actor preservation was validated.

## Decision

- Do not run long PPO.
- Do not run a PPO ladder.
- Do not use `tap_success_terminate=True` for this actor-preserved tap10cm
  collection gate.
- Do not claim learned-policy success, RoArm readiness, or mining automation
  readiness.
- Next work should be non-PPO final-coverage diagnostic:
  - identify which final envs miss contact/useful under D300;
  - inspect episode index, action magnitude, contact proxy, displacement, and
    overshoot for failed envs;
  - decide whether the issue is reset coverage, stop-after-disp hold behavior,
    or actor coverage.

## Verdict

`D300_COLLECTION_FINAL_TENSORBOARD_GATE_FAIL_NO_PROMOTION`

## Verification

- `python -m py_compile` passed for modified scripts.
- `git diff --check` passed after D300 documentation updates.
- D300 PPO commands exited cleanly.
- No Isaac/PPO/TensorBoard/torchrun process remained under the narrowed process
  filter.
- GPU was at `0%` utilization with the same baseline-class Python compute
  contexts visible (`1649MiB` used).

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `roarm_rl/train_cube_push_ppo.py`
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29801_1it/`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/`
