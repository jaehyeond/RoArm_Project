# D315 Candidate8 Primitive-Residual PPO Start

Date: 2026-07-07 KST

Scope: professor 10cm / 0.72kg cube top-view branch after the D314 9-row perturbation matrix. This session started primitive-residual PPO after the matrix, as required by D313. It did not run long PPO, RoArm deployment, Track A, VLA/SmolVLA fine-tuning, B200/SSH, pull, or `.ssh` copy.

## Why Not Train `tap_push_primitive`

Code inspection showed `rl_action_mode="tap_push_primitive"` is a baseline controller mode, not a learnable primitive-parameter policy mode:

- In `roarm_rl/roarm_cube_push_env.py`, the mode stores `policy_actions` for logging but computes `targets = self._tap_push_primitive_joint_target()` and applies those targets directly.
- Therefore PPO under `tap_push_primitive` would update a policy whose action does not change the executed target.

## Code Change

Updated `roarm_rl/train_cube_push_ppo.py` so PPO can run a learnable non-joint action space:

- Added `--rl_action_mode` for `joint_delta`, `candidate6_diffik_residual_joint`, and `candidate8_diffik_target_residual`.
- Intentionally did not expose `tap_push_primitive` as a train mode.
- Added `--policy_action_space`; `candidate8_diffik_target_residual` auto-sets/requires action space `3`.
- Added cube size/mass/friction overrides so D314 failure axes can become training axes.
- Added candidate6/candidate8 DiffIK parameter overrides and logging.
- Fixed training logs so action semantics no longer always claim `normalized_joint_delta action_dim=6`.

## First Primitive-Residual PPO Run

Run:

- `d315_candidate8_friction_low_5it`
- env: `RoArm-CubeTap10cm-Direct-v0`
- action mode: `candidate8_diffik_target_residual`
- action space: `3`
- friction: static/dynamic `0.8/0.6`
- envs/steps/iterations: `64` / `64` / `5`
- seed: `31415`
- candidate6 base: `legacy_far_face_through`, `previous_joint_target`, `start_pose`, goal `0.003m`, push steps `220`, step clip `0.010rad`

Artifacts:

- run dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d315/tap10cm/d315_candidate8_friction_low_5it`
- final trace: `collection_final_env_trace_iter_4.jsonl`
- final checkpoint: `model_4.pt`
- summary: `d315_candidate8_friction_low_5it_summary.json`
- `model_4.pt` sha256: `44b6fe43f61c5f0bdf2e859134479bec322a282c026e1580348004ccd3204114`
- trace sha256: `43f72a1e72ccafcf295e6cffc5ef9a21b0e37e5235e71e54935d5173750bce98`

## Final Collection Metrics

Final trace, 64 envs:

- contact/reaction: `64/64` / `64/64`
- useful: `32/64`
- overshoot: `32/64`
- current contact proxy: `62/64`
- XY `>=1mm`: `64/64`
- XY `>=3mm`: `63/64`
- XY `>=7mm`: `56/64`
- XY `>=20mm`: `32/64`
- max XY mean/min/max: `27.589/1.802/55.642mm`
- joint-delta cap mean/max: `0.0/0.0`

## Interpretation

This is not a learned-policy promotion. It proves the first post-matrix PPO path can execute real updates with a learnable primitive-residual action head. The behavior is still poor: it over-pushes heavily under low friction. The next PPO work should reduce overshoot while preserving the recovered contact/reaction coverage, not return to raw scalar joint deltas and not add another hand-written controller patch before analyzing this failure.

## Verdict

`D315_CANDIDATE8_FRICTION_LOW_PPO_STARTED_OVERSHOOT_FAIL_NO_PROMOTION`
