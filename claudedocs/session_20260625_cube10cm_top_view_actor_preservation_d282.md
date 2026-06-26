# Cube10cm Top-View PPO Actor-Preservation D282

Date: 2026-06-25 KST

## Scope

- Branch context: professor 10cm / 0.72kg cube top-view visual trajectory
  dataset branch.
- Track A, SmolVLA/VLA fine-tuning, RoArm deployment, B200/SSH, cleanup, and
  long PPO were not run.
- Goal: test whether the D280 actor/teacher bridge survives PPO updates before
  any longer PPO.

## Code Change

- Added `--actor_preserve_blend` to `roarm_rl/train_cube_push_ppo.py`.
- After warm-start/noise override, the runner snapshots actor-related state:
  `actor.*`, `actor_obs_normalizer.*`, `std`, and `log_std` when present.
- After each PPO update, the wrapper restores a blend:

```text
cur = (1 - actor_preserve_blend) * cur + actor_preserve_blend * ref
```

- This keeps the PPO runtime inside the existing `rsl_rl` runner and avoids
  editing third-party PPO internals.

## Protocol Correction

D281 initially evaluated saved checkpoints with `tap_useful_terminate` enabled.
That is too strict for frozen eval/trace summaries because successful episodes
can terminate, reset, and disappear from the final per-env summary.

Corrected saved-checkpoint protocol:

- `tap_contact_proxy_mode=link5_collision_aabb`
- D256 frame-0 reset, linspace sample mode
- `episode_length_s=6.0`
- `tap_stop_after_useful_seen=True`
- `vertical_gate_mode=min_contact`
- no `tap_useful_terminate`
- teacher-off eval uses `bc_teacher_blend=0.0`

Training runtime may still use `tap_useful_terminate`, but frozen eval and
actor-vs-teacher trace should normally omit it.

## D281 Correction

Re-evaluated the D281 conservative one-iteration checkpoint without useful
termination.

- Teacher-off eval:
  - verdict `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
  - useful/overshoot/joint-cap `0.8125/0.0/0.1666666716337204`
- Actor-vs-teacher trace:
  - verdict `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
  - MSE/cosine `0.04292111471295357/0.6536584496498108`
  - useful/overshoot `0.8125/0.0`

This corrects the prior interpretation: not every one-iteration PPO update is
unsafe. The stricter current claim is that every PPO candidate needs all three
gates: TensorBoard scalar gate, corrected teacher-off eval, and corrected
actor-vs-teacher trace.

## Actor-Freeze Smoke

Command class:

- one PPO iteration
- D280 actor warm-start
- `init_noise_std=0.1`
- conservative optimizer controls
- `actor_preserve_blend=1.0`

Runtime log:

```text
actor_preserve_after_update blend=1.000000 keys=13 max_pre_restore_delta=0.016347766 max_post_restore_delta=0.000000000
```

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke`

Results:

- TensorBoard gate: warning/manual-review only, no hard issue.
- Corrected teacher-off eval:
  - verdict `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
  - useful/overshoot/joint-cap `0.71875/0.0/0.2135416716337204`
- Corrected actor-vs-teacher trace:
  - verdict `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
  - MSE/cosine `0.05346343293786049/0.6641471982002258`
  - useful/overshoot `0.71875/0.0`

Interpretation: full actor preservation keeps the D280 actor behavior intact
while allowing the PPO runtime to exercise collection/update plumbing.

## Actor-Preserve095 Smoke

Command class:

- one PPO iteration
- D280 actor warm-start
- `init_noise_std=0.1`
- conservative optimizer controls
- `actor_preserve_blend=0.95`

Runtime log:

```text
actor_preserve_after_update blend=0.950000 keys=13 max_pre_restore_delta=0.016167104 max_post_restore_delta=0.000808358
```

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke`

Results:

- TensorBoard gate: warning/manual-review only, no hard issue.
- Corrected teacher-off eval:
  - verdict `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
  - useful/overshoot/joint-cap `0.71875/0.0/0.21875`
- Corrected actor-vs-teacher trace:
  - verdict `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
  - MSE/cosine `0.05328662693500519/0.6633936166763306`
  - useful/overshoot `0.71875/0.0`

Interpretation: 5% actor freedom for one iteration does not yet damage the
saved policy under the corrected gates.

## No-Preservation Conservative10 Smoke

Command class:

- 10 PPO iterations
- D280 actor warm-start
- `init_noise_std=0.1`
- conservative optimizer controls
- no actor preservation

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke`

TensorBoard result:

- verdict `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- issue: `joint-delta cap rate too high: max=0.6664496660232544`
- mean reward `-6.725722312927246 -> 5.878491401672363`
- useful last `0.03125`
- joint cap max `0.6664496660232544`
- action abs max last `0.9052953720092773`
- BC imitation MSE last `0.10183489322662354`
- raw TCP diagnostic last `0.5229541063308716`

Saved `model_9.pt` corrected teacher-off eval:

- verdict `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- useful/overshoot/joint-cap `0.65625/0.03125/0.2760416567325592`
- policy action abs max trace `2.2301270961761475`

Saved `model_9.pt` corrected actor-vs-teacher trace:

- verdict `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
- MSE/cosine `0.05501702427864075/0.6031392812728882`
- useful/overshoot `0.65625/0.03125`
- joint cap max `0.2760416567325592`

Interpretation: reward increased, but behavior did not promote. The actor hit
joint-delta limits too often, useful rate degraded during collection, and the
saved actor drifted enough from the teacher to block promotion.

## Decision

Verdict:

```text
D282_ACTOR_PRESERVATION_WIRED_1ITER_GATES_PASS_NO_PRESERVE_10ITER_FAIL_NO_LONG_PPO
```

Do not run long PPO from the no-preservation 10-iteration result.

The next valid runtime candidate is a short PPO with actor preservation enabled,
likely `--actor_preserve_blend 0.95`. It must pass:

1. TensorBoard scalar gate.
2. Corrected teacher-off frozen eval.
3. Corrected actor-vs-teacher trace.

No learned-policy claim, teacher-off final policy claim, RoArm-readiness claim,
or deployment claim exists.

## Validation

- `python -m py_compile roarm_rl/train_cube_push_ppo.py` passed after the actor
  preservation patch.
- `git diff --check -- roarm_rl/train_cube_push_ppo.py` passed after the patch.
- D282 Isaac Lab eval/trace processes exited cleanly.
- Post-run GPU check returned to baseline class state:
  `NVIDIA GeForce RTX 4090 Laptop GPU, 833 MiB used, 15111 MiB free, 0% util`.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md#d282`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `roarm_rl/train_cube_push_ppo.py`
- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
- `sim_scripts/cube10cm_top_view_actor_teacher_trace.py`
- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke/tensorboard_scalar_gate_d282_actor_freeze.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke/teacher_off_eval_after_actor_freeze_no_useful_term/teacher_off_policy_eval_summary_d282_after_actor_freeze_no_useful_term.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke/actor_teacher_trace_after_actor_freeze_no_useful_term/actor_teacher_trace_summary_d279.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke/tensorboard_scalar_gate_d282_actor_preserve095.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke/teacher_off_eval_after_actor_preserve095_no_useful_term/teacher_off_policy_eval_summary_d282_after_actor_preserve095_no_useful_term.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke/actor_teacher_trace_after_actor_preserve095_no_useful_term/actor_teacher_trace_summary_d279.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/tensorboard_scalar_gate_d282_conservative10.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/teacher_off_eval_model9_no_useful_term/teacher_off_policy_eval_summary_d282_conservative10_model9_no_useful_term.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/actor_teacher_trace_model9_no_useful_term/actor_teacher_trace_summary_d279.json`
