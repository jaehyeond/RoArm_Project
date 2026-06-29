# D295 Cube10cm Rate-Gated Short PPO Runtime

Date: 2026-06-29 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No Track A, SmolVLA/VLA fine-tuning, RoArm deployment, RunPod/B200/SSH, render,
cleanup, or long PPO was performed.

## Question

D294 added max/mean/rate displacement gates. D295 asks whether the D290
replay-batch actor can pass one constrained short PPO gate when the collection
horizon is long enough to observe physical displacement.

## Runtime Contract

- Runtime: one short PPO gate only.
- PPO iterations: `1`.
- Collection: `num_envs=32`, `num_steps_per_env=580`.
- Actor prior:
  `actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`.
- Actor preservation: `actor_preserve_blend=1.0`.
- D256 reset: active, frame `0`, `linspace`.
- BC teacher: off, `bc_teacher_blend=0.0`, imitation reward `0.0`.
- Contact proxy: `link5_collision_aabb`.
- Action contract:
  - `action_scale=0.04`;
  - `max_joint_delta_per_step_rad=0.04`;
  - `joint_target_lead_limit_rad=0.06`;
  - `joint_delta_reference=joint_pos`.

Command:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/ppo_command_d295.txt`

## PPO Runtime Result

Output root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/`

Artifacts:

- `model_0.pt`
- `events.out.tfevents.1782664164.cgxr-Legion-Pro-7-16IRX9H.269456.0`
- `tensorboard_scalar_gate_d295.json`
- `tensorboard_scalar_gate_d295.md`
- `tensorboard_dashboard_command_d295.txt`
- `teacher_off_eval_model0_d295_contract/teacher_off_policy_eval_summary_d295_model0.json`
- `teacher_off_eval_model0_d295_contract/teacher_off_policy_eval_summary_d295_model0.md`
- `teacher_off_eval_model0_d295_contract/teacher_off_policy_eval_steps_d295_model0.csv`

The PPO process exited cleanly. Actor preservation restored the actor exactly:

- `max_pre_restore_delta=0.270150483`
- `max_post_restore_delta=0.000000000`

`model_0.pt` sha256:

`d3073e7446652d6a7c7c6a160c336bfa7cdf8bf04ef988010adf6bd79b322b0a`

## TensorBoard Gate

Verdict:

`TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`

Issues:

- missing core TensorBoard scalars: `Train/mean_reward`,
  `Train/mean_episode_length`;
- tap contact/reaction/useful below the D293/D294 promotion threshold:
  max contact-like scalar `0.8786637783050537`;
- tap useful/success below threshold:
  max `0.8786637783050537`.

Warnings:

- short run: `Train/mean_reward` has `0` points;
- raw TCP-cube distance high for tap/AABB diagnostic:
  `0.12133253365755081`.

Key collection scalars:

| Metric | Value |
| --- | ---: |
| D256 reset active | `1.0` |
| BC teacher blend | `0.0` |
| contact / reaction / useful / success | `0.8786637783 / 0.8786637783 / 0.8710668087 / 0.8786637783` |
| overshoot | `0.0075969826` |
| max disp along / XY | `0.0025365781 / 0.0026646142m` |
| along / XY `>=1mm` rate | `0.3124461174 / 0.3125` |
| XY `>=3mm` rate | `0.2603987157` |
| joint cap / target lead limit | `0.0 / 0.0` |

Interpretation:

- D294 rate logging works.
- The 580-step horizon shows real displacement; this fixes the D292
  short-horizon tiny-displacement ambiguity.
- The collection gate still fails because useful/contact/reaction is below
  `0.90` and core Train scalars are missing from TensorBoard.

Dashboard command, if manual TensorBoard inspection is needed:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/tensorboard_dashboard_command_d295.txt`

## Teacher-Off Frozen Eval

This eval used the saved `model_0.pt`, BC teacher off, D256 reset active, and the
D295 action contract. It did not use the D292 useful-stop/zero-action safety
wrapper, so it tests raw frozen actor behavior under the D295 collection
contract.

Verdict:

`TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`

Issues:

- useful seen rate below threshold: `0.8125`;
- overshoot seen rate too high: `0.1875`.

Key metrics:

| Metric | Value |
| --- | ---: |
| D256 reset active / BC blend | `1.0 / 0.0` |
| contact / reaction / useful / success | `1.0 / 1.0 / 0.8125 / 1.0` |
| overshoot | `0.1875` |
| max disp along / XY | `0.0590043068 / 0.0590082631m` |
| mean disp along / XY | `0.0074102012 / 0.0106040258m` |
| along / XY `>=1mm` rate | `0.53125 / 0.6875` |
| XY `>=3mm` rate | `0.625` |
| joint cap max trace | `0.0` |
| policy action abs mean / max | `0.3058707444 / 1.0` |
| vertical gate value | `0.0` |

Interpretation:

- The actor can move the cube by meaningful distances when the horizon is long
  enough.
- That movement is not controlled enough: overshoot rises to `18.75%`.
- Therefore D295 is not a learned-policy success and must not promote to
  partial-preservation PPO or long PPO.

## Decision

Verdict:

`D295_RATE_GATED_SHORT_PPO_COLLECTION_PARTIAL_TEACHER_OFF_FAIL_NO_PROMOTION`

Next concrete order:

1. Do not run long PPO.
2. Do not claim learned-policy success or RoArm readiness.
3. Do not promote to partial actor preservation yet.
4. Next work should be a non-PPO overshoot-control diagnostic:
   compare raw actor eval against explicit action projection/constraint options
   such as `tap_stop_after_disp_m`, useful-stop, and proxy contact slowdown.
5. Only if max/mean/rate displacement and overshoot pass together should
   partial preservation or real PPO actor updates be reconsidered.

## Verification

- PPO command exit code: `0`.
- TensorBoard scalar gate exit code: `0`, verdict fail.
- Teacher-off eval exit code: `0`, verdict fail.
- TensorBoard dashboard command recorded; no long-running TensorBoard server was
  left active.
- No learned-policy or RoArm readiness claim exists.
