# D292 Cube10cm Top-View Tiny PPO Fresh-Gate Smoke

Date: 2026-06-28 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No Track A, SmolVLA/VLA fine-tuning, RoArm deployment, RunPod/B200/SSH, render,
cleanup, long PPO, or learned-policy claim was performed.

## Question

D291 showed that the D290 same-process reset-bin failure was likely a reused-env
diagnostic artifact. D292 asks whether the D290 replay-batch actor can be wired
through one tiny Isaac Lab PPO smoke with TensorBoard scalar gating and a saved
teacher-off checkpoint eval, without letting PPO damage the actor.

## Runtime Contract

- Runtime: one tiny PPO smoke only.
- Actor prior:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- Env: `tap10cm`, D256 reset hook enabled, `d256_reset_sample_mode=linspace`.
- PPO collection: `num_envs=32`, `num_steps_per_env=24`, `max_iterations=1`.
- Actor preservation: `--actor_preserve_blend 1.0`.
- PPO noise: `--init_noise_std 0.005`.
- Action contract:
  - `action_scale=0.04`
  - `max_joint_delta_per_step_rad=0.04`
  - `joint_target_lead_limit_rad=0.06`
  - `joint_delta_reference=joint_pos`
  - `tap_contact_proxy_mode=link5_collision_aabb`
- D257 MLP teacher path is not used:
  - `bc_teacher_checkpoint_path=NONE`
  - `bc_teacher_blend=0.0`
  - `bc_teacher_imitation_reward_scale=0.0`

## PPO Result

Output root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it/`

Artifacts:

- `model_0.pt`
- `events.out.tfevents.1782624133.cgxr-Legion-Pro-7-16IRX9H.3693017.0`
- `tensorboard_scalar_gate_d292.json`
- `tensorboard_scalar_gate_d292.md`
- `teacher_off_eval_model0/teacher_off_policy_eval_summary_d292_model0.json`
- `teacher_off_eval_model0/teacher_off_policy_eval_summary_d292_model0.md`
- `teacher_off_eval_model0/teacher_off_policy_eval_steps_d292_model0.csv`

The PPO process exited cleanly. Actor preservation restored the actor exactly:

- `max_pre_restore_delta=0.017272532`
- `max_post_restore_delta=0.000000000`

`model_0.pt` sha256:

`d56065796c2549bfc70c7d2200314118b924580e1d38f19a8265ee2c8aebf271`

## TensorBoard Scalar Gate

Verdict:

`TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`

Hard issues:

- none

Warnings:

- raw TCP-cube distance is high for tap/AABB diagnostics:
  `0.09063738584518433`
- tap max displacement remains small:
  `1.3096122529532295e-05m`

Key scalars from the one-iteration PPO collection:

| Metric | Value |
| --- | ---: |
| `Train/mean_reward` | `-8.49217414855957` |
| `Train/mean_episode_length` | `1.6521738767623901` |
| `Episode/cube_push_d256_reset_active_rate` | `1.0` |
| `Episode/cube_push_bc_teacher_blend_mean` | `0.0` |
| `Episode/cube_push_bc_teacher_imitation_mse` | `0.0` |
| `Episode/cube_push_joint_delta_cap_rate` | `0.0008680556202307343` |
| `Episode/cube_push_target_lead_limit_rate` | `0.0` |
| `Episode/cube_tap_useful_seen_rate` | `0.0768229216337204` |
| `Episode/cube_tap_success_rate` | `0.0729166716337204` |
| `Episode/cube_tap_overshoot_seen_rate` | `0.01302083395421505` |
| `Episode/cube_tap_max_disp_along_m` | `0.000013096122529532295` |

Dashboard command:

```bash
conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it --host 127.0.0.1 --port 6006
```

## Saved Checkpoint Teacher-Off Eval

The saved `model_0.pt` was evaluated without BC teacher action blending.

Verdict:

`TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`

Key metrics:

| Metric | Value |
| --- | ---: |
| eval steps/envs | `580 / 32` |
| D256 reset active rate | `1.0` |
| BC teacher blend last | `0.0` |
| useful/success/overshoot | `0.96875 / 0.96875 / 0.0` |
| contact/reaction seen | `0.96875 / 0.96875` |
| joint delta cap max trace | `0.015625` |
| policy action abs mean trace mean | `0.017667093202632305` |
| policy action abs max trace max | `1.0` |
| max disp along / XY | `0.0031909942626953125 / 0.00327563239261508m` |
| mean disp along / XY | `0.0001180088147521019 / 0.0001241182180820033m` |
| min-contact vertical offset max | `0.0m` |
| tcp-threshold contact seen rate | `0.0` |

The teacher-off eval used `tap_stop_after_useful_seen` and
`zero_actions_after_useful_seen`, but did not use `tap_useful_terminate`.

## Interpretation

D292 is a runtime/plumbing pass:

- PPO loads the D290 replay-batch actor.
- D256 env reset is active.
- The D257 MLP teacher is disabled.
- Actor preservation prevents the one PPO update from changing the actor.
- TensorBoard scalar extraction works.
- The saved checkpoint still passes teacher-off frozen eval.

D292 is not a learned-policy success:

- `actor_preserve_blend=1.0` intentionally restores the actor after PPO update.
- TensorBoard returned manual-review warnings, not a clean promotion verdict.
- The displacement numbers are small, especially mean displacement in the
  saved teacher-off eval.
- Contact/reaction is credible under the current AABB proxy contract, but
  meaningful push distance needs a clearer next gate.

## Decision

Verdict:

`D292_TINY_PPO_ACTORFREEZE_TENSORBOARD_WARN_TEACHER_OFF_PASS_NO_LONG_PPO`

Next concrete order:

1. Do not run long PPO.
2. Do not claim learned-policy success or RoArm readiness.
3. Keep the D290 replay-batch actor path; do not switch back to D257 MLP teacher
   blending for this PPO gate.
4. Before another PPO runtime, decide the displacement/horizon contract:
   separate short-horizon collection warnings from 580-step teacher-off
   physical displacement.
5. If explicitly approved, run only a constrained short PPO gate with actor
   preservation, TensorBoard scalar gate, and saved-checkpoint teacher-off eval.

## Verification

- Tiny PPO smoke exit code: `0`.
- TensorBoard scalar gate exit code: `0`.
- Teacher-off frozen eval exit code: `0`.
- GPU after runtime: `NVIDIA GeForce RTX 4090 Laptop GPU`, memory used
  `1649MiB`, utilization `0%`.
- Active Isaac/PPO/torchrun/TensorBoard process filter returned no matches.
