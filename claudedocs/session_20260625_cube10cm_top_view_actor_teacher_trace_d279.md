# 2026-06-25 - cube10cm top-view actor-vs-teacher diagnostic trace D279

## Scope

This session did not run PPO training, rendering, RunPod, B200, cleanup, or
RoArm control.

The goal was diagnostic only:

- keep the D278 teacher-off frozen eval contract fixed;
- load the D277 frozen actor checkpoint;
- load the D257 state-action teacher checkpoint only as a comparison sidecar;
- verify whether the actor learned the teacher action direction or mostly
  relied on `bc_teacher_blend=1.0` during D277.

## Added Script

- `sim_scripts/cube10cm_top_view_actor_teacher_trace.py`

The script records:

- actor action versus teacher sidecar action MSE/MAE/cosine;
- per-joint actor/teacher abs and signed means;
- raw actor clip exceed rate;
- phase alpha;
- AABB contact/useful/reaction/success;
- overshoot;
- vertical offset;
- displacement;
- joint delta cap rate;
- env-level first-event steps.

The teacher checkpoint is not allowed to blend actions:

- `bc_teacher_blend=0.0`;
- `bc_teacher_imitation_reward_scale=0.0`.

## Command

```bash
env PYTHONPATH=. conda run -n isaaclab python sim_scripts/cube10cm_top_view_actor_teacher_trace.py
```

## Runtime Contract

- env id: `RoArm-CubeTap10cm-Direct-v0`
- actor checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/model_0.pt`
- teacher sidecar checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- D256 reset CSV:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256/ppo_actor_prior_teacher_rows_d256.csv`
- `tap_contact_proxy_mode=link5_collision_aabb`
- `d256_reset_frame_index=0`
- `d256_reset_sample_mode=linspace`
- fixed +x push direction
- `episode_length_s=6.0`
- `eval_steps=580`
- `num_envs=32`
- `bc_teacher_feature_target_mode=env_target`
- `bc_teacher_phase_timing=direct_steps`

## Output

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_teacher_trace_d279/tap10cm/actor_teacher_trace_summary_d279.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_teacher_trace_d279/tap10cm/actor_teacher_trace_summary_d279.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_teacher_trace_d279/tap10cm/actor_teacher_trace_steps_d279.csv`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_teacher_trace_d279/tap10cm/actor_teacher_trace_envs_d279.csv`

Step CSV rows:

- `581` lines = header + `580` steps.

Env CSV rows:

- `33` lines = header + `32` envs.

## Verdict

`D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`

Diagnostic class:

`actor_teacher_mismatch_plus_unsafe_physics`

## Key Metrics

- D256 reset active rate: `1.0`
- BC teacher blend last: `0.0`
- actor-teacher MSE: `0.46601414680480957`
- actor-teacher MAE: `0.5703011751174927`
- actor-teacher cosine: `0.07783761620521545`
- actor clipped abs mean trace: `0.11846771989146183`
- teacher abs mean trace: `0.5554168882908236`
- actor raw clip exceed mean: `0.00006285919727564886`
- contact/reaction/useful: `0.875/0.875/0.5625`
- success flag: `0.875`
- overshoot: `0.3125`
- max displacement along mean/max:
  `0.0024283849634230137/0.018782615661621094`
- max displacement xy mean/max:
  `0.020250540226697922/0.10077980160713196`
- vertical max: `0.24940747022628784m`
- joint delta cap max trace: `0.15625`

Per-joint actor-vs-teacher abs gap:

- base: `0.7280789017677307`
- shoulder: `0.6867650151252747`
- elbow: `0.7172043919563293`
- wrist_pitch: `0.7668752074241638`
- wrist_roll: `0.4209713637828827`
- gripper: `0.1019124984741211`

## Failure-Mode Split

Overshoot group:

- count: `10`
- actor-teacher MSE: `0.3936862051486969`
- max disp xy mean: `0.059471823275089264`
- max vertical: `0.0`

Vertical-over-threshold group:

- count: `5`
- actor-teacher MSE: `0.4085454046726227`
- max disp xy mean: `0.0005952191422693431`
- max vertical: `0.14238949120044708`

This means the failure is not a single scalar problem. Overshoot and vertical
outlier behavior are partly separate, and both happen while the actor action is
not aligned with the teacher sidecar.

## Interpretation

D279 confirms D278 was not merely too strict.

The D277 actor is not following the D257 teacher action direction:

- action cosine is near zero;
- actor action magnitude is about one-fifth of teacher sidecar magnitude;
- per-joint action gaps are large on the arm joints that actually matter for
  contact.

The physics failure is also real:

- overshoot remains far above the `0.05` gate;
- vertical outliers reach `0.24940747022628784m`;
- success flag alone is not a valid promotion metric because overshoot can
  happen after or around the useful event.

Therefore:

- do not run long PPO;
- do not run a short PPO ladder from D279;
- do not claim learned policy, teacher-off success, or RoArm readiness.

## Next Work

The next work should improve the actor/teacher learning bridge before PPO
scale-up.

Recommended order:

1. Build a supervised actor warm-start/distillation path using the same D256
   reset/AABB observation contract.
2. Check actor-vs-teacher action MSE/cosine before running PPO.
3. Run teacher-off frozen eval only after the action match improves.
4. Repeat D279 actor-vs-teacher trace.
5. Only if teacher-off and D279-style gates improve, consider a tiny PPO smoke
   plus TensorBoard gate.

## Verification

- D279 command exited with code `0`.
- `python -m py_compile sim_scripts/cube10cm_top_view_actor_teacher_trace.py`
  passed.
- `git diff --check` passed.
- Narrowed D279/PPO/torchrun/rl_games process check found no active process.
- `nvidia-smi` returned to baseline class state in the 2026-06-25 local
  session: `833MiB` used, `0%` GPU util.
