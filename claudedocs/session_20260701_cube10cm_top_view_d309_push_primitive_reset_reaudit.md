# D309 Cube10cm Push Primitive Reset Re-audit

Date: 2026-07-01 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch after D307/D308. This session did not run PPO, tiny PPO trace gates, TensorBoard training, torchrun, learned-policy updates, RoArm deployment, Track A, VLA/SmolVLA fine-tuning, B200/SSH, pull, or `.ssh` copy.

## Starting Point

D307 showed that an action governor partially helped overshoot-heavy cases, but did not fix tiny-displacement/contact-geometry failures. D308 moved the default-off env governor into the runtime contract and broadened diagnostics, but its fresh32 random runs had overshoot `6/32` and several step-0 or near-step-0 large displacements.

The working question for D309 was whether this was genuinely hard actor/control behavior, or whether the diagnostic reset/control contract itself was producing misleading failures.

## Code Changes

- `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
  - Default `--env_hook_force_second_reset` changed from true to false.
  - Added `--exec_source actor|zero|tap_push_primitive`.
  - Added primitive CLI controls:
    - `--primitive_goal_disp_m`
    - `--primitive_push_steps`
    - `--primitive_speed_stop_mps`
    - `--primitive_diffik_step_clip_rad`
    - `--primitive_target_path_mode near_face_goal|legacy_far_face_through`
    - `--primitive_cube_reference_mode start_pose|current_pose`
    - `--primitive_target_base_mode actual_joint_pos|previous_joint_target`
  - Added reset/current contact diagnostics to CSV/JSON outputs.
  - Changed primitive stop behavior: on the first stop step, save current joint positions and hold those positions. This terminates the primitive instead of continuing to hold the previous deep push target.

## Reset Artifact Audit

Forced-second-reset, zero-action, 5 steps:

- contact/useful/overshoot: `18/13/6` of `32`
- max XY mean/max: `4.548/26.208mm`
- `>=20mm`: `6/32`
- step-0 large displacements:
  - env3 ep35: `21.152mm`
  - env4 ep473: `26.027mm`
  - env9 ep771: `19.614mm`
  - env24 ep753: `19.516mm`
  - env27 ep514: `20.927mm`
  - env31 ep993: `20.268mm`

Corrected no-force-second-reset, zero-action, 5 steps:

- contact/useful/overshoot: `17/17/0` of `32`
- max XY mean/max: `0.414/11.983mm`
- `>=20mm`: `0/32`
- step-0 max: `5.620mm`

Decision: D308's first-step overshoot cluster is not reliable actor-action evidence under the old forced-second-reset protocol. It must be reproduced under corrected reset before being used as a policy/control blocker.

## Primitive Comparison Under Corrected Reset

Existing actor, seed `30701`:

- contact/reaction/useful/overshoot: `32/32/32/0`
- final current proxy: `11/32`
- XY `>=1mm`: `14/32`
- max XY mean/max: `2.155/16.449mm`

`near_face_goal + actual_joint_pos` primitive, seed `30701`:

- final current proxy: `31/32`
- overshoot: `0/32`
- XY `>=1mm`: `8/32`
- max XY mean/max: `0.976/8.341mm`

`legacy_far_face_through + actual_joint_pos` primitive, seed `30701`:

- final current proxy: `32/32`
- overshoot: `0/32`
- XY `>=1mm`: `10/32`
- max XY mean/max: `1.202/8.341mm`

`legacy_far_face_through + previous_joint_target` before stop termination:

- seed `30701`: XY `>=1mm` `32/32`, overshoot `0/32`, max XY `13.049mm`
- seed `30702`: XY `>=1mm` `32/32`, overshoot `3/32`, max XY `25.022mm`

The failure was stop semantics: the primitive latched stop, but kept applying the previous deep push target.

## Final Stop-Termination Result

Best primitive:

- `exec_source=tap_push_primitive`
- `primitive_target_path_mode=legacy_far_face_through`
- `primitive_target_base_mode=previous_joint_target`
- `primitive_cube_reference_mode=start_pose`
- `primitive_goal_disp_m=0.003`
- `primitive_push_steps=220`
- `primitive_speed_stop_mps=0.200`
- `primitive_diffik_step_clip_rad=0.010`
- corrected reset: `--no-env_hook_force_second_reset`

Seed `30701`:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max: `3.802/10.025mm`
- XY `>=1mm`: `32/32`
- XY `>=3mm`: `31/32`
- XY `>=7mm`: `1/32`
- XY `>=20mm`: `0/32`

Seed `30702`:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max: `3.756/5.438mm`
- XY `>=1mm`: `32/32`
- XY `>=3mm`: `32/32`
- XY `>=7mm`: `0/32`
- XY `>=20mm`: `0/32`

Combined `64` envs:

- contact/reaction/useful/final proxy: `64/64/64/64`
- overshoot: `0/64`
- cap: `0.0`
- max XY mean/max: `3.779/10.025mm`
- XY `>=1mm`: `64/64`
- XY `>=3mm`: `63/64`
- XY `>=7mm`: `1/64`
- XY `>=20mm`: `0/64`

## Interpretation

The user's skepticism was justified. The task is not trivial, but D308 made it look harder than it was because:

1. The old env-hook forced-second-reset path could create zero-action step-0/early displacement artifacts.
2. The first primitive implementation used stop as a latch while continuing to hold the previous deep push target.
3. Headline `useful` is not enough; the gate must include current proxy, overshoot, cap, and `>=1mm` displacement.

The successful direction is explicit non-PPO tool/object primitive control: corrected reset, object-frame push-through target, accumulated joint target, and stop termination by holding current joint state.

## Verdict

`D309_PUSH_PRIMITIVE_STOPTERM_CORRECTED_RESET_PASS_NO_PPO_PROMOTION`

D309 is a non-PPO deployable-control proof point. It is not a learned policy, not PPO promotion, and not RoArm readiness. Next work is env/runtime primitive integration or broader non-PPO primitive diagnostics before any PPO gate.

## Primary Artifacts

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action/`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action_no_force_second_reset/`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30701/`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30702/`
