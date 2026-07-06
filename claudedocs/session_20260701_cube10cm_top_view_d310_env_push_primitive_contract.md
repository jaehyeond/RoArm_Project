# D310 Cube10cm Env Push Primitive Contract

Date: 2026-07-01 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch after D309. This session did not run PPO, tiny PPO trace gates, TensorBoard training, torchrun, learned-policy updates, RoArm deployment, Track A, VLA/SmolVLA fine-tuning, B200/SSH, pull, or `.ssh` copy.

## Starting Point

D309 showed that the working direction was not more PPO. The corrected non-PPO tool/object push primitive solved the reset-artifact and stop-latch failures, but it was still executed from the D290 probe through `_external_joint_targets_override`.

D310's question was narrower: does the same primitive still work after moving it into the env runtime contract?

## Code Changes

- `roarm_rl/roarm_cube_push_env.py`
  - Added default-off `rl_action_mode="tap_push_primitive"` support.
  - Added primitive stop config:
    - `tap_push_primitive_stop_disp_m`
    - `tap_push_primitive_speed_stop_mps`
    - `tap_push_primitive_stop_on_overshoot`
  - Added primitive buffers and log scalars:
    - stop-latched state
    - stop step
    - hold targets
    - primitive target delta mean/max
  - Added `_tap_push_primitive_joint_target()`.
  - In `_pre_physics_step()`, `tap_push_primitive` ignores the actor for target generation and writes the env-computed bounded DiffIK joint target.

- `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
  - Added `--exec_source env_tap_push_primitive`.
  - In this mode, the probe sends zero actions to `env.step()`.
  - The env computes the push primitive internally through `rl_action_mode="tap_push_primitive"`.
  - Summary now records `env_rl_action_mode`.

## Runtime Contract

The primitive uses the D309 best settings:

- target path: `legacy_far_face_through`
- target base: `previous_joint_target`
- cube reference: `start_pose`
- goal displacement: `0.003m`
- push steps: `220`
- speed stop: `0.200m/s`
- DiffIK step clip: `0.010rad`
- reset: corrected `--no-env_hook_force_second_reset`

Stop is a primitive termination condition, not a latch on the old push target. Once the stop condition is seen, the env saves the current joint positions and holds them.

## Fresh Multi-Episode Validation

Runs:

- Seed `30701`:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d310/tap10cm/fresh32_random_env_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30701/`
- Seed `30702`:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d310/tap10cm/fresh32_random_env_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30702/`

Seed `30701`:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max: `3.802/10.025mm`
- XY `>=1mm`: `32/32`
- XY `>=3mm`: `31/32`
- XY `>=7mm`: `1/32`
- XY `>=20mm`: `0/32`
- primitive stop-latched: `31/32`

Seed `30702`:

- contact/reaction/useful/final proxy: `32/32/32/32`
- overshoot: `0/32`
- cap mean/max: `0.0/0.0`
- max XY mean/max: `3.756/5.438mm`
- XY `>=1mm`: `32/32`
- XY `>=3mm`: `32/32`
- XY `>=7mm`: `0/32`
- XY `>=20mm`: `0/32`
- primitive stop-latched: `32/32`

Combined `64` envs from per-env CSV:

- contact/reaction/useful/final proxy: `64/64/64/64`
- overshoot: `0/64`
- primitive stop-latched: `63/64`
- cap: `0.0`
- max XY mean/max: `3.779/10.025mm`
- XY `>=1mm`: `64/64`
- XY `>=3mm`: `63/64`
- XY `>=7mm`: `1/64`
- XY `>=20mm`: `0/64`

The single non-latched env in seed `30701` still had contact/reaction/useful, no overshoot, final proxy true, and max XY `2.523mm`, so it was below the `3mm` primitive stop threshold rather than a failed contact case.

## Interpretation

D310 confirms that D309's result was not dependent on the probe override path. The primitive now exists as an env/runtime action contract.

This does not promote PPO and does not prove a learned policy. It proves a deployable non-PPO control primitive for the current corrected reset distribution. The remaining research work is to broaden diagnostics and harden the contact contract, not to start PPO immediately.

## Verification

- `python -m py_compile sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py roarm_rl/roarm_cube_push_env.py`: pass
- `git diff --check`: pass
- Fresh non-PPO Isaac diagnostics: pass for seeds `30701` and `30702`
- Residual Isaac/PPO/TensorBoard/torchrun process check: none found
- `nvidia-smi --query-compute-apps`: no compute apps
- Final `nvidia-smi --query-gpu=...utilization.gpu`: `0%`

## Verdict

`D310_ENV_PUSH_PRIMITIVE_CONTRACT_REPRODUCES_D309_NO_PPO_PROMOTION`

Next work: broader non-PPO primitive diagnostics over harder reset coverage and contact-proxy/approach-face edge cases before any tiny PPO trace gate is reconsidered.
