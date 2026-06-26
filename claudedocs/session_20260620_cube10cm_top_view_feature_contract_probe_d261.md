# D261 Cube10cm Top-view Feature-contract Teacher Probe

Date: 2026-06-20 KST

Scope:

- Professor 10cm / 0.72kg cube top-view visual trajectory branch only.
- No PPO learning.
- No long PPO.
- No teacher-off evaluation.
- No learned-policy, RoArm readiness, RunPod, B200, SSH, pull, or cleanup claim.

## Starting Point

D258 proved that the D257 checkpoint loads through the PPO `bc_teacher_*` hooks,
but behavior was near static. D259 then showed teacher-only contact failure and
identified feature-contract blockers:

- D258 used `RoArm-CubePush-Direct-v0`, which is the 3cm cube env.
- D256 train-clean teacher rows are +x only (`push_dx=1`, `push_dy=0`).
- D256 `target_position_world_m` is not the same semantic target as the env's
  online `_bc_teacher_tcp_target()`.
- Default reset/joint state is outside the visual trajectory distribution.

D260 made TensorBoard inspection mandatory for later PPO, but did not unblock
PPO. The next valid step was a teacher-only feature-contract probe.

## Code Changes

Updated `roarm_rl/roarm_cube_push_env.py`:

- Added `bc_teacher_feature_target_mode`.
- Default remains `tcp_target`.
- New `env_target` mode uses `self._target_world` for teacher features, matching
  the D256 visual-log `target_position_world_m` contract.

Updated `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`:

- Added `--bc_teacher_feature_target_mode`, default `env_target`.
- Added `--artifact_tag`.
- Summary JSON/markdown now record the target feature mode.

Updated `roarm_rl/train_cube_push_ppo.py`:

- Added `--env_kind {push3cm,tap10cm}`.
- Added dynamic env id selection and dynamic `gym.make(env_id, cfg=env_cfg)`.
- Added `--fixed_push_dir_x/--fixed_push_dir_y`.
- Added `--bc_teacher_feature_target_mode {tcp_target,env_target}`.

This PPO entrypoint change is a guardrail for a future tiny PPO smoke. It was
not used to run PPO in D261.

## Static Verification

Command:

```bash
python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_push_ppo.py sim_scripts/cube10cm_top_view_teacher_rollout_probe.py
```

Result: pass.

## Teacher-only Probe A: 10cm +x env_target, no IK reset

Command:

```bash
conda run -n isaaclab env PYTHONPATH=. python sim_scripts/cube10cm_top_view_teacher_rollout_probe.py --env_kind tap10cm --out_dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx --num_envs 32 --steps 580 --sample_every 20 --artifact_tag d261_envtarget_posx --fixed_push_dir_x 1 --fixed_push_dir_y 0 --bc_teacher_feature_target_mode env_target
```

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx/tap10cm/teacher_rollout_probe_summary_d261_envtarget_posx.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx/tap10cm/teacher_rollout_probe_summary_d261_envtarget_posx.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx/tap10cm/teacher_rollout_step_samples_d261_envtarget_posx.csv`

Key results:

- `env_id`: `RoArm-CubeTap10cm-Direct-v0`
- cube size x/z: `0.1` / `0.1`
- fixed push dir x/y: `1.0` / `0.0`
- `bc_teacher_feature_target_mode`: `env_target`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max:
  `0.2137620449066162` / `0.144382044672966` /
  `0.29925453662872314`
- max disp along mean/min/max:
  `1.163780689239502e-05` / `9.268522262573242e-06` /
  `2.765655517578125e-05`
- raw delta clip exceed rate: `0.7170689655172414`
- action cap rate: `0.37896012931034484`
- feature outside train min/max rate: `0.4267700351213282`
- feature outside train p01/p99 rate: `0.47317608556832697`

Contract improvements:

- `push_dx` now matches train-clean: `1.0`.
- `push_dy` now matches train-clean: `0.0`.
- `target_local_x_m` is inside train min/max.
- `target_local_z_m` exactly matches train-clean:
  `0.03788299858570099`.

Remaining blockers:

- Arm joint ranges are still outside D256 train-clean ranges.
- `tcp_local_z_m` is still too high:
  train `0.03739422559738159..0.0959378182888031`,
  env `0.10544683039188385..0.3680003881454468`.
- `target_to_tcp_x/y/z` remain outside train-clean range.
- No contact occurs, so this is not a PPO promotion result.

## Teacher-only Probe B: 10cm +x env_target, IK reset

Command:

```bash
conda run -n isaaclab env PYTHONPATH=. python sim_scripts/cube10cm_top_view_teacher_rollout_probe.py --env_kind tap10cm --out_dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx_ik --num_envs 32 --steps 580 --sample_every 20 --artifact_tag d261_envtarget_posx_ik --fixed_push_dir_x 1 --fixed_push_dir_y 0 --bc_teacher_feature_target_mode env_target --ik_endpoint_reset
```

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx_ik/tap10cm/teacher_rollout_probe_summary_d261_envtarget_posx_ik.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx_ik/tap10cm/teacher_rollout_probe_summary_d261_envtarget_posx_ik.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx_ik/tap10cm/teacher_rollout_step_samples_d261_envtarget_posx_ik.csv`

Key results:

- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max:
  `0.0902239978313446` / `0.07528560608625412` /
  `0.1327148675918579`
- max disp along mean/min/max:
  `1.2454720735549927` / `-0.027089953422546387` /
  `10.891912460327148`
- max disp xy mean/max:
  `3.124444007873535` / `11.32904052734375`
- raw delta clip exceed rate: `0.6805603448275862`
- raw delta abs mean/max:
  `0.9575997591018677` / `264.475830078125`
- action cap rate: `0.3602280890804598`
- feature outside train min/max rate: `0.43064734993614306`

Interpretation:

- IK reset reduced TCP-cube distance compared with Probe A.
- It did not produce contact.
- It reintroduced an unstable rollout/explosion mode, with displacement above
  `10m` and raw teacher deltas far beyond the `+-0.040rad` runtime cap.
- IK reset is therefore not a valid promotion fix.

## Post-run Checks

- No matching Isaac/PPO/teacher-probe/torchrun/rl_games process remained after
  the probes.
- GPU returned to the observed baseline: about `2509MiB` used /
  `13436MiB` free, with pre-existing non-Isaac contexts still present.

## Verdict

`D261_FEATURE_TARGET_CONTRACT_PARTIAL_FIX_TEACHER_ONLY_CONTACT_FAIL_NO_PPO_PROMOTION`

D261 fixed part of the feature contract but did not make the D257 teacher a
valid behavior prior. Do not run long PPO. Do not run even tiny PPO unless
explicitly overriding the failed teacher-only gate.

## Next Work

Next work is not TensorBoard and not PPO. TensorBoard remains mandatory for any
future PPO smoke, but there is no PPO candidate to dashboard yet.

The next concrete research task is one of:

- align env reset/action rollout distribution to D256 train-clean visual
  trajectory states, especially arm joint range, TCP height, and target-to-TCP
  geometry;
- or retrain a teacher on features produced by the actual env-side reset and
  rollout contract;
- then rerun teacher-only contact before any PPO.

Only after teacher-only reaches plausible contact without saturation/explosion
should the corrected tiny PPO smoke be considered, followed by TensorBoard
scalar gate and teacher-off frozen eval.
