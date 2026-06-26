# D259 Cube10cm Top-view Teacher-only Rollout / Feature-alignment Probe

Date: 2026-06-19 KST

Scope:

- Professor 10cm / 0.72kg cube top-view visual trajectory branch only.
- No long PPO.
- No teacher-off evaluation.
- No RoArm deployment or readiness claim.
- No RunPod/B200/SSH/pull/cleanup.

## Starting Point

D258 proved wiring only:

- D257 checkpoint loaded through `bc_teacher_checkpoint_path`.
- `cube_push_bc_teacher_blend_mean` and `cube_push_bc_teacher_imitation_mse`
  were logged.
- Behavior stayed unproven: near-zero displacement and zero success.

Therefore the next valid step was a policy-free teacher-only rollout and feature
alignment probe, not longer PPO.

## Code Changes

Added:

- `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`

Purpose:

- Load the D257 state-action teacher through the existing env sidecar.
- Step Isaac Lab envs using teacher actions only.
- Compare online env teacher features against D256 train-clean feature ranges.
- Record phase alpha timing, raw and clamped joint deltas, action cap rate,
  TCP-to-cube distance, contact threshold hits, and cube displacement.

Small env config fix:

- `RoArmCubeTap10cmEnvCfg` now defines `tap_overshoot_terminate: bool = False`.
- Reason: `_get_dones()` reads `self.cfg.tap_overshoot_terminate`; the first
  10cm probe attempt failed with `AttributeError` before this default was added.

## Commands

Static checks:

```bash
python3 -m py_compile sim_scripts/cube10cm_top_view_teacher_rollout_probe.py
python3 -m py_compile roarm_rl/roarm_cube_push_env.py
```

Teacher-only probes:

```bash
conda run -n isaaclab env PYTHONPATH=. python sim_scripts/cube10cm_top_view_teacher_rollout_probe.py --env_kind push3cm --num_envs 32 --steps 580 --sample_every 20
conda run -n isaaclab env PYTHONPATH=. python sim_scripts/cube10cm_top_view_teacher_rollout_probe.py --env_kind tap10cm --num_envs 32 --steps 580 --sample_every 20
conda run -n isaaclab env PYTHONPATH=. python sim_scripts/cube10cm_top_view_teacher_rollout_probe.py --env_kind tap10cm --out_dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d259_posx_ik --num_envs 32 --steps 580 --sample_every 20 --fixed_push_dir_x 1 --fixed_push_dir_y 0 --ik_endpoint_reset
```

## Outputs

Baseline D258 env reproduction:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d259/push3cm/teacher_rollout_probe_summary_d259.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d259/push3cm/teacher_rollout_probe_summary_d259.md`

Intended 10cm env:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d259/tap10cm/teacher_rollout_probe_summary_d259.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d259/tap10cm/teacher_rollout_probe_summary_d259.md`

10cm +x/IK alignment attempt:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d259_posx_ik/tap10cm/teacher_rollout_probe_summary_d259.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d259_posx_ik/tap10cm/teacher_rollout_probe_summary_d259.md`

Step-sample CSV files were generated but remain ignored by the existing `*.csv`
rule.

## Results

### D258 Env Reproduction: `push3cm`

- Env id: `RoArm-CubePush-Direct-v0`
- Cube size: `0.03m`
- Contact rate: `0.0`
- First contact step: `-1`
- First alpha > 0 step: `220`
- First alpha == 1 step: `309`
- Min TCP-cube distance mean/min/max:
  `0.21149027347564697` / `0.09386380761861801` / `0.3410947620868683`
- Max disp along mean/min/max:
  `0.0005538503755815327` / `-4.76837158203125e-06` /
  `0.01771417260169983`
- Raw delta clip exceed rate: `1.0`
- Action cap rate: `0.7770743534482759`
- Feature outside D256 train min/max rate: `0.5803001277139208`
- Feature outside D256 train p01/p99 rate: `0.6528715676883781`

This reproduces the D258 PPO smoke env kind and confirms it is the default 3cm
CubePush env, not the professor 10cm cube env.

### Intended 10cm Env: `tap10cm`

- Env id: `RoArm-CubeTap10cm-Direct-v0`
- Cube size: `0.1m`
- Contact rate: `0.0`
- First contact step: `-1`
- First alpha > 0 step: `220`
- First alpha == 1 step: `309`
- Min TCP-cube distance mean/min/max:
  `0.18824803829193115` / `0.07045303285121918` / `0.3121436536312103`
- Max disp along mean/min/max:
  `0.0033230045810341835` / `-2.384185791015625e-05` /
  `0.0329592227935791`
- Raw delta clip exceed rate: `1.0`
- Action cap rate: `0.7768139367816091`
- Feature outside D256 train min/max rate: `0.593532487228608`
- Feature outside D256 train p01/p99 rate: `0.6467712324393359`

The correct 10cm object size alone does not make the D257 teacher reach contact.

### 10cm +x/IK Alignment Attempt

- Env id: `RoArm-CubeTap10cm-Direct-v0`
- Cube size: `0.1m`
- `fixed_push_dir_x/y`: `1.0` / `0.0`
- `ik_endpoint_reset`: `True`
- Contact rate: `0.0`
- First contact step: `-1`
- First alpha > 0 step: `300`
- First alpha == 1 step: `519`
- Min TCP-cube distance mean/min/max:
  `0.08880475163459778` / `0.06847080588340759` / `0.13267822563648224`
- Max disp along mean/min/max:
  `1.254022479057312` / `-0.026723504066467285` /
  `11.039312362670898`
- Raw delta clip exceed rate: `0.9999676724137931`
- Action cap rate: `0.5438308189655172`
- Feature outside D256 train min/max rate: `0.60452187100894`
- Feature outside D256 train p01/p99 rate: `0.6507104086845467`

This fixed the `push_dx/push_dy` feature mismatch, but it still did not produce
valid teacher behavior. The cube displacement exploded in some envs, so this is
not a promotion path.

## Critical Findings

1. D258 PPO smoke used the wrong task geometry for the professor data.

`roarm_rl/train_cube_push_ppo.py` instantiates `RoArmCubePushEnvCfg()` and makes
`RoArm-CubePush-Direct-v0`. That env is the default 3cm cube path. The professor
dataset and D257 teacher are 10cm cube data.

2. Env push direction does not match D256 train-clean unless fixed.

D256 teacher-prior rows are +x-only:

- `push_dx=1.0`
- `push_dy=0.0`

The env randomizes over four directions unless `fixed_push_dir_x/y` is set.

3. The teacher feature contract is not aligned.

D256 feature construction uses `target_position_world_m` from the visual
trajectory logs. Env-side `_bc_teacher_feature_tensor()` uses
`_bc_teacher_tcp_target()` as `target_local_*`.

The clearest symptom:

- D256 `target_local_z_m` is fixed at `0.03788299858570099`.
- 10cm env teacher feature `target_local_z_m` is
  `0.0768829956650734..0.09088299423456192`.

This means the D257 teacher is being asked to infer actions from a different
feature meaning than it was trained on.

4. Reset/initial-joint distribution is not aligned.

Default env reset starts near `HOME_RAD` with jitter. D256 teacher-prior rows
start from visual trajectory poses. Turning on IK reset reduces part of the
initial distance but still leaves action saturation and unstable cube motion.

## Verdict

`D259_TEACHER_ROLLOUT_PROBE_CONTACT_FAIL_FEATURE_CONTRACT_MISMATCH_NO_LONG_PPO`

## Next Valid Work

Do not launch longer PPO.

Next work should be a feature-contract correction, then another teacher-only
probe:

1. Decide whether D257/D256 teacher features should use visual dataset target
   semantics or env-side TCP target semantics.
2. Update either the env feature tensor or rebuild/retrain the teacher so both
   sides use the same `target_*` meaning.
3. Make PPO/teacher runtime explicitly select the 10cm env, not the default 3cm
   CubePush env.
4. Fix push direction for this dataset (`fixed_push_dir_x=1`,
   `fixed_push_dir_y=0`) or retrain on randomized push directions.
5. Align reset/initial joint distribution to the visual trajectory before
   trusting teacher-on rollout.
6. Rerun teacher-only probe and require:
   - no extreme raw delta saturation;
   - no cube explosion;
   - TCP-to-cube distance reaches the contact threshold;
   - plausible contact/reaction before any PPO learning.

Only after that should a tiny 10cm PPO smoke be considered.
