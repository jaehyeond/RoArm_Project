# 2026-06-08 cube10cm link5-corner position runtime

## Scope

- Branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier branch.
- Executed exactly one local IsaacLab tiny runtime after guard checks and explicit user approval.
- Runtime change: `--tool_contact_proxy_mode link5_collision_corner_011` with `--diffik_command_type position`.
- Held fixed: seed962 y+ pre020 geometry, lateral, height, actuator, DLS, cap, top-margin, dataset/RL/RoArm.
- Not run: B200/SSH/pull, Track A, 1024/10240, dataset generation, PPO/RL, VLA, RoArm-M3-Pro control.

## Pre-Run Guards

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py`: PASS.
- `python sim_scripts/cube10cm_tool_contact_proxy_orientation_preflight.py`: PASS.
- `python sim_scripts/cube10cm_link5_proxy_pose_trace_contract_audit.py`: PASS.
- `python -m py_compile ...`: PASS.
- `git diff --check`: PASS.
- Output files did not exist before the run.
- GPU check before runtime: RTX 4090 Laptop, `13845 MiB` free, utilization `18%`.

## Runtime Command

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube10cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 962 \
  --fixed_cube_x_m 0.295 \
  --fixed_cube_y_m -0.044 \
  --fixed_push_dir 0 1 \
  --base_lateral_offset_m -0.020 \
  --xneg_tcp_center_height_offset_m 0.050 \
  --precontact_clearance_m 0.020 \
  --tool_contact_proxy_mode link5_collision_corner_011 \
  --diffik_command_type position \
  --diffik_pose_quat_mode current_link5 \
  --trace_diffik_diagnostics \
  --trace_all_envs \
  --trace_stride 4 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_trace.csv
```

Runtime stdout confirmed:

- `isaac_run=YES`, `num_envs=16`, `episodes=1`, 10cm cube, mass `0.720kg`.
- `tcp_height_mode=side_center`, `base_lateral_offset_m=-0.020000`, `precontact_clearance_m=0.020`.
- `command_type=position`, `tool_contact_proxy_mode=link5_collision_corner_011`.
- `training=NO`, `dataset_generation=NO`, `grasp=NO`, `rollout_object_posewrite=NO`.

## Post-Run Audits

1. Trace diagnostic:
   - `python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py ...`
   - Output: `diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_trace_diagnostic_summary.json`

2. Reaction gate:
   - `python sim_scripts/cube10cm_reaction_event_gate_audit.py ...`
   - Output: `diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_reaction_gate_audit.json`

3. Reaction-window contract:
   - `python sim_scripts/cube10cm_reaction_window_contract_audit.py ...`
   - Output: `cube10cm_reaction_window_link5corner_position_seed962_audit.json`

4. Comparison audit:
   - Added and ran `sim_scripts/cube10cm_link5corner_runtime_comparison_audit.py`.
   - Output: `cube10cm_link5corner_position_seed962_comparison_audit_summary.out`.

## Results

### Primary Gate

- Reaction/contact/no-posewrite/no-overshoot PASS.
- Reaction gate JSON:
  - `reaction_event_rate=1.0`
  - `contact_evidence_rate=1.0`
  - `overshoot_rate=0.0`
  - `no_posewrite=true`
  - `reaction_gate_pass=true`

### Tracking And Proxy

- Compared to seed962 pre020 baseline:
  - DiffIK clip improved `1.000000000 -> 0.515544884`.
  - Final TCP error improved `0.051811996m -> 0.038090036m`.
  - Final tool proxy target error mean was `0.002636088m`.
  - Min tool proxy target error mean was `0.000577164m`.
- Trace diagnostic still says clipping is dominant:
  - trace clip-any `0.516581633`
  - pre-stop clip-any `0.331521739`
  - post-stop clip-any `0.956896552`
  - likely mode `JOINT_STEP_CLIPPING_DOMINANT`

### Reaction Window And Quality

- Reaction-window audit accepted 16/16 windows.
- Quality tiers improved from baseline 2B+14C to new 16B.
- Follow p95/cap improved `1.160505840 -> 0.201606750`.
- Clean teacher remains false because window clip mean is still `0.666666667`.

### Tap Strength

- Tap became weaker than seed962 pre020 baseline:
  - max displacement `0.002923813m -> 0.001431603m`
  - max speed `0.127446551m/s -> 0.024424722m/s`
  - max displacement ratio `0.489635665`
  - speed ratio `0.191646785`
- Summary threshold rates:
  - 1mm: `1.0`
  - 5mm: `0.0`
  - 10mm: `0.0`
  - 20mm: `0.0`
  - 30mm: `0.0`
- Low-motion remains `1.0`.

## Verdict

- This runtime is a valid evidence point, not a data/RL unlock.
- It supports: link5-corner proxy retargeting improves tracking and turns all windows into Tier B.
- It also supports: retargeting weakens tap strength compared with seed962 pre020.
- Clean DiffIK teacher is still not ready.
- Dataset generation, IsaacLab RL, and RoArm-M3-Pro remain blocked.

## Next

- Do not repeat the same runtime.
- Do not jump to pose first.
- Do not scale dataset/RL/RoArm.
- Next evidence step should be visual proxy-contact inspection, or exactly one strength-preserving proxy variant only if the required tap is stronger than this verified weak 1mm event.
