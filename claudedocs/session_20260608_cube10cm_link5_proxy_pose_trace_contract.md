# 2026-06-08 cube10cm link5 proxy pose/trace code contract

## Scope

- Branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier branch.
- User request: explain and then proceed with local code preflight for `link5_collision:corner_011` proxy plus pose/trace support.
- Not run: GPU/IsaacLab runtime, dataset generation, PPO/RL, VLA, RoArm-M3-Pro control, Track A, B200/SSH/pull.
- Dirty/untracked state was preserved; no unrelated files were reverted.

## Starting State

- `START_HERE.md` said the current tool/contact-proxy preflight found `link5_collision:corner_011` as the stable physical proxy, but orientation/contact semantics were unvalidated and GPU was premature.
- D168 said not to treat `measured_contact_now` as mesh-contact proof because the probe derives it from cube displacement, and required local code preflight before any tiny runtime.
- Existing preflight summary line 9 said side-center retargeting via the stable link5 corner could reduce current link5 target error from `0.053769121m` to `0.040192105m` with ratio `0.748512386`, but line 10 said the current trace was position-only and orientation was not validated.

## What Changed

- Updated `sim_scripts/cube3cm_push_diffik_probe.py`:
  - Added `LINK5_COLLISION_CORNER_011_LOCAL_M`.
  - Added default-preserving CLI options:
    - `--diffik_command_type position|pose`, default `position`.
    - `--diffik_pose_quat_mode current_link5|initial_link5`, default `current_link5`.
    - `--tool_contact_proxy_mode hand_tcp|link5_collision_corner_011`, default `hand_tcp`.
  - Passed `args.diffik_command_type` into `DifferentialIKControllerCfg`.
  - Kept existing hand-TCP target behavior when defaults are used.
  - Added link5-corner proxy targeting by subtracting the selected tool-proxy offset from the contact target.
  - Added optional 7D pose command construction only when `--diffik_command_type pose` is explicitly selected.
  - Added trace fields for tool-contact target, tool proxy before/after position, proxy target error, proxy z error, and link5 before/target/after quaternions.
  - Added per-env and summary fields for tool proxy mode/local offset and min/final proxy target error.

## New Audit

- Added `sim_scripts/cube10cm_link5_proxy_pose_trace_contract_audit.py`.
- This audit is local-only: no IsaacLab runtime, no GPU, no dataset generation, no training, no robot control, no SSH.
- It statically checks the probe code and previous preflight summary to verify:
  - defaults preserve the old hand-TCP position path,
  - runtime mapping exists for the link5 proxy and pose command,
  - trace fields are present,
  - summary fields are present,
  - exactly one tiny runtime candidate can be considered after explicit approval.

## Audit Result

- Summary line 1: local-only, no GPU/runtime/data/training/robot/SSH.
- Summary line 3: defaults remain `command_type=position`, `proxy_mode=hand_tcp`, `pose_quat_mode=current_link5`, and `default_preserves_existing=True`.
- Summary line 4: runtime mapping is present: `diffik_cfg_uses_arg_command_type=True`, `link5_proxy_branch_present=True`, `pose_7d_command_present=True`, `proxy_offset_uses_target_quat=True`, `runtime_mapping_ok=True`.
- Summary line 5: 29 required trace fields are present.
- Summary line 6: 7 required summary keys are present.
- Summary line 7: pose support is available, but `pose_first_runtime_recommended=False`.
- Summary line 8: first tiny candidate is `seed962_yplus_pre020_link5corner_position_trace_only`.
- Summary line 9: code contract is ready for one tiny runtime consideration, but dataset/RL/RoArm remain blocked.

## Critical Decision

- Do not use pose in the first tiny runtime.
- Reason: pose command support is implemented, but first using pose would mix proxy retargeting with a 6D pose constraint on a 5-joint arm.
- The cleaner experiment is to isolate one behavioral change first: hand-TCP target proxy -> `link5_collision_corner_011`, while keeping `command_type=position`.
- Pose remains a later diagnostic if the position+proxy trace shows orientation drift or proxy semantics still fail.

## Tiny Runtime Candidate, Not Run

If explicitly approved later, the exact tiny candidate should be fixed seed962 y+ pre020 geometry with only the proxy mode changed:

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

Judge order for that future runtime:

1. reaction/contact/no-posewrite/no-overshoot,
2. tool proxy target error and link5 quaternion trace evidence,
3. quality tier,
4. final displacement only as secondary evidence.

## Verification

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py`: PASS.
- `python sim_scripts/cube10cm_next_research_step_audit.py`: PASS, but it still reflects older seed946 wording and is not the current D168/D169 next-step authority.
- `python sim_scripts/cube10cm_tool_contact_proxy_orientation_preflight.py`: PASS.
- `python sim_scripts/cube10cm_link5_proxy_pose_trace_contract_audit.py`: PASS.
- `python -m py_compile sim_scripts/cube10cm_tool_contact_proxy_orientation_preflight.py sim_scripts/cube10cm_link5_proxy_pose_trace_contract_audit.py sim_scripts/cube10cm_tap_objective_contract_audit.py sim_scripts/cube10cm_next_research_step_audit.py sim_scripts/cube10cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py sim_scripts/cube10cm_reaction_event_gate_audit.py sim_scripts/cube10cm_reaction_window_contract_audit.py`: PASS.
- `git diff --check`: PASS.

## Current State

- Code contract: ready for one tiny runtime consideration after explicit approval.
- Runtime: not run in this session.
- Dataset/RL/RoArm: still blocked.
- Next: either ask for/receive explicit approval for the one tiny local runtime above, or inspect the code contract further. Do not run pose first, do not mix lateral/height/actuator/DLS/cap/top-margin changes.
