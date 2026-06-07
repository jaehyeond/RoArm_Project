# 2026-06-08 cube10cm contact geometry mismatch

## Scope

- Active branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window +
  quality-tier branch.
- Not Track A. No B200/SSH/pull. No 1024/10240, dataset generation, training,
  PPO/RL, VLA, or RoArm-M3-Pro control.
- User request: after visual replay showed contact but not clean tap, isolate why
  seed962 contact frame has about 5cm target/TCP vertical mismatch.

## Boot / guard context

- Ran `git status --short --untracked-files=all --branch` first. Existing dirty
  state was preserved and not reverted.
- Re-read `CLAUDE.md`; lines 5-31 define the Current-State Protocol and require
  START_HERE/DECISIONS/LEDGER/session files plus local metric verification before
  claims.
- Re-checked the visual sanity blocker:
  - `cube10cm_visual_sim_sanity_audit_summary.out` line 5: env0 contact frame
    has `tcp_z=0.100452900`, `target_z=0.049999580`,
    `tcp_minus_target_z=0.050453320`, `tcp_target_err_before=0.050612349`,
    and `clip_any=1`.
  - line 6: `visual_contact_replay_pass=True`,
    `clean_tap_visual_verified=False`.
  - line 7: dataset/RL/RoArm remain unblocked `NO`.

## Code-path check

- `sim_scripts/cube3cm_push_diffik_probe.py` lines 519-542:
  `compute_tcp_targets()` uses `side_center` target z as `cube[:,2] +
  tcp_center_height_offset`.
- seed962 summary lines 134-135 record `tcp_center_height_offset_m=0.0` and
  `tcp_height_mode=side_center`; lines 51-56 show only `xneg` directional height
  offset was set, so it does not apply to fixed y+.
- `roarm_rl/roarm_stack_env.py` lines 86-87 define
  `TCP_LOCAL_OFFSET_M=(0,0,0.115428)`.
- `roarm_rl/roarm_stack_env.py` lines 1176-1179 compute TCP as
  `link5_pos + quat_rotate(link5_quat, _tcp_local)`.
- `sim_scripts/cube3cm_push_diffik_probe.py` lines 562-585 compensate this offset
  by computing `link5_target_w = tcp_target_w - tcp_offset_w`.

## Local audit

- Added `sim_scripts/cube10cm_contact_frame_geometry_mismatch_audit.py`.
- The script is local-only:
  - no IsaacLab runtime
  - no GPU
  - no dataset generation
  - no training
  - no robot control
  - no SSH
- It reads the existing seed962 16-env trace and selects the first
  `measured_contact_now=1` row per env.

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_contact_frame_geometry_mismatch_audit.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_contact_frame_geometry_mismatch_audit_summary.out`

Summary evidence:

- line 1: local audit only, no GPU/data/training/robot/SSH.
- line 2: source trace has `1568` rows, `16` envs, `16` first-contact envs,
  contact source lines `962-1028`.
- line 3: runtime contract is `tcp_height_mode=side_center`,
  `tcp_center_height_offset=0.0`, directional offsets xneg-only, applied offset
  mean `0.0`, DiffIK clip mean `1.0`.
- line 4: first-contact vertical mismatch is systemic:
  `tcp_minus_target_z_mean=0.052857013`,
  `link5_minus_target_z_mean=0.052857012`,
  `z_err_fraction_mean=0.983196354`,
  `tcp_target_xy_err_mean=0.009740644`.
- line 5: actual TCP is near live cube top, not side-center:
  `tcp_above_live_cube_center_z_mean=0.048793540`,
  `tcp_below_live_cube_top_z_mean=0.001206460`,
  `target_minus_live_cube_center_z_mean=-0.004063472`,
  `tcp_near_top_10mm_rate=1.0`,
  `tcp_near_center_10mm_rate=0.0`.
- line 6: TCP/link5 offset compensation is consistent:
  `offset_consistency_abs_max=0.000000007`.
- line 7: first-contact clipping is saturated:
  `clip_any_rate_at_first_contact=1.0`,
  `clip_mode=link1_to_link2`,
  `clip_mode_rate=1.0`.
- line 8: verdict
  `SIDE_CENTER_TARGET_NOT_TRACKED_TCP_CONTACTS_NEAR_TOP_UNDER_CLIPPING`.
- line 9: dataset/RL/RoArm remain blocked.

Raw trace cross-check:

- Full seed962 trace line 962 independently shows env0 contact frame:
  `frame=60`, `step=240`, `target_z=0.049999579787254333`,
  `tcp_z_before=0.10014015436172485`,
  `link5_target_z=0.15039949119091034`,
  `link5_z_before=0.20054006576538086`,
  `tcp_target_err_before=0.050612349063158035`, and `clip_any=1`.

## Negative control against simple target-height fix

- Prior seed944 y+ height050 summary lines 69-70: final TCP error improved to
  `0.022889409`, but first measured contact stayed `-1`.
- seed944 summary lines 99-103: max displacement was only
  `0.000058706849813461304m`, measured contact seen rate `0.0`.
- seed944 summary lines 126-127: this was the `tcp_center_height_offset_m=0.05`
  height-only trial.
- seed944 reaction gate lines 2-3 and 18-27: contact evidence `0.0`, reaction
  gate false, teacher quality false.

## Verdict

- The 5cm mismatch is not a missing TCP local-offset bug.
- The code asks for side-center target z, but at first contact the physical/visual
  TCP is near the cube top, and the target is not tracked under saturated clipping.
- This preserves visual contact evidence, but it does not verify a clean tap and
  does not unblock action-teacher dataset, IsaacLab RL, or RoArm-M3-Pro.
- Final 1cm/final retention is not the blocker here.

## Next

- Local-only next work: design the teacher contact frame before any GPU retest.
- The design question is now explicit: should the teacher target a true side-center
  TCP, an upper-edge/contact proxy, or a different tool/orientation path?
- Do not simply raise target z; seed944 shows height-only can reduce target error
  while killing contact.
