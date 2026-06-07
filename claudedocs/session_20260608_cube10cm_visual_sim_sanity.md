# 2026-06-08 Cube10cm Visual/Sim Sanity

## Scope

- Active branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window +
  quality-tier branch.
- Not Track A. No B200/SSH/pull. No 1024/10240, dataset generation, training,
  PPO/RL, VLA, or RoArm-M3-Pro control.
- User concern: numeric/code audits are not sufficient; visually verify whether
  seed962 actually looks like the intended 10cm target push/tap.

## Pre-Checks

- `git status --short --untracked-files=all --branch` initially showed
  `## master...origin/master`.
- Re-read `CLAUDE.md`, `START_HERE.md`, `DECISIONS.md`, local runtime summaries,
  and renderer code before acting.
- Guard scripts run before GPU/render work:
  - `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS.
  - `python sim_scripts/cube10cm_next_research_step_audit.py` default output was
    seed946-based, so it was not used as seed962 evidence.
  - Re-ran `cube10cm_next_research_step_audit.py` with explicit seed962 reaction
    and trace-diagnostic JSON; it kept next direction at
    `NARROW_ACTUATOR_IK_TRACKING_CLEANUP_INSIDE_WORKING_TAP_GEOMETRY`.
  - `python sim_scripts/cube10cm_yplus_precontact_candidate_audit.py` PASS.
  - `python -m py_compile ...` PASS.
  - `git diff --check` PASS.

## Local Trace Storyboard

- Added `sim_scripts/cube10cm_visual_sanity_trace_storyboard.py`.
- Output:
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_visual_sanity_trace_storyboard.html`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_visual_sanity_trace_storyboard.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_visual_sanity_trace_storyboard_summary.out`
- Summary lines:
  - line 1: local trace visual only; no GPU runtime, no dataset generation, no
    training, no robot control, no SSH.
  - line 2: 1568 trace rows, 16 envs, seed 962, fixed cube `(0.295,-0.044)`,
    y+ push, `precontact=0.02`.
  - line 3: reaction gate PASS, reaction/contact `1.0/1.0`, no posewrite true,
    overshoot `0.0`, teacher quality false.
  - line 4: `contact_to_p16` policy gives 16 Tier B rows but clip mean `1.0`.
  - line 5: env0 has first any reaction step 56, first y+ along 1mm step 240,
    speed step 56, z-delta step 64, contact/stop step 240.
  - line 6: all y+, contact/reaction/all max1mm true, clip saturated, actual
    rendered video not run.

## Live Record-Video Attempt

- With explicit approval, ran one local IsaacLab 1-env seed962 y+ visual attempt
  using `--record_video`.
- It booted IsaacLab headless rendering, printed the correct 10cm/0.72kg/y+
  runtime settings, and stated `training=NO dataset_generation=NO grasp=NO`.
- It produced no output files after several minutes:
  - video frame file count: `0`
  - no summary JSON
  - no rollout CSV
  - no trace CSV
- The process was stopped by a narrow `pkill -f` pattern for the visual output
  stem.
- Verdict: failed record-video path. Do not use it as evidence.

## Replay Renderer Repair

- Existing `sim_scripts/cube3cm_push_diffik_render_trace.py` was the correct
  prior visualization path, because it replays trace CSV with local RoArm STL
  mesh and records `physics_recomputed=false`.
- First replay attempt failed before SimulationApp because the seed962 trace
  contains string columns such as `trajectory_variant=v1`.
- Patched `load_trace()` to tolerate non-float columns by assigning `0.0` to
  unused string fields.
- First successful replay then revealed a serious visual bug: the renderer used
  hardcoded `0.015` cube scale, i.e. 3cm visual cube.
- Patched the renderer to set cube scale from trace `cube_size_x/y/z_m`, and to
  include `cube_size_m` in the render summary.

## Final Replay Evidence

- Created env0-only trace:
  - `diffik_probe_cube10cm_m072_fixed_yplus_visual_env0_seed962_trace.csv`
  - 99 lines total: header + 98 env0 rows.
- Generated final replay:
  - MP4: `diffik_probe_cube10cm_m072_render_replay_env0_seed962/diffik_probe_cube10cm_m072_render_replay_env0_seed962.mp4`
  - Frames: `diffik_probe_cube10cm_m072_render_replay_env0_seed962/frames/frame_0000.png` through `frame_0097.png`
  - Summary: `diffik_probe_cube10cm_m072_render_replay_env0_seed962_summary.json`
  - MP4 probe: `diffik_probe_cube10cm_m072_render_replay_env0_seed962_mp4_probe.out`
- Render summary lines 20-26: cube size env0 is `[0.1, 0.1, 0.1]`.
- Render summary lines 53-60: `fps=30`, `frames_written=98`, `height=720`,
  output path, and `physics_recomputed=false`.
- Render summary lines 61-74: local RoArm STL mesh visual mode,
  `training=false`, `dataset_generation=false`, `width=1280`.
- MP4 probe lines 1-8: `opened=True`, `frame_count=98`, `width=1280`,
  `height=720`, `fps=30.0`, `first_frame_ok=True`, nonzero size.
- Opened frames locally:
  - frame 0000: 10cm pink cube, black RoArm, target/TCP markers visible.
  - frame 0058/0060/0062: robot end effector reaches the 10cm cube side/top
    region; contact is visible, but it does not look like a clean side-center tap.
  - frame 0097: end effector remains close/in contact; no evidence that this
    should be treated as clean action teacher.

## Visual Sanity Audit

- Added and ran `sim_scripts/cube10cm_visual_sim_sanity_audit.py`.
- Summary lines:
  - line 1: local audit only, no GPU runtime, no dataset/training/robot/SSH.
  - line 2: trace storyboard ready; all y+, contact/reaction true, actual live
    render video false.
  - line 3: record-video attempt failed with no frames/summary/CSV/trace.
  - line 4: replay render PASS, 98 frames, MP4 opened, 10cm cube,
    `physics_recomputed=False`, `training=False`, `dataset_generation=False`.
  - line 5: contact frame env0 frame 60 / step 240 has
    `tcp_z=0.100452900`, `target_z=0.049999580`, vertical delta
    `0.050453320m`, `tcp_target_err_before=0.050612349`,
    `tcp_target_err_after=0.050918311`, `disp_along_push=0.001234770`,
    `cube_speed=0.016902385`, `tip_deg=2.371614218`, `clip_any=1`,
    `clip_joint=link1_to_link2`.
  - line 6: `visual_contact_replay_pass=True`, but
    `clean_tap_visual_verified=False`.
  - line 7: dataset/RL/RoArm/action-teacher remain blocked.

## Verdict

- Visual contact replay PASS: the seed962 env0 trace does show the robot reaching
  and contacting/reacting with the 10cm cube, and the generated MP4 is valid.
- Clean tap NOT verified: at the contact frame, the TCP/target vertical mismatch
  is about 5cm and clipping is active. This matches the existing teacher-quality
  blocker rather than resolving it.
- Do not build action-teacher dataset, do not run IsaacLab RL, and do not deploy
  to RoArm-M3-Pro from this state.

## Next

- The next research step is not more final displacement checking and not broad
  data/RL.
- Fix or retest teacher contact geometry/control tracking inside the existing
  10cm tap objective. The likely issue to isolate is why target side-center z and
  actual TCP/contact z differ by about 5cm at contact while clipping remains
  saturated.
