# Session 2026-05-27 - Professor Cube3cm DiffIK v3 Visualization Replay

## Scope

Generate professor-facing visual material for the already-completed v3 scripted
IsaacLab Differential IK cube-push result without calling it learning, dataset
generation, Track A grasp success, or new physics evidence.

## Protocol Checks

- `CLAUDE.md:5-31` defines the Current-State Protocol and requires
  `START_HERE.md`, `DECISIONS.md`, `EXPERIMENT_LEDGER.md`, `git status`, and
  log verification before claims.
- `START_HERE.md:9` says not to use `HANDOFF.md` or `TASKS.md`.
- `START_HERE.md:13-15` says B200 access is expired/disconnected and future work
  must not depend on B200 SSH or B200-only paths.
- GPU/IsaacLab commands in Codex require escalated execution because the default
  sandbox hides `/dev/nvidia*`; this session used local IsaacLab only.

## What Changed

- Updated `sim_scripts/cube3cm_push_diffik_probe.py` to support trace capture:
  `--trace_env_id`, `--trace_env_ids`, `--trace_stride`, `--trace_csv`, and
  explicit env-origin fields.
- Added `sim_scripts/cube3cm_push_diffik_render_trace.py`, a replay-only renderer
  that reads a trace CSV, places black actual RoArm URDF STL mesh links, pink
  cubes, blue TCP markers, and red target markers in Isaac Sim, writes PNG
  frames, and composes an MP4.
- No Track A files were modified, and no dataset/training pipeline was started.

## Selected Video Case

The representative video case is `env_id=3` from the v3 reach16 seed779 run:

- Source row: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_reach16_seed779.csv:5`
- Direction: `(1,0)`, meaning positive X in the local table/world frame.
- Cube start: `x=0.353590250m`, `y=-0.073313951m`.
- Outcome: controlled `1`, impact `0`, low-motion `0`, success marker `1`.
- Measured displacement along the push direction: `0.036002159m`.
- Final TCP target error for this row: `0.041068129m`.

This case is intentionally from the weak `(1,0)` direction, but it is a clean
successful sample rather than a failure sample.

## Four-Direction Parallel Replay

For professor visualization, the stronger render is the four-env 2x2 replay:

- MP4:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_fourdir_realroarm_env0_3_4_7_seed779/diffik_probe_v3_fourdir_realroarm_env0_3_4_7_seed779.mp4`
- Source trace:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_trace_fourdir_env0_3_4_7_seed779.csv`
- `env_id` means the IsaacLab parallel environment slot. The four selected slots
  are not four separate learned policies; they are four simultaneous scripted
  DiffIK trials from one local 16-env IsaacLab run.
- Directions:
  env0 `(0,-1)`, env3 `(1,0)`, env4 `(0,1)`, env7 `(-1,0)`.
- The trace has 580 rows: 4 envs times 145 frames.
- Render stdout line 447 confirms `frames=145`, `env_count=4`, env IDs
  `[0, 3, 4, 7]`, `training=NO`, `dataset_generation=NO`, and
  `physics_recomputed=NO`.
- Render summary lines 12-19 confirm color intent: white background, black robot,
  gray table, pink cube, blue TCP marker, red target marker.
- Render summary lines 91-112 confirm `30fps`, 145 frames, `1280x720`, 2x2
  layout, `robot_visual_mode=black_roarm_urdf_stl_mesh_from_trace_joints`,
  `physics_recomputed=false`, and `training=false`.
- MP4 probe lines 1-8 confirm the generated MP4 opens, has 145 frames,
  `1280x720`, `30fps`, first-frame decode OK, and size `1234819` bytes.
- Earlier black FK-proxy render artifacts were rejected because they were not
  actual RoArm geometry. The accepted final render uses the real local URDF STL
  visual meshes from `local_assets/roarm_m3/urdf/meshes`.

## Trace Evidence

Trace generation output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_trace_posx_env3_seed779_stdout.out:20-21`
  confirms local IsaacLab, built-in `DifferentialIKController`, no RoArm-local
  IK loop, no training, no dataset generation, no grasp/attach/posewrite,
  `trajectory_variant=v3`, and the local backup robot USD path.
- `diffik_probe_v3_trace_posx_env3_seed779_summary.json:46-49` confirms the
  trace CSV path, `trace_env_id=3`, `trace_frame_count=145`, and
  `training=false`.
- The trace CSV has 145 rows. First frame cube world position is
  `(3.353596210, 2.926681280, 0.014999384)m`; last frame cube world position is
  `(3.389451504, 2.920480251, 0.021177327)m`.
- Render summary subtracts env origin `(3,3,0)m`, so the local visualization
  starts at approximately `(0.353596210, -0.073318720, 0.014999384)m` and ends
  at approximately `(0.389451504, -0.079519749, 0.021177327)m`.

## Render Evidence

- MP4:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_posx_env3_seed779/diffik_probe_v3_posx_env3_seed779.mp4`
- Frames:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_posx_env3_seed779/frames/frame_0000.png`
  through `frame_0144.png`.
- Render stdout line 447:
  `frames=145`, MP4 path, trace path, `training=NO`,
  `dataset_generation=NO`, `physics_recomputed=NO`.
- Render summary lines 19-27: `fps=30`, `frames_written=145`,
  `height=720`, output directory, output MP4, `physics_recomputed=false`,
  trace CSV, `training=false`, `width=1280`.
- Local MP4 decode check
  `diffik_probe_v3_render_posx_env3_seed779_mp4_probe.out:1-8` confirms
  `opened=True`, `frame_count=145`, `width=1280`, `height=720`, `fps=30.0`,
  first frame decode OK, and file size `722185` bytes.

## Code / Artifact MD5

- `sim_scripts/cube3cm_push_diffik_probe.py`
  `1e39836eb02a22c12e084a4279e6b4e7`
- `sim_scripts/cube3cm_push_diffik_audit.py`
  `5ed85775e31f805f4d43885a1de80246`
- `sim_scripts/cube3cm_push_diffik_posthoc.py`
  `6bfc8ea3eac942d0af4c8fc852738f0e`
- `sim_scripts/cube3cm_push_diffik_render_trace.py`
  `2adb116ae2c441420873a8384d3a7b17`
- Trace CSV:
  `d81e820b0fe3543c9f3e3baae687f304`
- MP4:
  `d6f0c2d2f337b492c1cfbbe6af9733e4`
- Four-direction trace CSV:
  `f8ee20c10bef2495af05835b4b3a6932`
- Four-direction MP4:
  `4936a6f5b17309ad47c829f22f6a907f`

## Verification

- `python3 -m py_compile` passed for
  `cube3cm_push_diffik_probe.py`, `cube3cm_push_diffik_audit.py`,
  `cube3cm_push_diffik_posthoc.py`, and `cube3cm_push_diffik_render_trace.py`.
- `git diff --check` passed.
- A rendered mid-frame was opened locally and showed the robot, cube, TCP marker,
  and target marker; the MP4 decode check also confirmed the video is not an
  empty file.

## Interpretation

This is a professor-facing visualization of a scripted Differential IK physics
trajectory. It is valid to say: "This video replays a successful `(1,0)` v3
scripted IsaacLab Differential IK push/tap sample, with no training, no dataset
generation, no grasp/attach, and no object posewrite."

It is not valid to say it is learned-policy success, dataset readiness, or Track
A grasp success.
