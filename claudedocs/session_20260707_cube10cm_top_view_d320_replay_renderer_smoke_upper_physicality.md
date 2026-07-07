# D320 replay renderer smoke + upper-bin physicality audit

Date: 2026-07-07 KST

Verdict: `D320_REPLAY_RENDER_LEROBOT_PASS_UPPER_MIXED_DIRECTION_UNSTABLE`

## Scope

This session followed `CLAUDE.md` Current-State Protocol for the professor
10cm / 0.72kg cube top-view branch. It did not use B200/SSH/pull/RoArm/VLA/PPO,
and it did not add controller hand-conditions.

The failable experiment was the D319 replay-render smoke and LeRobot conversion
path. Failure conditions were render, conversion, or dataloader load failure.

## Code Added

- `sim_scripts/cube10cm_top_view_d320_prepare_replay_manifest.py`
  - Builds a 9-episode smoke manifest from D319 env-level conveyor rows.
  - Selects 2 accepted rows from each bin plus 3 upper-bin overshoot rows.
  - Writes the upper-bin physicality audit.
- `sim_scripts/cube10cm_top_view_d320_replay_render.py`
  - Replays D319 trajectories with D256 reset episode, friction override,
    candidate8 zero residual action, and `candidate8_hybrid_stop_after_useful`.
  - Uses reset-time cube material randomization to apply the selected friction
    values per replay episode.
- `sim_scripts/cube10cm_top_view_d320_lerobot_batch_check.py`
  - Loads one `torch.utils.data.DataLoader` batch from the resulting LeRobot
    dataset.
- `sim_scripts/cube10cm_top_view_d320_direction_probe.py`
  - Checks whether existing `fixed_push_dir_x/y` and candidate8 target-residual
    code can execute non-+x push directions.

## Runtime Results

### Replay Render

Initial sandboxed render failed before env creation because Isaac could not see
CUDA/NVML:

- `RuntimeError: No CUDA GPUs are available`
- `NVML_ERROR_DRIVER_NOT_LOADED`

This was a sandbox/GPU visibility issue, not a renderer-contract result.
`nvidia-smi` outside the sandbox saw the local RTX 4090. The same D320 render
was rerun with local GPU access and passed.

Render output:

- Render root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/render_d319_replay_smoke`
- Manifest:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/d320_replay_smoke_manifest.csv`
- Summary:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/render_d319_replay_smoke/render_summary.json`
- Frames:
  `9` episodes, `1314` frames, `146` frames/episode at capture stride `4`.
- Raw PNG size:
  `344057130` bytes total, `38.22857 MB/episode`.
- Render time:
  `202.47s`, effective captured FPS `6.49`.

Replay final metrics showed the low/mid bins reproduced D319 closely, while the
upper bin stayed dynamics-sensitive:

| d320 ep | role | d319 ep | D319 XY mm | replay XY mm | replay overshoot |
|---:|---|---:|---:|---:|---:|
| 0 | low accepted | 568 | 7.46 | 7.46 | 0 |
| 1 | low accepted | 335 | 7.67 | 7.66 | 0 |
| 2 | mid accepted | 206 | 12.26 | 12.26 | 0 |
| 3 | mid accepted | 109 | 12.21 | 11.12 | 0 |
| 4 | upper accepted | 939 | 19.11 | 18.50 | 0 |
| 5 | upper accepted | 422 | 18.09 | 160.70 | 1 |
| 6 | upper overshoot | 679 | 31.66 | 0.21 | 0 |
| 7 | upper overshoot | 997 | 22.41 | 18.47 | 0 |
| 8 | upper overshoot | 991 | 36.47 | 115.99 | 1 |

Interpretation: D319 replay rendering is viable, but upper-bin rows are not
deterministically stable enough to treat accepted/failure labels as immutable
without replay-side validation.

### LeRobot Conversion + Dataloader

LeRobot conversion failed in `isaaclab` because that conda env does not include
`lerobot`. It passed in the repo's `lerobot` env with `video_backend=pyav`, which
matches the D247 local backend contract.

- Dataset root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/render_d319_replay_smoke/lerobot_dataset`
- Validation:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/render_d319_replay_smoke/lerobot_validation_summary.json`
- Dataloader batch:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/render_d319_replay_smoke/dataloader_batch_validation_d320.json`

Conversion/load result:

- Status: `PASS`
- Frames/episodes: `1314 / 9`
- Codec: `av1`, `yuv420p`, `30fps`
- Video bytes: `3763147`
- Video size: `0.418127 MB/episode`
- Decode avg/max: `0.01177s / 0.02086s`
- PNG-vs-decoded sampled mean abs max: `0.88714`
- DataLoader first batch keys:
  `action`, `episode_index`, `frame_index`, `index`, `observation.images.top`,
  `observation.state`, `task`, `task_index`, `timestamp`
- Batch shapes:
  - `observation.images.top`: `[2, 3, 720, 1280]`
  - `observation.state`: `[2, 6]`
  - `action`: `[2, 6]`

## Upper-bin Physicality Audit

Audit output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/d320_upper_bin_physicality_audit.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/replay_smoke/d320_upper_bin_physicality_audit.md`

Pre-registered rule:

- Mostly `<300mm`: physical failure, RL contribution target.
- Meter-scale present: solver contamination present, isolate outliers before
  using upper bin.

Actual D319 upper-bin overshoot distribution:

- Overshoot rows: `242`
- `<300mm`: `236/242` (`97.52%`)
- `>=1m`: `6/242` (`2.48%`)
- Quantiles:
  - min `20.07mm`
  - p50 `30.94mm`
  - p90 `40.46mm`
  - p95 `45.58mm`
  - p99 `11124.63mm`
  - max `11140.39mm`

Decision: `MIXED_PHYSICAL_FAILURE_WITH_SOLVER_OUTLIERS`.

Interpretation: the upper `1.2-1.6` bin remains a valid RL contribution target
for most failures, but meter-scale rows must be isolated before scale-up or
training/evaluation claims.

## Direction Diversity Probe

Code support exists:

- `roarm_rl/roarm_cube_push_env.py:1593-1605` supports fixed push direction when
  D256 reset is disabled.
- `roarm_rl/roarm_cube_push_env.py:1226-1238` uses `push_dir` and its lateral
  basis in candidate8 target residuals.

Runtime probe:

- Output:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/direction_probe/direction_probe_summary.json`
- Setup:
  no D256 reset, `5` envs per direction, fixed directions `-x`, `+y`, `-y`,
  candidate8 zero residual action, hybrid stop enabled.

Results:

| direction | contact | reaction | useful filter | overshoot |
|---|---:|---:|---:|---:|
| `-x` | 5/5 | 5/5 | 3/5 | 2/5 |
| `+y` | 5/5 | 5/5 | 1/5 | 4/5 |
| `-y` | 5/5 | 5/5 | 3/5 | 2/5 |

Interpretation: non-+x direction control is possible in code, but the current
zero-action primitive is not robust for those directions. Direction diversity
requires an explicit direction-conditioned generator or learned primitive
parameter; it should not be claimed from the D319 +x conveyor.

## GPU / Process Check

- `nvidia-smi` after D320:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d320/nvidia_smi_after_d320.txt`
- GPU state: RTX 4090 visible, memory `25MiB`, GPU util `19%`, only Xorg listed.
- Process check with `ps -C python/python3/torchrun/tensorboard` and
  `ps -C kit/isaacsim`: no remaining Python, Isaac, torchrun, or TensorBoard
  processes.

## Decision

- D319 replay-render smoke path is now implemented and passed.
- LeRobot v3 conversion and one-batch dataloader load passed.
- Low/mid producer bins can use this replay-render path for future scale-up,
  subject to disk and explicit scale-up approval.
- Upper `1.2-1.6` remains the RL contribution target, but the target must be
  split into normal physical failures (`20-50mm` dominant) versus meter-scale
  solver outliers.
- Direction diversity is not solved by D319; code has direction parameters, but
  runtime robustness is currently poor outside +x.
- No PPO, VLA, RoArm, B200, or controller hand-condition was run.
