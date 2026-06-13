# Session 2026-06-12 - D232 Camera Contract + Visual Dataset Disk Gate

## Scope

Local repo/documentation audit only.

No file deletion, no runtime, no Isaac Sim render, no PPO, no L2/Large PPO, no
dataset generation, no VLA/action-teacher, no RoArm control, no SSH/B200, no
pull, and no Track A work.

## Current Decision

Professor feedback changes the next practical work from PPO promotion to a
camera-calibrated visual trajectory dataset path:

- Build a 10cm cube tap/push visual trajectory generator only after a camera
  contract is fixed.
- First milestone is a 5-10 episode smoke render, not 1000 episodes.
- The smoke render must prove that the top-view Isaac camera is physically
  reproducible with the real Azure Kinect setup.
- Full 1000/10000 episode generation remains blocked until camera contract,
  reprojection, occlusion, codec, fps, render-time, and disk gates pass.

## Camera Contract Requirements

Use the Azure Kinect intrinsic values as camera-intrinsic candidates only:

- `sim_scripts/kinect_calib.yaml` line 1 records `fx=608.33`, `fy=608.28`,
  `cx=638.31`, `cy=365.26`, `width=1280`, `height=720`.
- The same file records old hand-eye extrinsics on lines 2-8. Those extrinsics
  are not the new top-view setup. They must not be treated as the new dataset
  camera pose.

The new camera contract must specify:

- physical mounting method;
- table-to-camera height;
- camera roll/pitch/yaw and whether the camera is inverted;
- image flip/crop convention if inverted;
- RGB resolution, target fps, and depth mode policy;
- workspace bounds visible in image;
- robot/cube/table coverage;
- self-occlusion metrics for frames where the arm/tool covers the cube;
- original storage resolution and model resize rule.

`224x224` is a model preprocessing size, not the raw dataset storage size. Raw
render/video should stay at the Azure-Kinect-compatible `1280x720` contract
unless deliberately revised.

## Required Smoke Checks

Before any 1000 episode generation, run only a small approved smoke render and
measure:

1. Reprojection sanity: project known sim markers/cube corners with the selected
   intrinsics/extrinsics and compare against rendered pixels.
2. Self-occlusion: report cube fully visible / partially occluded / fully
   occluded frame rates.
3. Render speed: seconds per episode and effective rendered fps.
4. Storage cost: MB per episode and projected GB for 100/1000/10000 episodes.
5. Codec check: compare decoded MP4 frames against source PNG/debug frames.
6. LeRobot load check: generated video/parquet must load under the installed
   LeRobot API.

## Format Decision

Use a LeRobot-style visual trajectory layout rather than a raw PNG dataset:

- `observation.images.top` video at raw camera resolution;
- per-frame state/action/object/camera metadata in parquet;
- debug PNGs only for smoke/inspection, not as the full storage format.

Repo precedent:

- `convert_to_lerobot_v3.py` defines `observation.images.top` as video with
  shape `(720, 1280, 3)`.
- `sim_scripts/sim_to_lerobot.py` already uses a video path under
  `videos/observation.images.top/...`.
- `scripts/render_p6v12_trajectory_replay.py` shows the existing Kinect
  intrinsics to USD camera conversion pattern.

## Dataset Split Decision

Do not mix all samples into one undifferentiated train set.

Use explicit subsets:

- `train_success`: clean scripted/base tap-push trajectories for imitation.
- `eval_boundary`: perturbation boundary poses discovered in D225-D228.
- `eval_failure`: failure/overshoot regions for robustness evaluation.
- `debug_smoke`: tiny camera-contract smoke episodes.

Sampling regions and seeds must be recorded in metadata. Previous PPO/pose-bin
work is useful as dataset split design input, but it does not unblock PPO or
large dataset generation by itself.

## Disk Audit

Current local disk state from `df -h` on 2026-06-12:

- filesystem is 590G total, 522G used, 39G available, 94% used.
- `/home/cgxr/Documents/Robotics` is about 270G.
- `RoArm_Project` is about 269G, so the pressure is inside this project.

Largest inspected project directories:

- `outputs`: 96G, mostly SmolVLA checkpoint directories.
- `claudedocs`: 35G, dominated by `claudedocs/figures/p6v12_rollout/frames`
  at 34G with 73969 frame files; `p6v12_rollout.mp4` is only about 193KB.
- `collected_data`: 26G.
- `collected_data_v5`: 26G.
- `collected_data_v2_backup`: 19G.
- `b200_backup_20260521`: 19G.
- `b200_backup_20260522_final`: 18G.
- `collected_data_v6`: 13G.
- `openvla_oft_b200_pulls`: 8G.
- `sim_renders_v2/v3/v4/v5`: about 7.9G combined.

Cleanup candidates, pending explicit approval:

1. Highest-value delete/archive candidate:
   `claudedocs/figures/p6v12_rollout/frames` (34G raw/debug frames). Preserve
   compact summaries/video if needed.
2. SmolVLA `outputs/` preservation rule:
   keep `outputs/` by default. It contains SmolVLA results, so do not delete it
   just because it is large.
3. If disk pressure later requires touching `outputs/`, do not re-scan randomly
   or delete arbitrary runs. Use this fixed order only after explicit approval
   and manifest:
   - First remove/archive only `outputs/*/checkpoints/*/training_state`.
     Estimated reclaim: about 25.6GB decimal. `pretrained_model` inference
     artifacts remain; training resume state is lost.
   - If more space is needed, prune to one representative checkpoint per run:
     `smolvla_official=050000`, `smolvla_v2_cleaned=050000`,
     `smolvla_v3_sponge=050000`, `smolvla_v5_multipos=200000`,
     `smolvla_v6=last`, `smolvla_v6_b200=last`,
     `smolvla_v6_stacking_b200=last`,
     `smolvla_v6_stacking_v2_b200=010000`,
     `smolvla_v6_stacking_v3_b200=020000`.
     Estimated reclaim: about 90.15GB decimal total; old four large runs alone
     can reclaim about 74.1GB decimal.
4. Archive/move only with explicit approval, not blind-delete and not cleanup
   priority:
   `collected_data*`, `b200_backup_*`, and `openvla_oft_b200_pulls`. These are
   needed data/backups.
5. Lower-value cleanup:
   obsolete dryrun render folders and explicitly named `*_DISCARD` data, but
   only after confirming no current doc depends on them.

Do not start 100/1000 episode visual generation with only 39G free. Target at
least 100G free or an external/RunPod storage path before full generation.

## Active Next Step

1. Write/approve the camera contract.
2. Free or provision storage for smoke and chunked generation.
3. Implement/run only a 5-10 episode smoke render with `1280x720`, 30fps target,
   video+parquet output, reprojection check, occlusion report, and MB/sec
   measurements.
4. Review smoke output visually and numerically before any 1000 episode render.

## Sources Checked

- `sim_scripts/kinect_calib.yaml`
- `CLAUDE.md`
- `scripts/render_p6v12_trajectory_replay.py`
- `convert_to_lerobot_v3.py`
- `sim_scripts/sim_to_lerobot.py`
- local `df`, `du`, and `find` inspection on 2026-06-12
