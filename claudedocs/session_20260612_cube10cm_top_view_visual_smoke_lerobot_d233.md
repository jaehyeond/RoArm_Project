# Session 2026-06-12 - Cube10cm Top-View Visual Smoke + LeRobot Validation D233

## Scope

Approved local smoke only for the professor 10cm / 0.72kg cube top-view visual
trajectory dataset branch.

No B200/SSH, pull, `.ssh` copy, deletion, archive, move, PPO/L2/Large PPO,
VLA/action-teacher, RoArm deployment, Track A, 100 episode chunk, or full
1000/10000 episode dataset generation was performed.

This session follows D232:

- camera contract first;
- raw image target `1280x720`;
- LeRobot MP4+parquet as primary storage;
- PNG only for smoke/debug/extraction;
- full scale-up blocked by reprojection, occlusion, codec, fps/render-time,
  LeRobot load, professor confirmation, and disk/storage gates.

## Important Format Interpretation

Existing `lerobot_dataset_v6` is not the professor-requested data schema. It was
used only as a local read-only codec/backend fixture because it already contains
LeRobot AV1 video at `1280x720`.

The current dataset contract is governed by the professor 2026-06-11 top-view
visual trajectory requirement and D232 camera contract:

- frame-by-frame image-state pairs remain intact;
- primary storage is LeRobot video+parquet;
- arbitrary frames can be extracted to PNG immediately;
- `observation.images.top` is the new top-view RGB key for this branch.

## Scripts Added

- `sim_scripts/cube10cm_top_view_visual_smoke_render.py`
  - IsaacLab local render smoke.
  - Headless camera render only.
  - Uses `RoArm-CubeTap10cm-Direct-v0` and the Candidate6 tap/push contract as a
    scripted/base trajectory source.
  - Writes debug PNG frames plus `frames.jsonl` metadata.
  - Does not train, load PPO checkpoints, delete files, or generate a large
    dataset.
- `sim_scripts/cube10cm_top_view_smoke_to_lerobot.py`
  - Converts the smoke PNG/JSONL output to LeRobot video+parquet.
  - Validates LeRobot load/decode through `LeRobotDataset`, not OpenCV.
  - Measures source PNG vs decoded MP4 pixel difference.
- `extract_frames.py`
  - Extracts an arbitrary PNG by `episode_id` and `frame_id`.
  - Uses ffmpeg/imageio-ffmpeg, which is the right path for AV1 compatibility.

## Commands

Render command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube10cm_top_view_visual_smoke_render.py --num-episodes 5 --steps-per-episode 580 --capture-stride 3
```

LeRobot conversion and validation command:

```bash
conda run -n roarm python -u sim_scripts/cube10cm_top_view_smoke_to_lerobot.py --render-dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232 --out-dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_dataset_av1 --quality-samples 5 --vcodec libsvtav1
```

Frame extraction proof command:

```bash
conda run -n roarm python extract_frames.py --dataset claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_dataset_av1 --episode-id 3 --frame-id 50 --out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/extract_ep003_frame050.png
```

## Render Result

Output root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232`

Source summary:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/render_summary.json`

Measured render facts:

- status: local smoke render completed;
- runtime class: `ISAAC_RENDER_ONLY_NO_TRAINING_NO_SCALEUP`;
- camera contract id: `cube10cm_top_view_v1_candidate`;
- episodes: `5`;
- frames: `975`;
- frames per episode: `195`;
- resolution: `1280x720`;
- target fps metadata: `30`;
- elapsed render time: `180.79416966438293s`;
- effective render throughput: `5.392873021347648` captured frames/sec;
- cube poses:
  - `(0.24, 0.0)`;
  - `(0.14, -0.10)`;
  - `(0.14, 0.10)`;
  - `(0.34, -0.10)`;
  - `(0.34, 0.10)`;
- contract violations: `[]`.

Reprojection and occlusion:

- reprojection centroid error median: `3.074639061891291px`;
- reprojection centroid error max: `9.956731449704932px`;
- reprojection samples: `975`;
- all-frame visibility:
  - `cube_visible_full=975`;
  - `cube_visible_partial=0`;
  - `cube_occluded_full=0`;
- contact-window frames: `882`;
- contact-window visibility:
  - `cube_visible_full=882`;
  - `cube_visible_partial=0`;
  - `cube_occluded_full=0`.

Debug PNG storage:

- PNG bytes total: `261672389`;
- debug PNG MB/episode: `52.3344778`;
- projected PNG storage:
  - 100 episodes: `5.2334477800000005GB`;
  - 1000 episodes: `52.3344778GB`;
  - 10000 episodes: `523.344778GB`.

Interpretation:

- Camera candidate v1 passes the local 5-episode reprojection/visibility smoke.
- PNG-at-scale is rejected by measured size. It remains debug/extraction only.
- Render throughput is slow enough that local scale-up should not start casually.

## LeRobot Result

Validation summary:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_validation_summary.json`

Dataset root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_dataset_av1`

Measured LeRobot facts:

- validation status: `PASS`;
- video key: `observation.images.top`;
- requested codec: `libsvtav1`;
- actual codec: `av1`;
- pixel format: `yuv420p`;
- fps: `30`;
- total frames: `975`;
- total episodes: `5`;
- frame count match: `true`;
- video file: `videos/observation.images.top/chunk-000/file-000.mp4`;
- video bytes total: `2982439`;
- video MB/episode: `0.5964878`;
- projected video storage:
  - 100 episodes: `0.059648780000000005GB`;
  - 1000 episodes: `0.5964878GB`;
  - 10000 episodes: `5.964878GB`;
- sampled LeRobot decode average: `0.016793251037597656s`;
- sampled LeRobot decode max: `0.06672263145446777s`;
- sampled decoded image shape: `[720, 1280, 3]`;
- sampled `observation.state` shape: `[6]`;
- sampled `action` shape: `[6]`;
- source PNG vs decoded MP4 sampled mean abs diff max:
  `0.8939572482638889`;
- source PNG vs decoded MP4 sampled max abs diff max: `67`.

Interpretation:

- AV1 is locally acceptable through the installed LeRobot backend for this smoke.
- OpenCV AV1 failure is not a blocker for training if LeRobot/torchcodec/pyav can
  decode in the actual training environment.
- Codec is not globally closed until RunPod/H100 uses the same dataloader path.
  If RunPod fails AV1 decode or speed, switch the new dataset codec to H.264.

## PNG Extraction Proof

`extract_frames.py` successfully extracted:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/extract_ep003_frame050.png`

The extracted frame is a `1280x720` PNG from episode `3`, frame `50`, proving
the professor-facing claim that arbitrary PNGs can be produced immediately from
the LeRobot MP4+parquet dataset.

## Disk State After Smoke

After render and conversion, local filesystem was approximately:

- total: `590G`;
- used: `529G`;
- free: `32G`;
- use: `95%`.

Interpretation:

- Local disk pressure is worse than the D232 `39G` free audit.
- Do not start a 100 episode chunk locally without an approved output root or
  approved cleanup/archive action.
- Do not delete/move files opportunistically. Follow the D232 cleanup order.

## Side Effects Left In Place

The following artifacts were created and intentionally not deleted because the
session rule forbids unapproved deletion/move:

- final smoke root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232`;
- sanity roots:
  - `cube10cm_top_view_visual_smoke_d232_sanity`;
  - `cube10cm_top_view_visual_smoke_d232_sanity_envrender`;
  - `cube10cm_top_view_visual_smoke_d232_sanity_envrender2`;
- an empty failed-conversion directory under the final smoke root:
  `lerobot_dataset`;
- mistaken early BasicWriter output under:
  `/home/cgxr/omni.replicator_out/...`.

## Decision

Verdict:

`TOP_VIEW_CAMERA_CONTRACT_V1_LOCAL_SMOKE_PASS_LEROBOT_AV1_LOCAL_PASS_SCALEUP_BLOCKED`

Pass:

- 5-episode local render completed.
- Reprojection/centroid sanity passed for the smoke.
- No cube full/partial occlusion in all frames or contact-window frames.
- LeRobot video+parquet conversion passed.
- LeRobot dataloader decode passed locally.
- PNG extraction from MP4+parquet passed.
- AV1 is locally usable through LeRobot for this smoke.

Blocked:

- Professor view/format confirmation before 100 episode chunk.
- RunPod/H100 LeRobot AV1 dataloader decode/speed verification before using AV1
  for scale-up there.
- Storage decision before local 100 episode chunk; local disk is about `32G`
  free after smoke.
- 100 episode chunk requires explicit approval.
- 1000/10000 episode generation remains blocked.
- Any deletion/archive/move remains blocked without explicit approval.
- PPO/L2/Large PPO, VLA/action-teacher, RoArm deployment, SSH/B200, pull, and
  Track A remain out of scope.

## Sources

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/render_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_validation_summary.json`
- `sim_scripts/cube10cm_top_view_visual_smoke_render.py`
- `sim_scripts/cube10cm_top_view_smoke_to_lerobot.py`
- `extract_frames.py`
- `claudedocs/camera_contract_cube10cm_top_view_d232.md`
- `claudedocs/storage_plan_cube10cm_visual_dataset_d232.md`
