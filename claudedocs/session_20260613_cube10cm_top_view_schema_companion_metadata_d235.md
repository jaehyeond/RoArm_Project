# Session 2026-06-13 - Cube10cm Top-View Schema + Companion Metadata D235

## Scope

User asked to proceed step-by-step after questioning whether parquet/rich
metadata were being handled correctly.

This session did not run IsaacLab render, generate a 100 episode chunk, generate
1000/10000 episodes, delete/archive/move local files, train PPO/L2/Large PPO,
start SmolVLA/VLA fine-tuning, start action-teacher work, deploy to RoArm, use
SSH JHPark/B200, pull from B200, copy `.ssh`, or mix with Track A.

## Verified Starting Truth

- `START_HERE.md` D234 said the active branch was professor 10cm / 0.72kg cube
  top-view visual trajectory dataset camera-contract.
- D234 said the 100 episode chunk had not been run and should only launch with a
  fresh explicit run instruction.
- D234 selected AV1 for the next 100 episode chunk by local + RunPod/H100
  LeRobot dataloader evidence.
- `sim_scripts/cube10cm_top_view_visual_smoke_render.py` still caps
  `--num-episodes` to `[1, 10]`, so it remains a smoke script, not a 100/1000
  renderer.

## Format Recheck

Actual D233 smoke LeRobot dataset:

- video file:
  `videos/observation.images.top/chunk-000/file-000.mp4`
- core data parquet:
  `data/chunk-000/file-000.parquet`
- core data parquet columns:
  `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`,
  `index`, `task_index`
- shape:
  `975` rows / `7` columns

Interpretation:

- LeRobot core already satisfies the frame-indexed image/state/action structure.
- The image is not stored as PNG per row; it is stored as MP4 and indexed through
  LeRobot metadata.
- The existing render `frames.jsonl` contains richer camera/cube/projection/
  visibility/contact metadata that is useful for audit and reproducibility, but
  most of it is not currently in the LeRobot core parquet.

## D235 Decision

Use a two-layer dataset layout:

1. Standard LeRobot core:
   - `observation.images.top` MP4
   - `observation.state`
   - `action`
   - timestamp/index columns
2. Companion metadata:
   - per-frame camera/cube/projection/visibility/contact fields
   - joined to LeRobot core by `global_index`, `episode_index`, and `frame_index`

Do not put all rich metadata into the LeRobot core parquet by default. That would
increase coupling to LeRobot loader/training behavior before there is a concrete
need. The safer scale-up default is standard LeRobot core plus companion metadata.

## Files Added

- `claudedocs/cube10cm_top_view_visual_dataset_schema_d235.md`
- `sim_scripts/cube10cm_top_view_metadata_companion.py`

## Validation Run

Command:

```bash
conda run -n roarm python -u sim_scripts/cube10cm_top_view_metadata_companion.py
```

Result:

- status: `PASS`
- runtime: `NO_RENDER_NO_DATASET_GENERATION_NO_TRAINING`
- rows: `975`
- episodes: `5`
- LeRobot core validation checked:
  - `total_frames=975`
  - `total_episodes=5`
  - core columns:
    `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`,
    `index`, `task_index`
  - companion `global_index`, `episode_index`, and `frame_index` aligned with
    LeRobot core rows

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/metadata_companion_d235/per_frame_metadata.parquet`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/metadata_companion_d235/episode_metadata.parquet`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/metadata_companion_d235/metadata_schema.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/metadata_companion_d235/metadata_validation_summary.json`

Per-frame companion metadata:

- shape: `975 x 54`
- key fields include:
  `global_index`, `episode_index`, `frame_index`, `camera_contract_id`,
  `cube_position_world_*`, `tcp_position_world_*`, `target_position_world_*`,
  `cube_visibility`, `centroid_error_px`, and projection bbox/center fields.

Episode companion metadata:

- shape: `5 x 18`
- each smoke episode has `195` frames.
- all five smoke episodes have full visibility for all frames.

## Current Next Step

Next valid implementation step, if explicitly approved later:

1. Preflight disk and output root for D235 chunk100.
2. Add a separate chunk renderer or explicit chunk mode without weakening the
   smoke script's 1-10 episode guard.
3. Run only 0-99 first, not 0-999.
4. Convert to LeRobot AV1 core.
5. Generate companion metadata.
6. Validate LeRobot load/decode, PNG extraction, source-vs-decoded pixel
   difference, row alignment, storage projection, and visibility/reprojection.

Still blocked without explicit approval:

- IsaacLab render
- 0-99 generation
- 0-999 / 1000 / 10000 expansion
- deletion/archive/move
- PPO/L2/Large PPO
- SmolVLA/VLA fine-tuning
- action-teacher work
- RoArm deployment
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy
