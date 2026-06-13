# Cube10cm Top-View Visual Dataset Schema D235

Status: schema/metadata plan for the next 100 episode chunk. This file does not
authorize IsaacLab rendering, dataset generation, deletion, archive, move,
training, RoArm deployment, SSH/B200, pull, or Track A work.

## Verified Baseline

- Current branch remains the professor 10cm / 0.72kg cube top-view visual
  trajectory dataset branch.
- D233 smoke produced 5 episodes / 975 frames at `1280x720`.
- D233 LeRobot output stores RGB as `observation.images.top` video and stores
  `observation.state`, `action`, and index/timestamp columns in LeRobot parquet.
- D234 selected AV1 for the next chunk by local and RunPod/H100 LeRobot
  dataloader evidence.
- The 100 episode chunk has not been run. 1000/10000 remain blocked by storage
  and scale-up gates.

## Storage Split

Keep the scale-up dataset in two linked layers:

1. LeRobot core dataset:
   - `videos/observation.images.top/...mp4`
   - `data/chunk-*/file-*.parquet`
   - features: `observation.images.top`, `observation.state`, `action`,
     `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`
   - purpose: standard dataloader compatibility for later SmolVLA/VLA work

2. Companion metadata:
   - `metadata_companion/per_frame_metadata.parquet`
   - `metadata_companion/episode_metadata.parquet`
   - `metadata_companion/metadata_schema.json`
   - `metadata_companion/metadata_validation_summary.json`
   - primary join key: `global_index`
   - secondary join key: (`episode_index`, `frame_index`)
   - purpose: reproducibility, camera/projection audit, split design, professor
     review, and debugging without expanding the model-training feature set

Do not put every rich metadata field into the LeRobot core parquet by default.
That would increase coupling to LeRobot loader behavior and later training code.
The safer default is a standard LeRobot core plus a companion table keyed to the
same frame indices.

## LeRobot Core Fields

Required:

- `observation.images.top`: RGB video, shape `[720, 1280, 3]`, fps `30`
- `observation.state`: float32, shape `[6]`, RoArm joint positions
- `action`: float32, shape `[6]`, RoArm joint targets
- `timestamp`: float32
- `frame_index`: int64, per-episode frame id
- `episode_index`: int64
- `index`: int64 global frame index
- `task_index`: int64

Policy:

- Preserve `observation.images.top` as the single top-view RGB key.
- Keep raw storage at `1280x720`; any `224x224` resize belongs to later model
  preprocessing.
- Keep PNG for smoke/debug/extraction only.

## Companion Per-Frame Fields

Required scalar fields:

- `global_index`
- `episode_index`
- `frame_index`
- `sim_step`
- `timestamp_s`
- `sim_time_s`
- `source_png`
- `camera_contract_id`
- `camera_path`
- `camera_center_world_x`
- `camera_center_world_y`
- `camera_center_world_z`
- `camera_height_above_table_m`
- `image_width`
- `image_height`
- `image_convention`
- `cube_position_world_x`
- `cube_position_world_y`
- `cube_position_world_z`
- `cube_quat_w`
- `cube_quat_x`
- `cube_quat_y`
- `cube_quat_z`
- `cube_linear_velocity_x`
- `cube_linear_velocity_y`
- `cube_linear_velocity_z`
- `tcp_position_world_x`
- `tcp_position_world_y`
- `tcp_position_world_z`
- `target_position_world_x`
- `target_position_world_y`
- `target_position_world_z`
- `push_dir_x`
- `push_dir_y`
- `tap_contact_proxy`
- `tap_contact_seen`
- `tap_reaction_seen`
- `tap_overshoot_seen`
- `tap_success_flag`
- `tap_disp_along_m`
- `tap_disp_xy_m`
- `tap_speed_mps`
- `cube_visibility`
- `blue_coverage`
- `blue_pixels`
- `bbox_area`
- `centroid_error_px`
- `projection_center_u`
- `projection_center_v`
- `projection_bbox_x0`
- `projection_bbox_y0`
- `projection_bbox_x1`
- `projection_bbox_y1`

Optional JSON-string fields:

- `projection_uv_corners_json`

## Companion Episode Fields

Required:

- `episode_index`
- `num_frames`
- `first_global_index`
- `last_global_index`
- `first_sim_step`
- `last_sim_step`
- `first_cube_x`
- `first_cube_y`
- `first_cube_z`
- `last_cube_x`
- `last_cube_y`
- `last_cube_z`
- `full_visibility_frames`
- `partial_visibility_frames`
- `full_occlusion_frames`
- `contact_seen_any`
- `reaction_seen_any`
- `overshoot_seen_any`

## Validation Gate Before 0-99 Render

Before launching a new 100 episode chunk:

1. Confirm free disk space is at least the D234 local threshold.
2. Confirm output root is fresh and does not contain existing PNG/video/parquet.
3. Keep the D233 smoke script capped at 10 episodes; use a separate chunk script
   or explicit chunk mode for 100 episodes.
4. Keep LeRobot core fields standard.
5. Write companion metadata after render and conversion.
6. Validate row alignment:
   - `len(frames.jsonl) == LeRobot total_frames`
   - companion row count equals LeRobot row count
   - `episode_index`, `frame_index`, and `global_index` align
7. Validate video:
   - MP4 frame count matches metadata
   - sampled LeRobot decode works
   - sampled PNG extraction works
   - sampled source PNG vs decoded MP4 difference is reported

## Blocked Until Explicit Approval

- Running IsaacLab render
- Generating the 0-99 chunk
- Generating 0-999 or 1000/10000 episodes
- Deleting, moving, or archiving files
- PPO/L2/Large PPO
- SmolVLA/VLA fine-tuning
- action-teacher work
- RoArm deployment
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy
