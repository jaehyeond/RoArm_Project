# Session 2026-06-14 - Cube10cm D241 LeRobot + Metadata + Label-Aware Design D240

Status: d241 LeRobot AV1 conversion, local LeRobot load/decode validation,
companion metadata row alignment, PNG extraction proof, and label-aware 0-999
manifest design draft complete.

This session did not run Isaac render, did not generate 0-999 / 1000 / 10000
episodes, did not train, did not delete, move, or archive files, and did not use
SSH/B200.

## Scope

- Branch scope: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- Input render root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d241`
- Out of scope: PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher,
  RoArm deployment, B200/SSH, pull, cleanup, and Track A.

## LeRobot Conversion

Command class:

`sim_scripts/cube10cm_top_view_smoke_to_lerobot.py`

Output:

`cube10cm_top_view_visual_chunk100_d241/lerobot_dataset_av1`

Validation summary:

- status: `PASS`
- total frames: `19500`
- total episodes: `100`
- source frames: `19500`
- frame count match: `true`
- requested codec: `libsvtav1`
- actual codec: `av1`
- pixel format: `yuv420p`
- fps: `30`
- video bytes total: `56604396`
- video MB/episode: `0.56604396`
- projected video size: `0.56604396GB/1000ep`,
  `5.6604396GB/10000ep`
- sampled LeRobot decode avg/max: `0.008618485927581788s` /
  `0.09812450408935547s`
- sampled PNG-vs-decoded MP4 max mean abs diff: `0.8940353732638889`
- sampled PNG-vs-decoded MP4 max pixel abs diff: `74`
- validation elapsed: `14.933778762817383s`

Final LeRobot root size was about `56MB`. The final LeRobot root had `0`
temporary PNG files remaining under the dataset root.

## Companion Metadata

Command class:

`sim_scripts/cube10cm_top_view_metadata_companion.py`

Output:

`cube10cm_top_view_visual_chunk100_d241/metadata_companion_d241`

Validation summary:

- status: `PASS`
- rows: `19500`
- episodes: `100`
- LeRobot validation checked: `true`
- LeRobot total frames: `19500`
- LeRobot total episodes: `100`
- LeRobot core data columns: `observation.state`, `action`, `timestamp`,
  `frame_index`, `episode_index`, `index`, `task_index`

The companion metadata is keyed to the LeRobot core rows by `index`,
`episode_index`, and `frame_index`. This keeps the training loader-compatible
core parquet separate from rich camera/cube/projection/contact audit metadata.

## PNG Extraction Proof

Command class:

`extract_frames.py`

Extracted frame:

`cube10cm_top_view_visual_chunk100_d241/debug_extract_frames_d241/episode_000099_frame_000050.png`

Result:

- Extracted from `lerobot_dataset_av1/videos/observation.images.top/chunk-000/file-000.mp4`
- Local video frame: `19355`
- Resolution: `1280x720`
- Same-frame raw source: `raw_env_render_frames/rgb_019355.png`
- Same-frame source-vs-extracted mean abs diff: `0.7776012731481482`
- Same-frame source-vs-extracted max abs diff: `30`

This proves that primary storage can remain LeRobot MP4 + parquet while an
arbitrary frame can be extracted as PNG for professor-facing inspection.

## Label-Aware 0-999 Design Draft

Added:

`claudedocs/cube10cm_top_view_label_aware_0_999_manifest_design_d240.md`

Critical design conclusion:

- Existing v6 data remains irrelevant as the professor schema target. It was
  useful only as a codec/backend reference.
- D241's `split_candidate` is a sampling bucket, not a final label.
- Actual training/eval filtering must use post-render numeric labels:
  `label_useful_clean_numeric`, `label_overshoot_numeric`, and
  `label_camera_contract_numeric`.
- D241 observed `61/100` useful clean episodes and `39/100` overshoot episodes.
  Therefore, a naive 0-999 render should not be described as 1000 clean
  train-positive demonstrations.
- The high-boundary samples are camera-covered but mostly overshoot:
  `eval_boundary_candidate` produced `1` clean and `14` overshoot episodes.

Draft 1000-episode sampling counts:

- `clean_prior_candidate=650`
- `transition_mixed_probe=200`
- `overshoot_eval_candidate=100`
- `debug_camera_anchor=50`

All rows must still set `requires_posthoc_label_validation=True`; final labels
are assigned only after rendering and numeric validation.

## Still Blocked

- Any 0-999 / 1000 / 10000 Isaac render.
- Any dataset scale-up beyond the existing d241 0-99 render.
- Any deletion, move, archive, or cleanup.
- PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/session_20260613_cube10cm_top_view_chunk100_render_labels_d239.md`
- `claudedocs/cube10cm_top_view_label_aware_0_999_manifest_design_d240.md`
- `sim_scripts/cube10cm_top_view_smoke_to_lerobot.py`
- `sim_scripts/cube10cm_top_view_metadata_companion.py`
- `extract_frames.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d241/lerobot_validation_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d241/metadata_companion_d241/metadata_validation_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d241/postrender_label_validation_d241/episode_labels.csv`
