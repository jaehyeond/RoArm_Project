# Session 2026-06-16 - Cube10cm 0-999 LeRobot + metadata D247

## Scope

- Active branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- User approved proceeding with LeRobot conversion and leaving current storage
  as-is.
- This session converted the existing D246 raw render to LeRobot v3 and generated
  validation/metadata/extraction artifacts.
- No deletion, move, archive, Isaac render, PPO, L2, Large PPO, VLA/SmolVLA
  fine-tuning, action-teacher, RoArm deployment, RunPod runtime, B200/SSH/pull,
  `.ssh` copy, or Track A work was run.

## Inputs

Raw render root:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242
```

D246 label state:

- `1000` episodes / `195000` frames.
- Camera-gated usable labels: `819` clean useful taps and `167` overshoot taps.
- Camera-quality failures: `14`.

## LeRobot Conversion

Output:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/lerobot_dataset_av1_d247
```

Build command class:

```bash
conda run -n lerobot --no-capture-output python -u sim_scripts/cube10cm_top_view_smoke_to_lerobot.py --render-dir .../cube10cm_top_view_visual_0_999_d242 --out-dir .../lerobot_dataset_av1_d247 --repo-id roarm_cube10cm_top_view_0_999_d247 --vcodec libsvtav1 --quality-samples 10
```

Build result:

- All `1000` episodes were saved.
- LeRobot root size: about `540M`.
- Video files:
  - `file-000.mp4`: `209608404` bytes
  - `file-001.mp4`: `209520878` bytes
  - `file-002.mp4`: `129441901` bytes

Initial validation caveat:

- The first combined build+validation command exited nonzero after the build
  because the default `lerobot` env validation path tried to load torchcodec.
- Local `torchcodec` failed against `torch==2.10.0+cu128` and missing FFmpeg
  shared libraries such as `libavutil.so.58/59/60`.
- This was a local decoder/backend problem after the dataset had already been
  built; it was not evidence that the 1000-episode LeRobot dataset was missing.

## Validation Fix

Patched:

```text
sim_scripts/cube10cm_top_view_smoke_to_lerobot.py
```

Patch purpose:

- Add `--validate-only` so an existing dataset can be validated without rebuild.
- Add `--video-backend` so local validation can use `pyav` instead of broken
  default torchcodec.

Validation command class:

```bash
HF_HOME=/tmp/roarm_hf_cache HF_DATASETS_CACHE=/tmp/roarm_hf_datasets_cache conda run -n lerobot --no-capture-output python -u sim_scripts/cube10cm_top_view_smoke_to_lerobot.py --validate-only --video-backend pyav --render-dir .../cube10cm_top_view_visual_0_999_d242 --out-dir .../lerobot_dataset_av1_d247 --repo-id roarm_cube10cm_top_view_0_999_d247 --vcodec libsvtav1 --quality-samples 10
```

Validation result:

- status: `PASS`
- total frames: `195000`
- total episodes: `1000`
- frame count match: `true`
- video key: `observation.images.top`
- codec: `av1`
- pixel format: `yuv420p`
- fps: `30`
- video bytes total: `548571183`
- video MB/episode: `0.548571183`
- sampled decode avg/max: `0.015330815315246582s` /
  `0.017406463623046875s`
- sampled PNG-vs-decoded mean abs max: `0.898435691550926`
- sampled max pixel abs diff: `80`

Independent PyAV video frame count:

- `file-000.mp4`: `67275` frames
- `file-001.mp4`: `87945` frames
- `file-002.mp4`: `39780` frames
- total: `195000` frames
- resolution/fps: `1280x720@30fps`
- status: `PASS`

## Companion Metadata

Output:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/metadata_companion_d247
```

Result:

- status: `PASS`
- rows: `195000`
- episodes: `1000`
- LeRobot core row alignment checked: `true`
- core columns: `observation.state`, `action`, `timestamp`, `frame_index`,
  `episode_index`, `index`, `task_index`
- companion size: about `34M`

## PNG Extraction

Command class:

```bash
conda run -n lerobot --no-capture-output python -u extract_frames.py --dataset .../lerobot_dataset_av1_d247 --episode-id 999 --frame-id 194 --out .../debug_extract_frames_d247/episode_000999_frame_000194.png
```

Result:

- extracted from `file-002.mp4`
- local video frame: `39779`
- output PNG: `1280x720`, RGB
- raw source: `raw_env_render_frames/rgb_194999.png`
- source-vs-extracted mean abs diff: `0.792978515625`
- source-vs-extracted max abs diff: `31`
- status: `PASS`

## Storage And Runtime State

- `lerobot_dataset_av1_d247`: about `540M`
- `metadata_companion_d247`: about `34M`
- `debug_extract_frames_d247`: about `104K`
- full render root remains about `49G`
- `df -h .`: about `590G` total, `528G` used, `32G` available, `95%` used
- GPU stayed at baseline; this was CPU/PyAV/SVT-AV1 encoding and file I/O, not
  a CUDA training/render workload.

## Decision

`D247_0_999_LEROBOT_AV1_PYAV_VALIDATION_METADATA_PASS`

The 0-999 professor top-view corpus now exists as LeRobot v3 AV1+parquet with
companion metadata and arbitrary PNG extraction proof.

Critical caveat:

- Local default torchcodec decode is currently broken in the `lerobot` env.
- Use `video_backend=pyav` locally unless torchcodec/FFmpeg is repaired.
- Earlier RunPod/H100 AV1 decode evidence remains a separate target-environment
  gate, not proof that this local torchcodec install is healthy.

Still blocked until explicit approval:

- Any raw PNG cleanup, archive, move, or deletion.
- 1000/10000 expansion beyond this 0-999 corpus.
- PPO/L2/Large PPO.
- VLA/SmolVLA fine-tuning.
- Action-teacher work.
- RoArm deployment.
- RunPod runtime.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `sim_scripts/cube10cm_top_view_smoke_to_lerobot.py`
- `sim_scripts/cube10cm_top_view_metadata_companion.py`
- `extract_frames.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/lerobot_validation_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/lerobot_video_frame_counts_pyav_d247.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/metadata_companion_d247/metadata_validation_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/debug_extract_frames_d247/extract_frame_pixel_diff_d247.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242_lerobot_d247_stdout.log`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242_lerobot_validate_pyav_d247_stdout.log`
