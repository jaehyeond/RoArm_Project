# D232 Storage Plan - Cube10cm Top-View Visual Dataset

Status: planning document plus D233 local smoke storage result and the
2026-07-14 D247/D249 USB archive preflight. No deletion, move, filesystem
repair, or USB write is authorized by this file.

## Current Disk Truth

Use the D232 disk audit as the current planning baseline:

- Local filesystem was about `590G` total, `522G` used, `39G` free, `94%`
  used.
- `RoArm_Project` was about `269G`, so the pressure is inside the project.
- Main pressure points were `outputs` at about `96G`, `claudedocs` at about
  `35G`, and `claudedocs/figures/p6v12_rollout/frames` at about `34G`.
- `collected_data*`, `b200_backup_*`, and `openvla_oft_b200_pulls` are needed
  data/backups. They are archive/move-only with explicit approval, not blind
  delete candidates.

## Cleanup Order

No cleanup is automatic.

1. First choice, if approval is granted:
   archive or remove `claudedocs/figures/p6v12_rollout/frames` after confirming
   compact outputs and any referenced summaries/videos are preserved.
2. Preserve SmolVLA `outputs/` by default.
3. If disk pressure requires touching `outputs/`, first create a manifest, then
   only with explicit approval remove/archive:
   `outputs/*/checkpoints/*/training_state`.
   Estimated reclaim: about `25.6GB` decimal. `pretrained_model` inference
   artifacts are preserved; training resume state is lost.
4. If more space is still needed and approval is granted, prune to one
   representative checkpoint per run:
   - `smolvla_official=050000`
   - `smolvla_v2_cleaned=050000`
   - `smolvla_v3_sponge=050000`
   - `smolvla_v5_multipos=200000`
   - `smolvla_v6=last`
   - `smolvla_v6_b200=last`
   - `smolvla_v6_stacking_b200=last`
   - `smolvla_v6_stacking_v2_b200=010000`
   - `smolvla_v6_stacking_v3_b200=020000`

Estimated reclaim for the keep-one path is about `90.15GB` decimal total, with
old four large runs contributing about `74.1GB`.

## Dataset Storage Policy

Primary dataset storage should be LeRobot video+parquet:

- RGB video key: `observation.images.top`.
- Raw resolution: `1280x720`.
- Per-frame metadata/state/action/object/camera fields in parquet.
- PNG frames are smoke/debug/extraction artifacts, not the scale-up storage
  format.

The professor-facing guarantee is not "we never produce PNG." It is:

- frame-by-frame image-state pairs remain intact;
- storage is MP4+parquet for scale;
- arbitrary frames can be extracted to PNG immediately with `extract_frames.py`.

## Codec Decision Gate

Codec must be selected before 100-episode chunk generation.

Candidate policy:

- Prefer AV1 only if the actual training LeRobot dataloader can load and decode
  frames reliably in the intended environment.
- Fall back to H.264 if AV1 has dataloader, tooling, or speed failures.
- Keep `extract_frames.py` ffmpeg extraction separate from the training-loader
  gate. ffmpeg extraction proves meeting/debug access, not training readiness.

Read-only preflight before smoke:

1. Use an existing LeRobot dataset such as `lerobot_dataset_v6` as a fixture.
2. Load frames through the same LeRobot API/backend expected in training.
3. Report codec, backend, decoded shape, sample count, and average decode time.
4. Do this locally first; repeat on RunPod/H100 only when that environment is
   available and explicitly approved.

Local D232 preflight result:

- Command required writable HuggingFace cache variables:
  `HF_HOME=/tmp/roarm_hf_cache` and
  `HF_DATASETS_CACHE=/tmp/roarm_hf_datasets_cache`.
- `lerobot_dataset_v6` reports codec `av1`, pix_fmt `yuv420p`, fps `30`, raw
  shape `[720, 1280, 3]`.
- `LeRobotDataset` decoded 5 sampled frames successfully as `torch.float32`
  images with shape `[3, 720, 1280]`, state/action shape `[6]`.
- Average sampled decode time was about `0.029s`; max was about `0.108s`.
- Source:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_lerobot_codec_preflight_v6_d232.json`.

Interpretation: AV1 is not rejected locally. It remains conditional on smoke
dataset decode speed and later RunPod/H100 environment verification. If either
environment fails, switch the new visual dataset codec to H.264.

## Scale-Up Storage Gate

Do not start 100/1000/10000 episode generation while free space is near the D232
audited `39G`.

Before 100-episode chunk:

- smoke MB/episode measured;
- projected `100`, `1000`, and `10000` episode size reported;
- target output root selected;
- cleanup/archive action explicitly approved if local disk is used;
- codec selected by the dataloader gate.

Before 1000-episode expansion:

- 100-episode chunk loads in LeRobot;
- sampled frame extraction works;
- storage projection still fits with margin;
- professor accepts camera view and PNG extraction workflow.

## Blocked Actions

Still blocked without explicit approval:

- deleting, moving, or archiving files;
- additional full or smoke rendering;
- 100/1000/10000 episode dataset generation;
- PPO/L2/Large PPO;
- VLA/action-teacher;
- RoArm deployment;
- SSH/B200 reconnect, pull, or `.ssh` copy.

## Local Smoke Storage Result - 2026-06-12 D233

Approved local smoke generated:

- 5 episodes;
- 975 total frames;
- raw frame size `1280x720`;
- debug PNG root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/raw_env_render_frames`;
- LeRobot root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_dataset_av1`.

Measured storage:

- debug PNG bytes total: `261672389`;
- debug PNG MB/episode: `52.3344778`;
- projected debug PNG size:
  - 100 episodes: `5.2334477800000005GB`;
  - 1000 episodes: `52.3344778GB`;
  - 10000 episodes: `523.344778GB`;
- LeRobot AV1 video bytes total: `2982439`;
- LeRobot AV1 video MB/episode: `0.5964878`;
- projected LeRobot AV1 video size:
  - 100 episodes: `0.059648780000000005GB`;
  - 1000 episodes: `0.5964878GB`;
  - 10000 episodes: `5.964878GB`.

Codec/load result:

- LeRobot validation status: `PASS`;
- codec: `av1`;
- pixel format: `yuv420p`;
- fps: `30`;
- frame count match: `true`;
- sampled LeRobot decode average/max: `0.016793251037597656s` /
  `0.06672263145446777s`;
- sampled source PNG vs decoded MP4 mean-abs max: `0.8939572482638889`;
- arbitrary PNG extraction succeeded from MP4+parquet.

Disk state after smoke was about `590G` total, `529G` used, `32G` free, `95%`
used.

Storage decision update:

- PNG-at-scale is rejected by measurement. PNG is debug/extraction only.
- LeRobot MP4+parquet is the only viable scale-up storage path under current
  disk pressure.
- AV1 is locally acceptable through LeRobot, but RunPod/H100 must repeat the
  dataloader decode/speed gate before using AV1 there.
- Do not start a local 100 episode chunk with about `32G` free unless a target
  output root and cleanup/archive plan are explicitly approved.

## Storage / Output-Root Decision - 2026-06-13 D234

User approved closing the storage/output-root and RunPod/H100 codec gates.

Decision:

- Primary retained dataset format remains LeRobot MP4+parquet.
- Codec for the next chunk: AV1.
- PNG remains debug/extraction only.
- Next 100 episode chunk, if explicitly launched later, should use:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d235`.
- This local root is acceptable only for one 100 episode chunk because measured
  retained debug PNG projects to about `5.2334477800000005GB` for 100 episodes,
  and AV1 video projects to about `0.059648780000000005GB`.
- Pre-run free-space gate: require at least about `25GB` free. Stop before
  launch if local free space drops below that.
- No deletion/archive/move is required for that one 100 episode chunk under the
  current estimate.
- 1000/10000 episodes remain blocked on local disk and require external/RunPod
  storage, a no-full-PNG-retention pipeline, or explicit cleanup/archive
  approval.

RunPod/H100 result:

- Full 975-frame AV1 decode PASSed through LeRobot dataloader.
- Avg/max decode: `0.017871856689453125s` /
  `0.10865616798400879s`.
- Result JSON:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_full_d234.json`.

## D247/D249 0-999 USB Archive Preflight - 2026-07-14

The user authorized a dependency and archive-plan audit for the old 0-999
script corpus. No copy, move, deletion, remount, or filesystem repair was run.
This is an operational storage sidecar and does not change the active grasp
case or consume the recommended D345 case number.

Verified classification:

- The 0-999 corpus was generated and converted before the first PPO runtime.
- D319/D321 later use its labels and D256 derivative as a script-only control
  and reset source, but current D322-D344 grasp work does not read it.
- The compact D247 LeRobot dataset, labels, metadata, D256, and D257 remain
  lineage-critical and stay local.
- The `51386208295`-byte raw PNG source is archive-safe but not disposable. It
  is the only part eligible for later local space reclamation.
- Never move the D242 parent root. Copy the full non-raw/control payload once,
  then archive raw frames in five 200-episode verified shuttle batches.
- USB copy success is not local deletion approval. Require source-to-USB and
  USB-to-final-destination hash PASS for core plus all five batches, followed by
  a separate user approval before touching local raw PNGs.

Canonical plan and machine inventory:

- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/ARCHIVE_PLAN.md`
- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/archive_inventory_20260714.json`
- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/raw_batches_20260714.tsv`

The USB observed during preflight is exFAT UUID `B0C1-F936`, currently mounted
read-only and containing unrelated `RELOC.zip`. A later transfer must begin by
asking the user to connect/reconnect it and approve one exact trip, then
rechecking identity, read-write state, and free space.
