# Session 2026-06-15 - Cube10cm Top-View 0-999 Runtime Preflight Blocked D243

## Scope

- Active branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- User explicitly approved the next runtime direction, but requested expected time/capacity verification.
- No IsaacLab render was started.
- No dataset generation, LeRobot conversion, cleanup, deletion, archive, move, PPO, VLA, action-teacher, RoArm deployment, SSH/B200, pull, or `.ssh` copy was run.

## Current-State Checks

- `git status --short --untracked-files=all --branch` showed existing dirty/untracked state and branch `master...origin/master`.
- D242 current truth already required disk/output-root preflight and `--render-approved` before any actual 0-999 render.
- D241 render evidence remains the only measured source for 0-999 projection:
  - 100 episodes / 19,500 frames.
  - raw PNG total `5142551626` bytes.
  - `51.42551626MB/episode`.
  - elapsed `4647.953013896942s`.
  - effective captured FPS `4.195395250704307`.
- D240 LeRobot evidence remains:
  - codec `av1`, `30fps`.
  - 100 episodes / 19,500 frames.
  - video bytes `56604396`.
  - `0.56604396MB/episode`.
  - decode/load validation PASS.

## Disk And Runtime Preflight

Local filesystem:

- Repo filesystem: `/dev/nvme0n1p5`.
- `df -B1 .`: total `632825225216`, used `572916858880`, available `27687411712`, use `96%`.
- `df -B1 /tmp`: same filesystem, available `27687079936`.
- Therefore `/tmp` is not an alternate output root.

Measured local storage pressure:

- `outputs`: `96G`.
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480`: `6.3G`.
- D241 100ep render root: `5.0G`.
- D241 0-999 manifest root: `1.1M`.

GPU visibility:

- `nvidia-smi` saw `NVIDIA GeForce RTX 4090 Laptop GPU`.
- Memory total/free: `16376MiB` / `13432MiB`.
- GPU is not the current blocker; disk is.

## 1000 Episode Projection From D241

Projection uses the measured D241 100ep render multiplied by 10:

- Episodes: `1000`.
- Frames: `195000`.
- Expected render time: `46479.530s` = `12.911h`.
- Practical wall-time estimate: about `13-15h` after startup and I/O overhead.
- Raw PNG projection: `51425516260` bytes = `51.426GB` decimal = `47.894GiB`.
- AV1 video projection after conversion: `566043960` bytes = `0.566GB`.
- `frames.jsonl` projection: `457242210` bytes.
- companion per-frame parquet projection: `37500380` bytes.
- companion episode parquet projection: `158550` bytes.
- Minimal raw PNG + AV1 video + JSONL + companion metadata projection: `52.486GB`.

## Verdict

`LOCAL_0_999_RUNTIME_NOT_STARTED_DISK_HARD_BLOCK_D243`

The approved runtime was not launched because available local disk is only about `27.69GB`, while the current renderer writes all raw PNG frames first and needs at least about `52.49GB` for raw PNG plus expected post-render artifacts. This does not include a safety margin. Starting the 0-999 render now would likely fail with disk-full before completion, around the mid-run range, and would leave a partial output root that still needs cleanup.

The renderer behavior was checked in `sim_scripts/cube10cm_top_view_visual_manifest_render.py`: actual render requires `--render-approved`, creates `raw_env_render_frames`, writes one PNG per captured frame, then writes `frames.jsonl` and `render_summary.json`. The actual root `cube10cm_top_view_visual_0_999_d242` remains absent.

## Next Safe Choices

1. Use an output root on storage with at least `60GB` free for this exact current renderer, preferably more to keep a safety margin.
2. Or explicitly approve a cleanup/archive plan first. The first D232 cleanup path, `outputs/*/checkpoints/*/training_state`, is estimated around `25.6GB` and is not enough by itself for a safe 1000ep local run with margin.
3. Or redesign the renderer/conversion pipeline to avoid retaining all raw PNGs at once, for example chunked render plus LeRobot conversion plus explicit raw-PNG disposal/archive policy. That is a code/pipeline change and should be approved separately because it changes artifact retention.
4. Or run on RunPod/external storage, with the cost rule preserved: copy results back, stop the pod immediately after work, and terminate/delete the pod if no remote environment or volume must be preserved.

## Still Blocked

- Actual 0-999 local render on the current output root.
- 1000/10000 scale-up claims.
- Any deletion/archive/move/cleanup without explicit approval and manifest.
- PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.
