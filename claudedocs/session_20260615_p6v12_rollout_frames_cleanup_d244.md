# Session 2026-06-15 - P6v12 Rollout Raw Frames Cleanup D244

## Scope

- User explicitly approved deleting only
  `claudedocs/figures/p6v12_rollout/frames`.
- No IsaacLab render, dataset generation, LeRobot conversion, training,
  RunPod, SSH/B200, pull, `.ssh` copy, PPO, VLA, action-teacher, or RoArm work
  was run.
- No `outputs/`, `collected_data*`, `b200_backup_*`, or
  `openvla_oft_b200_pulls` files were touched.

## What This Folder Was

`claudedocs/figures/p6v12_rollout/frames` was a raw PNG dump from the older
P6v12 lab-meeting visualization path, not part of the active professor 10cm /
0.72kg cube top-view visual trajectory dataset branch.

Repo evidence:

- `claudedocs/labmeeting_p6v12_rollout_20260513.md` records P6v12 as a lab
  meeting qualitative video artifact for the close-hover farming failure mode.
- `scripts/render_p6v12_policy_rollout.py` writes BasicWriter PNGs under
  `<out_dir>/frames` and then encodes `p6v12_rollout.mp4`.
- The preserved compact/semantic artifacts are:
  - `claudedocs/figures/p6v12_rollout/p6v12_rollout.mp4`;
  - `claudedocs/figures/p6v12_rollout/p6v12_trajectory.csv`;
  - `claudedocs/figures/p6v12_rollout/replay`;
  - `claudedocs/figures/p6v12_rollout/replay_old_camera`;
  - `claudedocs/figures/p6v12_rollout/replay_silver_backup`.

## Pre-delete Manifest

Manifest directory:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/p6v12_rollout_frames_cleanup_d243`

Pre-delete exact file-list manifest:

- `frames_manifest_predelete.tsv`
- line count: `73366`
- sha256:
  `45a0ece58f4e86cd605b262e4e43bce17b11c1326cb3285dfcccded6ea922e26`
- manifest size: `3.7M`

Pre-delete measurements:

- `df -h .`: `590G` total, `534G` used, `26G` available, `96%` used.
- target path file count: `73366`.
- target path type: directory.
- target path size by `du -sh`: `34G`.

## Deletion

Deleted exactly:

- `claudedocs/figures/p6v12_rollout/frames`

Post-delete verification:

- `test -e claudedocs/figures/p6v12_rollout/frames` returned exit `1`.
- `claudedocs/figures/p6v12_rollout/replay` still contains `200` PNG files.
- `claudedocs/figures/p6v12_rollout/p6v12_rollout.mp4` remains present.
- `claudedocs/figures/p6v12_rollout/p6v12_trajectory.csv` remains present.

Preserved sizes after cleanup:

- `claudedocs/figures/p6v12_rollout`: `67M`.
- `p6v12_rollout.mp4`: `189K`.
- `p6v12_trajectory.csv`: `25K`.
- `_concat.txt`: `27K`.
- `replay`: `21M`.
- `replay_old_camera`: `26M`.
- `replay_silver_backup`: `21M`.

## Disk Result

Post-delete `df -h .`:

- `590G` total, `501G` used, `60G` available, `90%` used.

Post-delete `df -B1 .`:

- total `632825225216`;
- used `537226887168`;
- available `63377383424`;
- use `90%`.

Compared with the D243 preflight baseline:

- before available: `27687411712` bytes;
- after available: `63377383424` bytes;
- net available-space increase: `35689971712` bytes, about `35.69GB`
  decimal.

## Verdict

`P6V12_RAW_FRAMES_CLEANUP_COMPLETE_D244`

The highest-value D232 cleanup candidate was removed after manifest and explicit
approval. The compact P6v12 lab-meeting evidence remains preserved. Local free
space improved from about `26G` to about `60G`.

This improves the local storage situation, but it does not automatically start
or approve the 0-999 render. D243 projected the current raw-PNG-first renderer at
about `52.49GB` minimum for 1000 episodes before safety margin, so the current
`60G` free space is close to the lower bound. A safer local run still benefits
from additional cleanup, a larger output root, or a chunked/no-full-PNG-retention
pipeline.
