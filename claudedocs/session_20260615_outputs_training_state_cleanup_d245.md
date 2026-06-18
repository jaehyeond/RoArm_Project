# Session 2026-06-15 - Outputs Training State Cleanup D245

## Scope

- User asked to critically reassess the second D232 cleanup path and free space.
- Deleted only `outputs/*/checkpoints/*/training_state` after manifest.
- No `outputs/*/checkpoints/*/pretrained_model` directories were deleted.
- No `collected_data*`, `b200_backup_*`, `openvla_oft_b200_pulls`, IsaacLab
  render, dataset generation, LeRobot conversion, PPO, VLA fine-tuning,
  action-teacher, RoArm deployment, RunPod, SSH/B200, pull, or `.ssh` copy was
  run.

## Critical Recheck

The cleanup is not raw research-data deletion. It removes SmolVLA training
resume state:

- optimizer state;
- scheduler state;
- RNG state;
- step counter.

The preserved model artifact is `pretrained_model`, including
`model.safetensors`, config, processor files, and train config. This means
inference, evaluation, deployment tests, and new fine-tuning from the saved
weights remain possible. Exact optimizer-state resume from those checkpoints is
lost.

Repo/document evidence:

- D232 says preserve `outputs/` by default and, under disk pressure, remove only
  `outputs/*/checkpoints/*/training_state` after manifest and explicit approval.
- D232 records that this preserves `pretrained_model` inference artifacts while
  losing training resume state.
- D243 records this cleanup alone was not enough before the D244 p6v12 frame
  cleanup; after D244 it became a reasonable margin-improving step.

## Pre-delete Measurements

Before this D245 cleanup:

- `df -h .`: `590G` total, `501G` used, `60G` available, `90%` used.
- `df -B1 .`: total `632825225216`, used `537239658496`, available
  `63364612096`, use `90%`.
- `du -sch outputs/*/checkpoints/*/training_state`: `23G` total.
- Training-state directories found: `58`.
- Training-state files found: `290`.
- Pretrained-model directories found and intended to preserve: `66`.

Run-level training-state sizes:

- `outputs/smolvla_official`: `10` dirs, `3.9G`.
- `outputs/smolvla_v2_cleaned`: `10` dirs, `3.9G`.
- `outputs/smolvla_v3_sponge`: `10` dirs, `3.9G`.
- `outputs/smolvla_v5_multipos`: `20` dirs, `7.7G`.
- `outputs/smolvla_v6`: `5` dirs, `2.0G`.
- `outputs/smolvla_v6_b200`: `3` dirs, `1.2G`.

## Manifest

Manifest directory:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/outputs_training_state_cleanup_d245`

Files:

- `training_state_dirs_predelete.txt`
  - lines: `58`
  - sha256:
    `89f4801443f551bd397b7f826772f5de7b9f460ba6ed1568a054b5eb7025d704`
- `training_state_files_predelete.tsv`
  - lines: `290`
  - sha256:
    `b8f83a38571c057eeca389bfc4d0b3891a6a6f939a044e86c83e17fcacc99525`
- `training_state_du_predelete.txt`
  - sha256:
    `0a504a43a31f2a40452fa31b126211d5c3a408087f3fa6e2e6e72b1d54277bda`
- `pretrained_model_dirs_preserved_predelete.txt`
  - lines: `66`
  - sha256:
    `e578be7c558b9345725168fceea6e2d7a1b1a2bcb16739c2b256bb083df28292`

Guard check:

- `training_state_dirs_predelete.txt` contained only paths ending in
  `/training_state`.

## Deletion

Deleted exactly the `58` directories listed in:

- `training_state_dirs_predelete.txt`

Post-delete verification:

- Remaining `training_state` directories under `outputs`: `0`.
- Remaining `pretrained_model` directories under `outputs`: `66`.
- Representative preserved model paths exist:
  - `outputs/smolvla_official/checkpoints/050000/pretrained_model`;
  - `outputs/smolvla_v5_multipos/checkpoints/200000/pretrained_model`;
  - `outputs/smolvla_v6_b200/checkpoints/015000/pretrained_model`.

Post-delete output sizes:

- `outputs/smolvla_official`: `12G`.
- `outputs/smolvla_v2_cleaned`: `12G`.
- `outputs/smolvla_v3_sponge`: `12G`.
- `outputs/smolvla_v5_multipos`: `23G`.
- `outputs/smolvla_v6`: `5.6G`.
- `outputs/smolvla_v6_b200`: `4.5G`.
- `outputs/smolvla_v6_stacking_b200`: `1.2G`.
- `outputs/smolvla_v6_stacking_v2_b200`: `2.3G`.
- `outputs/smolvla_v6_stacking_v3_b200`: `4.5G`.

## Disk Result

Post-delete `df -h .`:

- `590G` total, `479G` used, `82G` available, `86%` used.

Post-delete `df -B1 .`:

- total `632825225216`;
- used `513302863872`;
- available `87301406720`;
- use `86%`.

Compared with D245 pre-delete:

- before available: `63364612096` bytes;
- after available: `87301406720` bytes;
- net available-space increase: `23936794624` bytes, about `23.94GB`
  decimal.

Compared with the original D243 hard-block baseline:

- D243 available: `27687411712` bytes;
- D245 after cleanup available: `87301406720` bytes;
- net available-space increase since D243: `59613995008` bytes, about
  `59.61GB` decimal.

## Verdict

`OUTPUTS_TRAINING_STATE_CLEANUP_COMPLETE_D245`

The D232 second cleanup path was executed after manifest and user approval.
SmolVLA pretrained model artifacts remain available, but exact resume state for
the affected historical SmolVLA checkpoints is intentionally gone.

Local storage is now materially better for the active professor 10cm top-view
dataset branch. However, this cleanup does not itself launch the 0-999 render.
D243 projected the current raw-PNG-first renderer at about `52.49GB` minimum for
1000 episodes before safety margin; current free space is about `82G`, which is
usable with more margin than D244 but still below the older D232 conservative
`100G` target.
