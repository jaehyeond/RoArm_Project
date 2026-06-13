# Session 2026-06-13 - Cube10cm Top-View Chunk100 Sampling Contract D236

## Scope

Continuation after D235 schema/companion metadata validation. This step checks
whether it is safe to proceed directly to a 100 episode renderer.

This session did not run IsaacLab render, generate a 100 episode chunk, generate
1000/10000 episodes, delete/archive/move local files, train PPO/L2/Large PPO,
start SmolVLA/VLA fine-tuning, start action-teacher work, deploy to RoArm, use
SSH JHPark/B200, pull from B200, copy `.ssh`, or mix with Track A.

## Evidence Checked

- D232 requires explicit dataset splits:
  `train_success`, `eval_boundary`, `eval_failure`, and `debug_smoke`.
- D232 says sampling ranges and seeds must be recorded in metadata.
- D233 smoke used five camera-contract poses:
  `(0.24,0.00)`, `(0.14,-0.10)`, `(0.14,+0.10)`, `(0.34,-0.10)`,
  `(0.34,+0.10)`.
- D230 useful-tap evidence says fixed xy10 corners clean-passed, but randomized
  xy10 showed stable overshoot failures; therefore a pose bucket is not a final
  label until post-render metrics validate it.

## Decision

Do not implement the 0-99 renderer by simply lifting the smoke script cap and
repeating the five D233 poses.

Before a chunk renderer/run, create or confirm an explicit sampling manifest.
The renderer should consume that manifest and refuse to run without it.

## File Added

- `claudedocs/cube10cm_top_view_chunk100_sampling_contract_d236.md`

## Draft 0-99 Split

Draft only, not yet rendered:

- `000-004`: 5 `debug_smoke`
- `005-069`: 65 `train_success_candidate`
- `070-084`: 15 `eval_failure_candidate`
- `085-099`: 15 `eval_boundary_candidate`

These are intended sampling buckets, not final success/failure labels.

Required manifest fields:

- `episode_index`
- `split_candidate`
- `cube_x_m`
- `cube_y_m`
- `seed`
- `sampling_rule`
- `sampling_cell_id`
- `source_decision`
- `requires_posthoc_label_validation`

Required post-render labels:

- `contact_seen_any`
- `reaction_seen_any`
- `overshoot_seen_any`
- `full_visibility_frames`
- `partial_visibility_frames`
- `full_occlusion_frames`
- `centroid_error_px_max`
- `label_status`

## Current Next Step

Next non-render implementation step:

1. Generate a deterministic 0-99 manifest from the D236 contract.
2. Validate the manifest locally without IsaacLab.
3. Then build a chunk renderer that requires the manifest.

Still blocked without explicit approval:

- IsaacLab render
- 0-99 chunk generation
- 0-999 / 1000 / 10000 expansion
- deletion/archive/move
- PPO/L2/Large PPO
- SmolVLA/VLA fine-tuning
- action-teacher work
- RoArm deployment
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy
