# Session 2026-06-13 - Cube10cm Top-View Chunk100 Manifest D237

## Scope

Continuation after D236 sampling contract. This step generates and validates the
deterministic 0-99 episode manifest only.

This session did not run IsaacLab render, generate a 100 episode chunk, generate
1000/10000 episodes, delete/archive/move local files, train PPO/L2/Large PPO,
start SmolVLA/VLA fine-tuning, start action-teacher work, deploy to RoArm, use
SSH JHPark/B200, pull from B200, copy `.ssh`, or mix with Track A.

## File Added

- `sim_scripts/cube10cm_top_view_chunk100_manifest.py`

The script is local/non-render only and writes:

- `episode_manifest.csv`
- `episode_manifest.json`
- `manifest_summary.json`

under:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_chunk100_manifest_d236`

## Command

```bash
python3 -u sim_scripts/cube10cm_top_view_chunk100_manifest.py
```

## Result

Status: `PASS`

- rows: `100`
- episode ids: `0..99`
- split counts:
  - `debug_smoke=5`
  - `train_success_candidate=65`
  - `eval_failure_candidate=15`
  - `eval_boundary_candidate=15`
- every row has:
  - `episode_index`
  - `split_candidate`
  - `cube_x_m`
  - `cube_y_m`
  - `seed`
  - `sampling_rule`
  - `sampling_cell_id`
  - `source_decision`
  - `requires_posthoc_label_validation=True`

## Critical Notes

- This manifest is not a success/failure label file.
- It is an intended sampling plan. Final labels must come from post-render
  metadata.
- Boundary rows use `y=0.15`, which is wider than the D233 smoke `y=0.10`
  coverage. They are explicitly marked with
  `close_x_high_y_boundary_candidate_camera_coverage_required`.
- The D233 smoke script remains capped at 1-10 episodes. Do not weaken that
  guard for chunk generation.

## Current Next Step

Next non-render implementation step:

1. Build a separate manifest-fed chunk renderer.
2. Renderer must refuse to run without a manifest.
3. Renderer must write the manifest copy into the render root.
4. Renderer must preserve the D233 camera contract and D235 companion metadata
   join keys.

Runtime remains blocked until explicit launch approval:

- IsaacLab render
- 0-99 chunk generation
- LeRobot AV1 conversion of the new chunk
- 0-999 / 1000 / 10000 expansion
- deletion/archive/move
- PPO/L2/Large PPO
- SmolVLA/VLA fine-tuning
- action-teacher work
- RoArm deployment
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy
