# Session 2026-06-13 - Cube10cm Top-View Manifest-Fed Renderer D238

## Scope

Continuation after D237 manifest generation. This step writes the manifest-fed
chunk renderer and validates its non-render front gate.

This session did not run IsaacLab render, generate a 100 episode chunk, generate
1000/10000 episodes, delete/archive/move local files, train PPO/L2/Large PPO,
start SmolVLA/VLA fine-tuning, start action-teacher work, deploy to RoArm, use
SSH JHPark/B200, pull from B200, copy `.ssh`, or mix with Track A.

## File Added

- `sim_scripts/cube10cm_top_view_visual_chunk_render.py`

Key properties:

- separate from the D233 smoke renderer;
- requires an episode manifest;
- refuses to run if expected episode count is not exactly `100`;
- refuses to run if output root is non-empty;
- copies `episode_manifest.csv` into the render root;
- attaches manifest fields to every frame metadata row:
  - `split_candidate`
  - `manifest_seed`
  - `sampling_rule`
  - `sampling_cell_id`
  - `source_decision`
  - `requires_posthoc_label_validation`
- supports `--validate-only` to check manifest/args before IsaacLab starts.

## Static / Non-Render Checks

Commands:

```bash
python3 -m py_compile sim_scripts/cube10cm_top_view_visual_chunk_render.py sim_scripts/cube10cm_top_view_chunk100_manifest.py sim_scripts/cube10cm_top_view_metadata_companion.py
python3 sim_scripts/cube10cm_top_view_visual_chunk_render.py --help
python3 -u sim_scripts/cube10cm_top_view_visual_chunk_render.py --validate-only
```

Result:

- compile: PASS
- `--help`: PASS
- `--validate-only`: PASS
  - episodes: `100`
  - manifest:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_chunk100_manifest_d236/episode_manifest.csv`
  - split counts:
    - `debug_smoke=5`
    - `train_success_candidate=65`
    - `eval_failure_candidate=15`
    - `eval_boundary_candidate=15`

## Current Next Step

Next runtime step, only after explicit approval:

1. Run preflight:
   - `git status --short --untracked-files=all --branch`
   - `df -h .`
   - confirm output root is absent or empty
   - confirm no RunPod/B200/Track A action
2. Run exactly one local 0-99 render with the manifest-fed renderer.
3. Convert that chunk to LeRobot AV1.
4. Generate D235 companion metadata for the new chunk.
5. Validate:
   - LeRobot load/decode
   - PNG extraction
   - source PNG vs decoded MP4 pixel diff
   - row alignment
   - storage projection
   - visibility/reprojection
   - boundary camera coverage, especially `y=0.15`

Still blocked without explicit launch approval:

- IsaacLab render
- 0-99 chunk generation
- 0-999 / 1000 / 10000 expansion
- deletion/archive/move
- PPO/L2/Large PPO
- SmolVLA/VLA fine-tuning
- action-teacher work
- RoArm deployment
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy
