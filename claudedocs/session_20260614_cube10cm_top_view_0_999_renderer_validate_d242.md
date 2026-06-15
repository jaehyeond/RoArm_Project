# Session 2026-06-14 - Cube10cm 0-999 Renderer Validate-Only D242

Status: new 0-999-capable manifest renderer added and validate-only guard passed.

This session did not run IsaacLab, did not render images, did not build a
LeRobot dataset, did not train, did not delete, move, or archive files, and did
not use SSH/B200 or RunPod.

## Scope

- Branch scope: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- Approved step interpreted as the next non-render step from D241:
  create a renderer update/new renderer that can accept the 0-999 label-aware
  manifest and validate it without launching IsaacLab.
- Out of scope: actual 0-999 render, 1000/10000 scale-up runtime,
  PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment,
  B200/SSH, pull, cleanup, and Track A.

## Verified Inputs

- D241 current truth says the 0-999 manifest passed validation but is only a
  render plan, not a dataset.
- D241 also says the old `cube10cm_top_view_visual_chunk_render.py` is scoped to
  exactly 100 episodes, so actual 0-999 render requires a separately approved
  renderer update or new renderer.
- D241 manifest summary records rows `1000`, episode ids `0..999`, all rows
  require post-render numeric label validation, and intended bucket counts
  `debug_camera_anchor=50`, `clean_prior_candidate=650`,
  `transition_mixed_probe=200`, `overshoot_eval_candidate=100`.

## Implementation

Added:

`sim_scripts/cube10cm_top_view_visual_manifest_render.py`

Design:

- Keeps the existing 100ep renderer unchanged for d239/d241 reproducibility.
- Accepts a manifest with arbitrary expected episode count; default is the D241
  0-999 manifest and `--expected-episodes=1000`.
- `--validate-only` reads and validates the manifest, writes a validation JSON,
  and exits before importing or launching IsaacLab.
- Actual render requires omitting `--validate-only` and passing
  `--render-approved`; otherwise the script raises before rendering.
- The renderer preserves core manifest fields and carries label-aware intent
  fields into frame metadata if actual render is later approved.

## Validate-Only Result

Command class:

`python3 sim_scripts/cube10cm_top_view_visual_manifest_render.py --validate-only`

Output summary:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_manifest_render_validate_d242.json`

Validation result:

- status: `PASS`
- runtime: `VALIDATE_ONLY_NO_RENDER_NO_DATASET_GENERATION_NO_TRAINING`
- rows: `1000`
- expected episodes: `1000`
- episode index range: `0..999`
- split counts:
  - `debug_camera_anchor=50`
  - `clean_prior_candidate=650`
  - `transition_mixed_probe=200`
  - `overshoot_eval_candidate=100`
- intended bucket counts match the split counts
- all rows require post-render label validation: `true`
- all rows require camera coverage: `true`
- seed unique: `true`
- robot USD exists: `true`
- output render root exists: `false`
- output render root empty: `true`
- render approved: `false`
- width/height/fps: `1280x720@30`
- steps/capture stride: `580` / `3`

Critical non-render evidence:

- The actual render output root
  `cube10cm_top_view_visual_0_999_d242` does not exist.
- The only new runtime output from this step is the 4KB validate-only JSON.

## Still Blocked

- Actual 0-999 / 1000 / 10000 Isaac render.
- Any dataset scale-up beyond the existing d241 0-99 render.
- Any deletion, move, archive, or cleanup.
- PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/session_20260614_cube10cm_top_view_labelaware_manifest_0_999_d241.md`
- `sim_scripts/cube10cm_top_view_labelaware_manifest_0_999.py`
- `sim_scripts/cube10cm_top_view_visual_manifest_render.py`
- `sim_scripts/cube10cm_top_view_visual_chunk_render.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_labelaware_manifest_0_999_d241/manifest_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_manifest_render_validate_d242.json`
