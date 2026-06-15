# Session 2026-06-14 - Cube10cm Label-Aware 0-999 Manifest D241

Status: label-aware 0-999 manifest generator implemented and validate-only
manifest generation passed.

This session did not run IsaacLab, did not render images, did not build a
LeRobot dataset, did not train, did not delete, move, or archive files, and did
not use SSH/B200 or RunPod.

## Scope

- Branch scope: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- Approved step interpreted as the next non-render step from D240:
  implement and validate a label-aware 0-999 manifest generator.
- Out of scope: 0-999 actual render, 1000/10000 scale-up runtime,
  PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment,
  B200/SSH, pull, cleanup, and Track A.

## Verified Inputs

- `CLAUDE.md` Current-State Protocol requires START_HERE, DECISIONS, LEDGER,
  referenced session docs, git status, and metric verification before claims.
- `START_HERE.md` D240 says the next non-render step is a label-aware 0-999
  manifest generator/validation, while runtime render remains blocked pending
  explicit approval and disk/output-root preflight.
- D240 design says intended buckets are not final labels and every row must
  still require post-render numeric label validation.
- Existing renderer contract requires the core manifest fields:
  `episode_index`, `split_candidate`, `cube_x_m`, `cube_y_m`, `seed`,
  `sampling_rule`, `sampling_cell_id`, `source_decision`, and
  `requires_posthoc_label_validation`.

## Implementation

Added:

`sim_scripts/cube10cm_top_view_labelaware_manifest_0_999.py`

The script writes a deterministic 1000 episode plan only. It explicitly does not
run IsaacLab, render images, build LeRobot, train, delete, archive, move, or
touch B200.

Manifest fields:

- renderer-compatible fields:
  `episode_index`, `split_candidate`, `cube_x_m`, `cube_y_m`, `seed`,
  `sampling_rule`, `sampling_cell_id`, `source_decision`,
  `requires_posthoc_label_validation`
- label-aware fields:
  `intended_sampling_bucket`, `intended_role`, `camera_coverage_required`,
  `expected_postrender_labels`, `label_policy`

Forbidden pre-render final label fields:

- `label_useful_clean_numeric`
- `label_overshoot_numeric`
- `label_camera_contract_numeric`
- `label_status`

## Manifest Output

Output root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_labelaware_manifest_0_999_d241`

Files:

- `episode_manifest.csv`
- `episode_manifest.json`
- `manifest_summary.json`

Output size:

- root total: about `1.1M`
- CSV: `331947` bytes
- JSON: `704711` bytes
- summary: `2433` bytes

## Validation Result

`manifest_summary.json`:

- status: `PASS`
- rows: `1000`
- episode index range: `0..999`
- seed base: `2410`
- seed unique: `true`
- all rows require post-render label validation: `true`
- final label fields present: `false`
- x range: `0.09..0.39`
- y range: `-0.1..0.15`

Bucket counts:

- `debug_camera_anchor`: `50`
- `clean_prior_candidate`: `650`
- `transition_mixed_probe`: `200`
- `overshoot_eval_candidate`: `100`

Independent CSV check also passed:

- rows: `1000`
- ids contiguous: `true`
- seeds unique: `true`
- requires-posthoc all true: `true`
- camera-coverage-required all true: `true`
- forbidden final label fields: `[]`

## Critical Blocker

This manifest is not a rendered dataset. It is only a render plan.

Also, the current `sim_scripts/cube10cm_top_view_visual_chunk_render.py` is
intentionally scoped to exactly 100 episodes. Therefore this 0-999 manifest
cannot be used for an actual 1000 episode render until a separately approved
renderer update or new renderer is implemented and validated.

## Still Blocked

- Actual 0-999 / 1000 / 10000 Isaac render.
- Any dataset scale-up beyond the existing d241 0-99 render.
- Any deletion, move, archive, or cleanup.
- PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.

## Sources

- `CLAUDE.md`
- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `claudedocs/session_20260614_cube10cm_top_view_d241_lerobot_metadata_labelaware_d240.md`
- `claudedocs/cube10cm_top_view_label_aware_0_999_manifest_design_d240.md`
- `sim_scripts/cube10cm_top_view_labelaware_manifest_0_999.py`
- `sim_scripts/cube10cm_top_view_visual_chunk_render.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_labelaware_manifest_0_999_d241/manifest_summary.json`
