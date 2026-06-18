# Session 2026-06-17 - Cube10cm label package + camera-fail audit D248

## Scope

- Active branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- User asked to package train/eval/quarantine episode lists and inspect the 14
  camera-fail episodes with visual and numeric evidence.
- This session read existing D246/D247 artifacts only.
- No render, training, deletion, move, archive, PPO, L2, Large PPO,
  VLA/SmolVLA fine-tuning, action-teacher, RoArm deployment, RunPod runtime,
  B200/SSH/pull, `.ssh` copy, or Track A work was run.

## Inputs

Render root:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242
```

Input label CSV:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246/episode_labels.csv
```

D246 source label state:

- total: `1000` episodes
- `clean_useful_tap=819`
- `contact_reaction_with_overshoot=167`
- `camera_quality_fail=14`
- camera contract pass: `986/1000`

## Script

Added:

```text
sim_scripts/cube10cm_top_view_package_label_splits.py
```

Run:

```bash
python3 sim_scripts/cube10cm_top_view_package_label_splits.py
```

Output package:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/label_package_d248
```

## Packaging Policy

- `train_clean_positive`: camera-pass `clean_useful_tap` only, excluding the
  deterministic clean holdout.
- `eval_clean_holdout`: deterministic `10%` holdout from camera-pass clean
  useful taps.
- `eval_overshoot_diagnostic`: camera-pass contact/reaction episodes with
  overshoot. These are useful for diagnostic evaluation but should not be used as
  positive behavior-cloning demonstrations by default.
- `quarantine_camera_fail`: camera contract failures. Exclude from train/eval
  until camera coverage/projection handling is reviewed.

## Package Result

Summary JSON:

```text
label_package_d248/split_package_summary.json
```

Counts:

- total: `1000`
- train: `737`
- eval: `249`
- quarantine: `14`
- by subsplit:
  - `train_clean_positive=737`
  - `eval_clean_holdout=82`
  - `eval_overshoot_diagnostic=167`
  - `quarantine_camera_fail=14`

Generated files:

- `episode_split_manifest.csv`
- `train_clean_positive.csv`
- `train_clean_positive_episode_ids.txt`
- `eval_clean_holdout.csv`
- `eval_clean_holdout_episode_ids.txt`
- `eval_overshoot_diagnostic.csv`
- `eval_overshoot_diagnostic_episode_ids.txt`
- `quarantine_camera_fail.csv`
- `quarantine_camera_fail_episode_ids.txt`
- `camera_fail_details.csv`
- `camera_fail_contact_sheet.png`
- `split_package_summary.json`

Package size: about `2.4M`.

## Camera-Fail Numeric Audit

Camera-fail detail file:

```text
label_package_d248/camera_fail_details.csv
```

Rows:

| episode | reason | split | cell | x | y | clean | overshoot | projection | max error px |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 451 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x01_y16 | 0.148333 | -0.020000 | 0 | 1 | 195/195 | 20.416585 |
| 475 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x00_y17 | 0.140000 | -0.015000 | 1 | 0 | 195/195 | 23.171721 |
| 476 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x01_y17 | 0.148333 | -0.015000 | 1 | 0 | 195/195 | 24.329415 |
| 477 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x02_y17 | 0.156667 | -0.015000 | 1 | 0 | 195/195 | 22.748333 |
| 478 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x03_y17 | 0.165000 | -0.015000 | 1 | 0 | 195/195 | 20.153790 |
| 500 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x00_y18 | 0.140000 | -0.010000 | 0 | 1 | 195/195 | 27.135012 |
| 501 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x01_y18 | 0.148333 | -0.010000 | 1 | 0 | 195/195 | 26.394715 |
| 502 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x02_y18 | 0.156667 | -0.010000 | 1 | 0 | 195/195 | 21.927794 |
| 525 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x00_y19 | 0.140000 | -0.005000 | 0 | 1 | 195/195 | 26.458589 |
| 526 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x01_y19 | 0.148333 | -0.005000 | 1 | 0 | 195/195 | 22.624042 |
| 575 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x00_y21 | 0.140000 | 0.005000 | 1 | 0 | 195/195 | 21.477693 |
| 576 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x01_y21 | 0.148333 | 0.005000 | 1 | 0 | 195/195 | 20.631811 |
| 601 | reprojection_error_gt_gate | clean_prior_candidate | clean_prior_x01_y22 | 0.148333 | 0.010000 | 1 | 0 | 195/195 | 20.926935 |
| 721 | projection_outside+reprojection_error_gt_gate | transition_mixed_probe | transition_x01_y01 | 0.150526 | 0.036111 | 0 | 1 | 7/195 | 33.666152 |

Interpretation:

- `13/14` camera failures are not missing-object frames. They have full
  visibility and projection-inside counts `195/195`, but fail the strict
  `20px` reprojection max gate.
- Episode `721` is qualitatively different: projection is inside only `7/195`
  frames and the contact sheet shows coverage/projection loss in later frames.
  Treat it as the strongest camera-contract warning.

## Visual Audit

Contact sheet:

```text
label_package_d248/camera_fail_contact_sheet.png
```

File check:

- PNG image data
- `1780 x 2688`
- RGB

Visual finding:

- Episodes `451, 475, 476, 477, 478, 500, 501, 502, 525, 526, 575, 576, 601`
  show the cube visible through the selected first/contact/reaction/max-error/last
  frames. These look like reprojection-gate/calibration-margin failures, not
  total visibility failures.
- Episode `721` visibly differs: later selected frames show projected bbox
  invalid/outside and should remain quarantined regardless of the event label.

## Decision

`D248_LABEL_PACKAGE_TRAIN_EVAL_QUARANTINE_CAMERA_FAIL_AUDIT_PASS`

Default next data usage:

- Use `train_clean_positive` for positive behavior-cloning train.
- Use `eval_clean_holdout` for held-out clean eval.
- Use `eval_overshoot_diagnostic` for failure/overshoot diagnostics only.
- Keep `quarantine_camera_fail` excluded until camera contract/projection handling
  is reviewed.

Still blocked until explicit approval:

- Any training or fine-tuning.
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
- `sim_scripts/cube10cm_top_view_package_label_splits.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246/episode_labels.csv`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/label_package_d248/split_package_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/label_package_d248/camera_fail_details.csv`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/label_package_d248/camera_fail_contact_sheet.png`
