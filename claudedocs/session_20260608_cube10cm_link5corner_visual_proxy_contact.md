# 2026-06-08 cube10cm link5-corner visual proxy-contact inspection

## Scope

- Branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier branch.
- Task: inspect the existing link5-corner position runtime trace visually/logically before any dataset/RL/RoArm step.
- Added local-only script: `sim_scripts/cube10cm_link5corner_visual_proxy_contact_inspection.py`.
- Not run: IsaacLab runtime, GPU physics, dataset generation, training, RL/PPO, RoArm-M3-Pro control, B200/SSH/pull, Track A.

## Inputs

- Trace:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_trace.csv`
- Rollout CSV:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962.csv`
- Summary JSON:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_summary.json`
- Reaction gate:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_reaction_gate_audit.json`
- Trace diagnostic:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_trace_diagnostic_summary.json`

## Method

- The script constructs local contact-frame diagrams from trace columns only:
  `tool_proxy_x/y/z_after_m`, `tool_contact_target_x/y/z_m`, TCP, cube pose, cube size, push direction, contact/stop flags, displacement, speed, and clipping.
- It writes:
  - HTML/SVG:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_link5corner_visual_proxy_contact_inspection.html`
  - JSON:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_link5corner_visual_proxy_contact_inspection.json`
  - Summary:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_link5corner_visual_proxy_contact_inspection_summary.out`
- Validation note: the first draft incorrectly tried to read proxy columns as
  `tool_proxy_after_x_m`; trace uses `tool_proxy_x_after_m`. The bad output showed
  zero proxy positions and was rejected before documentation. The script was fixed,
  rerun, and the final numbers below are from the corrected output.

## Results

### Contract

- Summary line 1: local trace-only; no GPU runtime, dataset generation, training, robot control, or SSH.
- Summary line 2: contract OK, 1568 trace rows, 16 envs,
  `tool_contact_proxy_mode=link5_collision_corner_011`,
  `tool_proxy_label=link5_collision:corner_011`, `command_type=position`,
  and reaction gate PASS.

### Proxy Tracking

- Summary line 3:
  - proxy-target error mean `0.003834021m`
  - proxy-target z error mean `0.001500659m`
  - <=3mm rate `0.0`
  - <=5mm rate `1.0`
- Interpretation: proxy tracking is visually close enough for a side-contact
  sanity check, but still not a clean/Tier-A action teacher.

### Height Semantics

- Summary line 4:
  - proxy minus live cube center z mean `0.000392607m`
  - proxy below live cube top mean `0.049607393m`
  - side-center z near-5mm rate `1.0`
  - proxy-not-top rate `1.0`
- Interpretation: the link5-corner proxy is no longer top contact. It is at
  side-center height relative to the live cube.

### Face Placement And Weak Tap

- Summary line 5:
  - proxy gap to live side face mean `-0.006247562m`
  - target gap to live side face mean `-0.002771095m`
  - proxy outside live face rate `1.0`
  - target outside live face rate `1.0`
  - proxy lateral from cube center mean `0.019393899m`
  - target lateral from cube center mean `0.019994020m`
- Summary line 6:
  - max-displacement proxy gap mean `-0.004976191m`
  - max-displacement target gap mean `-0.003074710m`
  - max-displacement proxy outside live face rate `1.0`
  - contact-stop same as contact rate `1.0`
  - summary max displacement mean `0.001431603m`
  - summary speed mean `0.024424722m/s`
  - summary low-motion rate `1.0`
- Interpretation: the point is side-center height, but it is outside/grazing
  the approach face. The controller freezes as soon as contact is detected, so
  the result stays a weak 1mm tap rather than a clean stronger 2-3mm transient.

## Verdict

- Primary gate remains PASS.
- Side-center proxy visual semantics are verified.
- Top-contact explanation is rejected for this link5 proxy.
- Grazing/outside-face placement and early freeze are supported.
- Clean tap strength is NOT visually verified.
- Clean DiffIK action teacher is NOT ready.
- Dataset generation, IsaacLab RL, and RoArm-M3-Pro remain blocked.

## Next

- If 1mm tap/reaction is sufficient: stop contact-geometry GPU tuning and carry
  quality-tier metadata separately.
- If 2-3mm transient tap is required: explicitly record that requirement first,
  then design one local strength-preserving timing/through/contact-stop candidate
  before any runtime.
- Do not repeat the same link5-corner position run.
- Do not jump to pose first.
- Do not mix lateral/height/actuator/DLS/cap/top-margin/data/RL/RoArm.
