# Session 2026-07-09 — D324 G0a Visual Debug Infrastructure

Verdict: `D324_VIZ_DEBUG_SNAPSHOTS_PASS`

이번 case의 신규 변수: `[]`. D324 is a tool/infrastructure session for the
active G0a case. It does not change G0a criteria, 42/10mm offsets, object,
friction, gripper state, or ladder stage.

## Scope

- Active case: G0a only.
- Purpose: make D323 target-vs-actual frame mismatch visually inspectable before
  redefining an attainable G0a alignment criterion.
- Failable experiment: D323 strict and position-only cases must produce
  single-frame diagnostics where frame mismatch is readable.
- Non-goals: no G0b, no cylinder, no gripper close, no grasp/lift, no RL/PPO, no
  VLA/RoArm/B200, no large render, no criterion adoption.

## Code And Docs

- Added common helper: `roarm_rl/viz_debug.py`
  - `draw_frames(pairs)` uses Isaac Lab `VisualizationMarkers` when an Isaac app
    is available.
  - `snapshot(path, pairs=...)` can try viewport capture and then falls back to a
    deterministic matplotlib frame plot.
  - `log_rerun(...)` is optional.
- Added D324 demo: `sim_scripts/cube10cm_top_view_d324_viz_debug_demo.py`.
- Added HOWTO: `claudedocs/HOWTO_viz_debug.md`.
- Added D324+ durable rule in `CLAUDE.md`: geometry/pose/contact probes should
  emit target-vs-actual frame diagnostics and decision-time snapshots.
- Added opt-in `--viz_debug_snapshots` to:
  - `sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py`
  - `sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py`
- Updated `claudedocs/direction_20260708_grasp_pivot.md` to point the gripper
  coordinate-frame canonical section at the D324 visual artifacts.

## Runtime Artifacts

Output folder:

`claudedocs/runtime_logs/grasp_track/viz_infra_d324/`

Primary snapshots:

| artifact | path | result |
|---|---|---|
| strict target vs best-attempt | `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_strict_target_vs_best_attempt.png` | readable: TCP miss and tool-axis tilt visible |
| position-only tangent -1 | `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_position_only_tangent_minus1.png` | readable: origin nearly overlaps, axes diverge |

Candidate sketches:

| candidate | position err mm | jaw +x err deg | tool +z err deg | snapshot |
|---|---:|---:|---:|---|
| `position_only_tangent_minus1` | 0.261 | 9.148 | 69.124 | `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_candidate_position_only_tangent_minus1.png` |
| `tilt_reduced_weight_0p02` | 4.942 | 5.486 | 53.541 | `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_candidate_tilt_reduced_weight_0p02.png` |
| `strict_best_weight_0p10` | 35.729 | 5.942 | 43.015 | `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_candidate_strict_best_weight_0p10.png` |

Machine-readable summary:

`claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_viz_debug_summary.json`

Candidate table:

`claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_candidate_pose_table.md`

## Visual Gate

The final D324 run was:

```bash
conda run -n isaaclab --no-capture-output \
  python sim_scripts/cube10cm_top_view_d324_viz_debug_demo.py --skip_isaac_markers
```

Result:

`[d324-viz] verdict=D324_VIZ_DEBUG_SNAPSHOTS_PASS visual_gate=True marker_ok=False rerun_ok=False out_dir=claudedocs/runtime_logs/grasp_track/viz_infra_d324`

The two required PNGs were opened and inspected:

- Strict case: the actual TCP is visibly displaced from the target, and the
  actual blue `+z` tool axis is tilted away from the target radial axis. The
  image annotation records `35.729mm`, `43.015deg`, and `5.942deg`.
- Position-only case: the target and actual TCP origins nearly overlap, but the
  actual axes diverge from the target axes. The image annotation records
  `0.261mm`, `69.124deg`, and `9.148deg`.

Therefore the D324 failable experiment passed.

## Marker And Rerun Status

- `draw_frames()` is implemented in `roarm_rl/viz_debug.py`.
- A first run with Isaac marker path initialized local Isaac/Kit on the RTX 4090
  Laptop GPU, but Kit terminated before the script wrote summary metadata. To
  avoid treating a marker lifecycle issue as a G0a visual-gate failure, the final
  failable demo used the deterministic matplotlib snapshot backend.
- `rerun-sdk` is not installed in the local environment:
  `ModuleNotFoundError("No module named 'rerun'")`.
- `.rrd` was therefore not generated. This is acceptable under the D324 prompt:
  rerun logging is optional and failure is non-blocking if PNG/frame diagnostics
  pass.

## Interpretation

D324 does not solve G0a. It makes the D323 blocker visible:

- The strict horizontal side-grasp family is not merely off by a scalar offset;
  its requested orientation is incompatible with the reachable wrist family at
  the current 10cm cube side target.
- The position-only target is reachable, so the next valid G0a step is to define
  a reachable alignment criterion from the audited frame contract, not to tune
  `42mm` or `10mm`.

## Verification

- `python -m py_compile roarm_rl/viz_debug.py sim_scripts/cube10cm_top_view_d324_viz_debug_demo.py sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py` PASS.
- D324 visual gate PASS.
