# START_HERE.md

Last updated: 2026-07-09 KST (D324 current truth: G0a visual debug infra installed; D323 strict side-grasp infeasibility now has readable frame snapshots; no ladder advance.)

## Current Truth

- Active pivot is now **grasp track G0a**, not cube tap expansion, PPO, VLA, Track A, RoArm deployment, or B200.
- Tap track is frozen at D321 as a useful producer/data-pipeline asset:
  `1920/2000` accepted low/mid script-v2 episodes (`96.0%`), LeRobot append passed, but it is +x-only, not direction-diverse, and not learned-policy success.
- New direction doc:
  `claudedocs/direction_20260708_grasp_pivot.md`.
- D322 prompt source:
  `claudedocs/D322_PROMPT.md`.
- D322 session doc:
  `claudedocs/session_20260709_grasp_g0a_d322_alignment_fail.md`.
- D322 primary runtime output:
  `claudedocs/runtime_logs/grasp_track/g0a_d322/g0a_alignment_summary.json`.
- D322 long-hold diagnostic sidecar:
  `claudedocs/runtime_logs/grasp_track/g0a_d322_longtrack_check/g0a_alignment_summary.json`.
- D323 repair session doc:
  `claudedocs/session_20260709_grasp_g0a_d323_frame_repair_infeasible.md`.
- D323 frame audit / infeasibility outputs:
  `claudedocs/runtime_logs/grasp_track/g0a_d323/frame_audit.json`,
  `claudedocs/runtime_logs/grasp_track/g0a_d323/g0a_d323_alignment_summary.json`.
- D324 visual debug session doc:
  `claudedocs/session_20260709_grasp_g0a_d324_viz_debug.md`.
- D324 visual debug outputs:
  `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_viz_debug_summary.json`,
  `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_strict_target_vs_best_attempt.png`,
  `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_position_only_tangent_minus1.png`.

## Active Case: G0a

- New variable remains D322's `grasp pose geometry`; D323 and D324 introduced no new variable and are repair/tool-only.
- Invariants:
  - Existing 10cm cube, mass `0.72kg`.
  - Fixed object position `(x=0.30m, y=0.00m)`.
  - Friction stays `static=1.5`, `dynamic=1.2`.
  - State-only, no render, no RL/PPO, no gripper close, no grasp, no lift.
  - 10cm cube is too wide for the measured practical gripper opening (~40-45mm), so G0a is alignment-only.
- Pre-registered success criteria:
  1. TCP pose error `<=5mm` and pose/orientation error `<=3deg`.
  2. Fixed-jaw grasp face to cube face gap `<=3mm` with no penetration.
  3. Cube XY displacement `<5mm`.
  4. All 10 trials pass the same condition.
- Output paths: D322 `claudedocs/runtime_logs/grasp_track/g0a_d322/`; D323 `claudedocs/runtime_logs/grasp_track/g0a_d323/`; D324 `claudedocs/runtime_logs/grasp_track/viz_infra_d324/`.

## Latest Result

- D324 tool verdict: `D324_VIZ_DEBUG_SNAPSHOTS_PASS`.
- D324 installed `roarm_rl/viz_debug.py`, `claudedocs/HOWTO_viz_debug.md`, and
  opt-in `--viz_debug_snapshots` hooks for the D322/D323 G0a probes.
- D324 visual gate:
  - strict target vs best-attempt PNG is readable and shows the D323 miss:
    TCP error `35.729mm`, link5 `+x` error `5.942deg`, link5 `+z` error
    `43.015deg`.
  - position-only tangent `-1` PNG is readable and shows the core trade-off:
    TCP error `0.261mm`, link5 `+x` error `9.148deg`, link5 `+z` error
    `69.124deg`.
  - candidate sketch table is in
    `claudedocs/runtime_logs/grasp_track/viz_infra_d324/d324_candidate_pose_table.md`.
- D324 did not change G0a criteria, offsets, object/friction, gripper state, or
  ladder stage. Isaac marker helper exists, but final visual gate used the
  deterministic matplotlib backend; rerun `.rrd` was not generated because
  `rerun-sdk` is not installed.
- D323 repair verdict: `D323_G0A_STRICT_POSE_INFEASIBLE_STOP`.
- D323 frame audit:
  - `hand_tcp` is not a separate runtime body; TCP is computed from link5.
  - `TCP = link5 + link5_rotation * [0, 0, 0.115428]m`.
  - measured TCP-in-link5 offset error across three static poses: `0.000044~0.000063mm`.
  - `gripper_link` origin is not the fixed-jaw face; audited `gripper_link in link5` is approximately `[0, 0.018821, 0.052035]m`.
- D323 strict pose feasibility:
  - requested family: link5 `+z` horizontal radial, link5 `+x` horizontal tangent, tangent offset `42mm`, radial tip depth `10mm`.
  - best strict attempt still failed: TCP error `35.729mm`, link5 `+x` error `5.942deg`, link5 `+z` error `43.015deg`.
  - position-only side target is reachable (`0.261mm` TCP error), but then link5 `+z` is about `69.124deg` away from radial.
  - Step 3 retrial was not run because the prompt required stopping when the strict pose family is impossible.
- D322 primary G0a runtime verdict: `D322_G0A_ALIGNMENT_FAIL` remains the previous runtime failure.
- Primary 10-trial result:
  - pass all criteria: `0/10`.
  - failure counts: TCP pose `10/10`, fixed-jaw gap `10/10`, fixed-jaw penetration `10/10`, cube displacement `0/10`, base yaw `0/10`.
  - mean TCP pose error: `96.63mm`.
  - mean fixed-jaw signed face gap: `-56.00mm` (penetration by proxy definition).
  - mean cube XY displacement: `0.019mm`.
  - mean max arm joint tracking error: `0.174rad`.
- Long-hold diagnostic (`500` approach + `500` hold, `12s`) also failed:
  - pass all criteria: `0/10`.
  - mean TCP pose error: `96.49mm`.
  - mean fixed-jaw signed face gap: `-55.78mm`.
  - mean max arm joint tracking error: `0.174rad`.
- Interpretation:
  - D322's offset-direction repair was necessary but insufficient.
  - D323 confirms the TCP frame contract and rejects the stricter horizontal side-grasp orientation family as unreachable at the 10cm cube center height.
  - The remaining G0a problem is target-representation feasibility, not PPO, not friction, not offset-number tuning.

## Next Concrete Action

Do **not** start G0b, cylinder spawn, gripper close, lift, RL/PPO, render, VLA, RoArm, B200, friction/material changes, or position randomization.

Next session should repair G0a only:

1. Do not advance to G0b/cylinder/gripper close.
2. Do not loop on `42mm` or `10mm` offsets; D323 shows the blocker is orientation-family feasibility.
3. Define an attainable G0a alignment criterion from the audited frame contract and D324 snapshots: fixed-jaw/TCP side position plus reachable wrist-axis family, not strict link5 `+z` horizontal radial if that remains infeasible.
4. Only after that criterion is explicit, rerun the same fixed cube/friction/no-close/no-render 10-trial G0a alignment criteria.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md`
4. `claudedocs/EXPERIMENT_LEDGER.md`
5. `claudedocs/D322_PROMPT.md`
6. `claudedocs/direction_20260708_grasp_pivot.md`
7. `claudedocs/session_20260709_grasp_g0a_d323_frame_repair_infeasible.md`
8. `claudedocs/session_20260709_grasp_g0a_d324_viz_debug.md`
9. `claudedocs/HOWTO_viz_debug.md`
10. `claudedocs/session_20260709_grasp_g0a_d322_alignment_fail.md`
11. `sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py`
12. `sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale; do not use them as current truth.
- Memory is only a helper index. Repo docs/logs/code are current truth.
- B200/JHPark/SSH/pull/.ssh copy are forbidden for this branch unless the user explicitly changes the rule.
- Existing dirty/untracked/ahead state must not be reverted.
- Variable Ladder Protocol (D322~):
  - Each case may introduce only one or two new variables.
  - Session docs must state `이번 case의 신규 변수: [...]`.
  - Future ideas go to `claudedocs/BACKLOG.md`, not into implementation.
  - `START_HERE.md` Active Case is the single source of truth.
  - New grasp outputs go under `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`.
- Visualization Definition of Done (D324~):
  - Geometry/pose/contact probes should emit target-vs-actual frame diagnostics
    and decision-time snapshots via `roarm_rl.viz_debug` when practical.
  - This is single-frame debugging only, not permission for large renders.

## Frozen / Background Assets

- D321 tap dataset:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/lerobot_dataset`.
- D321 session:
  `claudedocs/session_20260708_cube10cm_top_view_d321_physicality_gate_low_mid_production.md`.
- D321 design draft:
  `claudedocs/design_d321_goal_conditioned_primitive.md`.
- D319 RL data-factory direction is now background, not the active case:
  `claudedocs/direction_20260708_rl_data_factory.md`.
