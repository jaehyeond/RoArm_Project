# START_HERE.md

Last updated: 2026-07-10 KST (D328 current truth: cube-removal discriminator confirms G0a runtime stall is cube-present collision/path geometry, not pure drive override; one path repair still fails 0/10; no ladder advance.)

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
- D325 redefined alignment session doc:
  `claudedocs/session_20260709_grasp_g0a_d325_redefined_alignment_fail.md`.
- D325 runtime outputs:
  `claudedocs/runtime_logs/grasp_track/g0a_d325/g0a_d325_alignment_summary.json`,
  `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_01_snapshot.png`,
  `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_01_frames.rrd`.
- D326 execution-contract diagnosis session doc:
  `claudedocs/session_20260709_grasp_g0a_d326_teleport_static_fail.md`.
- D326 runtime outputs:
  `claudedocs/runtime_logs/grasp_track/g0a_d326/g0a_d326_execution_contract_summary.json`,
  `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_check.png`,
  `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_v2.rrd`.
- D327 standoff/execution session doc:
  `claudedocs/session_20260710_grasp_g0a_d327_standoff_execution_fail.md`.
- D327 runtime outputs:
  `claudedocs/runtime_logs/grasp_track/g0a_d327/g0a_d327_standoff_execution_summary.json`,
  `claudedocs/runtime_logs/grasp_track/g0a_d327/g0a_d327_final_retest_trials.csv`,
  `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_final_effort_retest_trace_v2.rrd`.
- D328 collision-vs-drive session doc:
  `claudedocs/session_20260710_grasp_g0a_d328_collision_path_repair_fail.md`.
- D328 runtime outputs:
  `claudedocs/runtime_logs/grasp_track/g0a_d328/g0a_d328_collision_vs_drive_summary.json`,
  `claudedocs/runtime_logs/grasp_track/g0a_d328/g0a_d328_final_retest_trials.csv`,
  `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trace_v2.rrd`.

## Active Case: G0a

- New variable remains D322's `grasp pose geometry`; D323-D328 introduced no ladder variable and are repair/tool/criterion/execution-diagnostic only inside G0a.
- Invariants:
  - Existing 10cm cube, mass `0.72kg`.
  - Fixed object position `(x=0.30m, y=0.00m)`.
  - Friction stays `static=1.5`, `dynamic=1.2`.
  - State-only, no render, no RL/PPO, no gripper close, no grasp, no lift.
  - 10cm cube is too wide for the measured practical gripper opening (~40-45mm), so G0a is alignment-only.
  - D327 alignment target uses a fixed `2mm` standoff: tangent offset `D/2 - 8mm + 2mm`.
    This is separate from the future grasp flush formula `D/2 - 8mm`.
- Active D325 G0a criterion:
  1. TCP position error `<=5mm`.
  2. `link5 +x` jaw-separation axis aligns with tangent `-1` within `15deg`.
  3. Fixed-jaw face to cube side: horizontal gap `<=5mm`, no penetration, and contact point at least `15mm` below cube top.
  4. Cube XY displacement `<5mm`; all 10 trials pass.
- Output paths: D322 `claudedocs/runtime_logs/grasp_track/g0a_d322/`; D323 `claudedocs/runtime_logs/grasp_track/g0a_d323/`; D324 `claudedocs/runtime_logs/grasp_track/viz_infra_d324/`; D325 `claudedocs/runtime_logs/grasp_track/g0a_d325/`; D326 `claudedocs/runtime_logs/grasp_track/g0a_d326/`; D327 `claudedocs/runtime_logs/grasp_track/g0a_d327/`; D328 `claudedocs/runtime_logs/grasp_track/g0a_d328/`.

## Latest Result

- D328 collision-vs-drive verdict:
  `D328_G0A_COLLISION_DRIVE_REPAIR_FAIL`.
- D328 decision experiment:
  - cube removed: TCP error `1.512mm`, commanded TCP error `0.927mm`,
    joint error `0.00193rad`.
  - cube present evidence: TCP error `72.178mm`, commanded TCP error
    `0.927mm`, joint error `0.132rad`.
  - judgement: branch A collision/path confirmed. The target and joint command
    path are reachable in free space; the cube-present runtime path is the
    blocker.
- D328 evidence caveat:
  - torque saturation remained visible in the cube-present trial
    (`max=1.0`, final `0.8`), but now reads as a consequence of the blocked
    cube-present path rather than a standalone drive-semantics root cause.
  - D328 ContactSensor logging returned `0.000N` max contact force even though
    removing the cube changed the outcome decisively. Treat that contact channel
    as insufficient evidence until contact instrumentation is repaired.
- D328 branch-A repair:
  - candidate paths checked: `d327_radial`, `far_side_slide`,
    `high_corridor_drop`.
  - selected repair: `far_side_slide`, based on IK feasibility, `70.000mm`
    approach TCP-over-top clearance, and lower max IK error (`0.910mm`) than
    `d327_radial`.
  - limitation: the clearance metric is an approach-corridor proxy, not a full
    moving-jaw/link5 collision sweep.
- D328 final 10-trial retest still failed `0/10`:
  - TCP pose failures `10/10`.
  - fixed-jaw gap failures `10/10`.
  - contact-height failures `10/10`.
  - jaw tangent failures `0/10`, penetration failures `0/10`, cube displacement
    failures `0/10`.
  - final TCP errors were `58.656-59.379mm`, improved from the cube-present
    evidence `72.178mm` but still far outside the `<=5mm` gate.
- D328 implication: the next valid G0a repair is true open-gripper collision
  geometry/sweep and contact-witness audit. Do not continue blind drive/effort
  tuning, do not tune standoff/gates/offsets, and do not advance to G0b.
- D327 standoff/execution verdict:
  `D327_G0A_STANDOFF_EFFORT_REPAIR_FAIL`.
- D327 fixed the D326 static blocker by separating alignment and future grasp
  targets:
  - alignment tangent offset is now `D/2 - 8mm + 2mm`.
  - future grasp flush formula remains `D/2 - 8mm`.
  - epsilon/standoff is fixed at `2mm`; do not tune it.
- D327 teleport-static result passed all D325 gates:
  - TCP error `0.349mm` PASS.
  - jaw tangent error `9.602deg` PASS.
  - fixed-jaw gap `1.837mm` PASS.
  - penetration `0.000mm` PASS.
  - contact point below top `49.733mm` PASS.
  - cube displacement `0.000mm` PASS.
- D327 runtime diagnosis:
  - baseline runtime pass-all `0/10`, trial-1 TCP error `71.004mm`, commanded TCP error `0.926mm`.
  - x3 approach/hold did not help: trial-1 TCP error worsened to `72.719mm`.
  - lead-limit and step-clip explanations were rejected for this path.
  - joint/drive saturation was supported: torque saturation max `1.0`, final joint error `0.143rad`.
- D327 applied exactly one execution-contract repair, `arm_effort_limit_sim=8.0`,
  and still failed final `0/10`:
  - TCP pose failures `10/10`.
  - fixed-jaw gap failures `10/10`.
  - contact-height failures `10/10`.
  - jaw tangent failures `0/10`, penetration failures `0/10`, cube displacement failures `0/10`.
  - final TCP errors stayed `67.556-70.849mm` while commanded TCP errors stayed under `0.928mm`.
- D327 implication: the remaining blocker is not static geometry, not time budget,
  and not a simple effort-limit increase. Next G0a work is actuator/drive
  semantics: USD-authored drive stiffness/damping/limits, external target override
  semantics, and commanded-vs-actual joint evolution in Rerun traces.
- D326 historical correction: before D327, teleport-static failed only by
  fixed-jaw penetration `0.151mm`; D327's standoff resolves that specific blocker.
- D325 criterion repair verdict: `D325_G0A_REDEFINED_ALIGNMENT_FAIL`.
- D325 adopted D324 `position_only_tangent_minus1` as the active G0a family:
  `link5 +z` tool axis is free; `link5 +x` must align with tangent `-1`;
  42mm tangent offset and 10mm radial tip depth were not changed.
- D325 10-trial runtime result:
  - pass all criteria: `0/10`.
  - failure counts: TCP pose `10/10`, jaw tangent `0/10`, fixed-jaw gap
    `10/10`, penetration `0/10`, contact height `10/10`, cube displacement
    `0/10`.
  - mean TCP error `58.096mm`; mean jaw tangent error `10.765deg`; mean gap
    `11.996mm`; mean contact point below cube top `1.175mm`; mean cube
    displacement `0.026mm`.
  - max arm joint tracking error was `0.116-0.137rad`.
- D325 interpretation: the adopted tangent family itself is not the blocker
  because tangent passed `10/10`; runtime motion did not reach the low side TCP
  target. Next work is actuator/trajectory contract diagnosis, not G0b and not
  gate/offset tuning.
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
  - D325 showed runtime did not reach the low side TCP pose.
  - D326 shows actuator/trajectory repair is premature: teleport-static D325 geometry still violates the fixed-jaw no-penetration gate by `0.151mm`.
  - D327 shows the static fixed-jaw geometry blocker is repaired by the fixed
    `2mm` alignment standoff, but runtime still cannot reach the low-side target
    even after the single effort-limit repair to `8.0`.
  - The remaining G0a problem is runtime actuator/drive semantics, not PPO, not
    friction, not target/gate/epsilon tuning, and not simple step-extension/effort
    escalation.

## Next Concrete Action

Do **not** start G0b, cylinder spawn, gripper close, lift, RL/PPO, render, VLA, RoArm, B200, friction/material changes, or position randomization.

Next session should repair G0a only:

1. Do not advance to G0b/cylinder/gripper close.
2. Do not tune D327 standoff, `42mm`, `10mm`, `15deg`, `15mm`, object/friction,
   or pose family.
3. Do not apply another blind effort/step escalation. D327 already rejected x3
   time extension and `arm_effort_limit_sim=8.0`; D328 shows free-space runtime
   reaches the same target.
4. Do not rely on the current D328 ContactSensor force trace as the sole contact
   witness. It returned `0.000N` while the cube-removal decision experiment
   changed the outcome.
5. Audit true open-gripper collision geometry and sweep: fixed jaw, moving jaw,
   link5, cube, and table. Use snapshots/Rerun plus a repaired contact witness
   or deterministic geometry sweep before trying another path repair.
6. Re-run D325 four-condition 10-trial gate only after one specifically justified
   collision-path/contact-instrumentation repair. Keep Visualization DoD.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md`
4. `claudedocs/EXPERIMENT_LEDGER.md`
5. `claudedocs/D322_PROMPT.md`
6. `claudedocs/direction_20260708_grasp_pivot.md`
7. `claudedocs/session_20260709_grasp_g0a_d323_frame_repair_infeasible.md`
8. `claudedocs/session_20260709_grasp_g0a_d324_viz_debug.md`
9. `claudedocs/session_20260709_grasp_g0a_d325_redefined_alignment_fail.md`
10. `claudedocs/session_20260709_grasp_g0a_d326_teleport_static_fail.md`
11. `claudedocs/session_20260710_grasp_g0a_d327_standoff_execution_fail.md`
12. `claudedocs/session_20260710_grasp_g0a_d328_collision_path_repair_fail.md`
13. `claudedocs/HOWTO_viz_debug.md`
14. `claudedocs/session_20260709_grasp_g0a_d322_alignment_fail.md`
15. `sim_scripts/cube10cm_top_view_d328_grasp_g0a_collision_vs_drive_probe.py`
16. `sim_scripts/cube10cm_top_view_d327_grasp_g0a_standoff_execution_probe.py`
17. `sim_scripts/cube10cm_top_view_d326_grasp_g0a_execution_contract_probe.py`
18. `sim_scripts/cube10cm_top_view_d325_grasp_g0a_redefined_alignment_probe.py`
19. `sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py`
20. `sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py`

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
- IsaacLab environment package rule (D326~):
  - After any package install into `isaaclab`, record dependency impact and
    verify/restore `numpy==1.26.0` and `psutil==5.9.8`.

## Frozen / Background Assets

- D321 tap dataset:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/lerobot_dataset`.
- D321 session:
  `claudedocs/session_20260708_cube10cm_top_view_d321_physicality_gate_low_mid_production.md`.
- D321 design draft:
  `claudedocs/design_d321_goal_conditioned_primitive.md`.
- D319 RL data-factory direction is now background, not the active case:
  `claudedocs/direction_20260708_rl_data_factory.md`.
