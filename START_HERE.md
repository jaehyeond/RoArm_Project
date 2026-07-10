# START_HERE.md

Last updated: 2026-07-11 KST (D331 current truth: an external second-AI
critique of the D330 analysis was audited line-by-line and all six corrections
were verified; link5/gripper_link convexHull collision is now direct USD stage
evidence; the next failable experiment is the pre-registered D332 static
collision discriminator.)

## Current Truth

- Active pivot is **grasp track G0a (redefined: cylinder D34 x H90)**, not cube
  repairs, cube collision audits, tap expansion, PPO, VLA, Track A, RoArm
  deployment, or B200.
- D331 audit conclusion (session
  `claudedocs/session_20260711_grasp_g0a_d331_critique_audit_d332_design.md`):
  - D330 trial regimes: 5 low + 1 intermediate (trial 1: TCP `27.233mm`,
    z `+19mm`, displacement `39.456mm` max) + 4 stall (trials 5/7/9/10:
    `70.3-80.5mm`, z `0.090-0.098m`, joint err `0.142-0.170rad`).
  - The 10 parallel envs are NOT identical: reset applies per-env `±0.02rad`
    joint jitter (`roarm_rl/roarm_cube_push_env.py:1636`) and the IK is
    position-only (2-dim wrist null space). Per-env commanded joint vectors
    and the object's final z/quaternion were never recorded (rrd = env0 only).
  - Direct USD evidence (pxr instance-proxy traversal): link5 and gripper_link
    collision use `physics:approximation=convexHull` — the open-jaw pocket is
    filled by the hull. This upgrades D231's inference. gripper_link collision
    is a ~4mm proxy (effectively no moving-jaw collision).
  - Physics: tipping threshold `~2.67N` << sliding `~10.59N` for the standing
    cylinder at a center-height push — D330's XY displacement cannot be
    attributed to slide vs tip, and upright-object proxy gates are silently
    invalid if the object tilts.
  - Committed D330 `mass_note` "real 100-120g spec" is an unmeasured estimate
    (user-confirmed); real cylinder mass must be measured before G0b.
  - External scratch overlap pilot (raw STL `+4.04mm` clearance vs single hull
    `~6.545mm` penetration at env0 commanded pose) is unrecorded/out-of-repo:
    supportive but not decision evidence until reproduced in D332.
- D330 runtime conclusion (session
  `claudedocs/session_20260710_grasp_g0a_d330_cyl_alignment_fail.md`):
  final 10-trial gate `0/10` (TCP pose `8/10`, tangent `0/10`, gap `3/10`,
  penetration `0/10`, contact height `4/10`, object displacement `10/10`);
  mean TCP error `36.033mm` (min/max `1.884/80.530mm`), commanded `0.404mm`,
  mean object XY displacement `19.070mm`. Wrong-object (D329) was necessary
  but not sufficient. Robot-link ContactSensors fail PhysX view init even with
  `activate_contact_sensors=True`; `0.000N` is a sensor-contract failure.
- D329 audit (session
  `claudedocs/session_20260710_grasp_g0a_d329_object_mismatch_audit.md`):
  G0a ran on the tap-track 10cm cube D322-D328 (7 sessions, 0/10); user
  approved redefining the object to the G0b cylinder D34 x H90. Gripper
  convention: `q` LOW = OPEN (`roarm_rl/roarm_stack_env.py:702-704`).
- Grasp-track session docs (evidence trail, all under `claudedocs/`):
  `session_20260709_grasp_g0a_d322_alignment_fail.md` ... d323, d324, d325,
  d326, d327, d328, `session_20260710_grasp_g0a_d329_object_mismatch_audit.md`,
  `session_20260710_grasp_g0a_d330_cyl_alignment_fail.md`,
  `session_20260711_grasp_g0a_d331_critique_audit_d332_design.md`.
- Runtime outputs: `claudedocs/runtime_logs/grasp_track/g0a_d322/` ...
  `g0a_d330/` (+ `viz_infra_d324/`). D329/D331 are audit sessions, no sim run.

## Active Case: G0a (redefined 2026-07-10, user-approved)

- Object: **cylinder D34 x H90** (radius `0.017m`, height `0.090m`). Mass is
  the `0.72kg` placeholder; **real mass unmeasured — the committed "100-120g
  spec" note is an estimate, measure before G0b** (D331 correction).
- 이번 case의 신규 변수: 없음 — D332 is a static discriminator on the same
  object/target/gates.
- Invariants:
  - Fixed object position `(x=0.30m, y=0.00m)`; friction `static=1.5`,
    `dynamic=1.2`; state-only; no render (single-frame viz only); no RL/PPO;
    no gripper close; no grasp; no lift; no randomization.
  - Pose family: D325 `position_only_tangent_minus1` (link5 `+x` = tangent
    `-1` within `15deg`, tool axis free).
  - Alignment standoff fixed `2mm`: tangent offset `11mm`; radial offset
    `7mm`; TCP z = cylinder center height (env-local `+0.0329`). Future grasp
    flush formula stays `D/2 - 8mm`.
  - Gates (structure unchanged): TCP `<=5mm`; tangent `<=15deg`; gap
    `[0, 5mm]`; no penetration; contact `>=15mm` below object top
    (`+0.0779` env-local); displacement `<5mm`; 10/10 trials.
- Contact witness status: robot-link ContactSensors are broken (PhysX view
  init failure). D332 uses a cylinder-side scene-owned pre-PLAY sensor with
  robot-link filters, plus an explicit force>0 validation step (init success
  is not reporting proof — D328/D330 lesson).
- Latest runtime output path: `claudedocs/runtime_logs/grasp_track/g0a_d330/`.
  Next: `claudedocs/runtime_logs/grasp_track/g0a_d332/`.

## Latest Result

- D331 verdict: `D331_G0A_ANALYSIS_AUDIT_D332_DESIGN` (audit, no sim run;
  failable experiment deferred to D332 with explicit justification).
- D330 verdict: `D330_G0A_CYL_ALIGNMENT_FAIL` — correct-cylinder alignment
  failed `0/10`; G0a incomplete; G0b blocked.
- D329 verdict: `D329_G0A_WRONG_OBJECT_CASE_REDEFINE`.
- D322-D328 history: see `claudedocs/EXPERIMENT_LEDGER.md` rows and session
  docs (cube-era: D322 96.63mm fail -> D323 strict family infeasible -> D324
  viz infra -> D325 tangent family, runtime fail 58.096mm -> D326
  teleport-static 0.151mm stop -> D327 standoff static pass, effort repair
  fail -> D328 cube-present collision discriminator, repair fail).

## Next Concrete Action

Do **not** resume cube-targeted work, waypoint search, blind drive/effort
tuning, collision-mesh re-authoring, G0b grasp/lift, gripper close, RL/PPO,
render, randomization, VLA, RoArm, or B200.

D332 (pre-registered in the D331 session doc; user confirms design first):

1. Offline actual-hull overlap: convex hull of the link5 collision mesh at
   commanded poses vs cylinder D34 x H90 — penetration/clearance as a recorded
   artifact. No AABB shortcut. (No Isaac needed.)
2. Fixed reset (jitter=0) + per-env commanded joint vectors logged to CSV.
3. Teleport + controlled settle steps; log object full pose (xy, z,
   quaternion) — do not clone d326's zero-physics-step static check.
4. Validated cylinder-side contact witness (see Active Case).
5. Wrist null-space family scan for overlap pose-sensitivity (explains the
   D330 regime split candidate).

Semantics: hull overlap at commanded pose confirmed -> blocker = collision
geometry class -> repair options (true jaw collision authoring vs alignment
family change) go to the user. No overlap + no static disturbance ->
runtime/drive audit reopens.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` (tail: D330, D331)
4. `claudedocs/EXPERIMENT_LEDGER.md` (tail rows)
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260711_grasp_g0a_d331_critique_audit_d332_design.md`
7. `claudedocs/session_20260710_grasp_g0a_d330_cyl_alignment_fail.md`
8. `claudedocs/session_20260710_grasp_g0a_d329_object_mismatch_audit.md`
9. `sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale; do not use them as current truth.
- Memory is only a helper index. Repo docs/logs/code are current truth.
- B200/JHPark/SSH/pull/.ssh copy are forbidden for this branch unless the user
  explicitly changes the rule.
- Existing dirty/untracked/ahead state must not be reverted.
- Variable Ladder Protocol (D322~): 1-2 new variables per case; session docs
  state `이번 case의 신규 변수: [...]`; future ideas go to
  `claudedocs/BACKLOG.md`; this Active Case section is the single source of
  truth; grasp outputs only under
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`; folders forward-only.
- D329 durable rule: an active case must use an object class consistent with
  the ladder's completion target unless the user explicitly approves a proxy.
- D331 durable rule: unmeasured estimates must not be recorded as "real spec"
  in artifacts; unrecorded out-of-repo pilot numbers are not decision evidence
  until reproduced as recorded artifacts.
- Visualization DoD (D324~) and IsaacLab package rule (D326~; verify/restore
  `numpy==1.26.0`, `psutil==5.9.8` after any install) stay in force.

## Frozen / Background Assets

- Tap track frozen at D321: `1920/2000` accepted (96.0%), dataset under
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/lerobot_dataset`;
  session `claudedocs/session_20260708_cube10cm_top_view_d321_physicality_gate_low_mid_production.md`;
  design draft `claudedocs/design_d321_goal_conditioned_primitive.md`.
- D319 RL data-factory direction is background:
  `claudedocs/direction_20260708_rl_data_factory.md`.
- G0b precondition (BACKLOG): `tool_surface_union` (D231) — moving-jaw 4mm
  collision proxy cannot support grasp physics.
