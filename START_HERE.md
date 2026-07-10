# START_HERE.md

Last updated: 2026-07-10 late KST (D329 current truth: G0a has been running on
the wrong object — the tap-track 10cm cube — since D322; user approved
redefining the G0a object to the G0b cylinder D34 x H90. Next failable
experiment is the D330 cylinder alignment probe.)

## Current Truth

- Active pivot is **grasp track G0a (redefined: cylinder D34 x H90)**, not cube
  repairs, cube collision audits, tap expansion, PPO, VLA, Track A, RoArm
  deployment, or B200.
- D329 audit conclusion (session
  `claudedocs/session_20260710_grasp_g0a_d329_object_mismatch_audit.md`):
  - G0a ran on the tap-track `10cm/0.72kg` cube since D322, pinned by
    `claudedocs/D322_PROMPT.md:60` as an invariant, while the professor
    direction #6 says start with graspable shapes
    (`claudedocs/direction_20260708_grasp_pivot.md:10`; practical opening
    `40-45mm` `:28`).
  - Geometry: the D327 alignment target TCP `(0.260, +0.044, +0.0379)`
    env-local lies inside the cube xy footprint, `50mm` below the cube top;
    the moving-jaw-side volume is solid cube. No D328 candidate path has a
    collision-free corridor. The 100mm cube makes the alignment goal
    structurally collision-bound; the D34 cylinder (< 40-45mm opening) removes
    this collision class.
  - D328's reported numbers all match the `g0a_d328` logs, but two
    instrumentation defects were found: candidate "clearance `70.000mm`" is a
    fallback constant, not a measurement (`d328 probe:205-206`), and
    ContactSensor `0.000N` is caused by `activate_contact_sensors=False`
    (`roarm_rl/roarm_stack_env.py:150`).
  - Gripper convention pinned: `q` LOW = OPEN, HIGH = CLOSED
    (`roarm_rl/roarm_stack_env.py:702-704`); G0a's `q=0.0` hold is fully open.
- New direction doc: `claudedocs/direction_20260708_grasp_pivot.md`.
- D322 prompt source: `claudedocs/D322_PROMPT.md`.
- Grasp-track session docs (evidence trail, all under `claudedocs/`):
  `session_20260709_grasp_g0a_d322_alignment_fail.md`,
  `session_20260709_grasp_g0a_d323_frame_repair_infeasible.md`,
  `session_20260709_grasp_g0a_d324_viz_debug.md`,
  `session_20260709_grasp_g0a_d325_redefined_alignment_fail.md`,
  `session_20260709_grasp_g0a_d326_teleport_static_fail.md`,
  `session_20260710_grasp_g0a_d327_standoff_execution_fail.md`,
  `session_20260710_grasp_g0a_d328_collision_path_repair_fail.md`,
  `session_20260710_grasp_g0a_d329_object_mismatch_audit.md`.
- Runtime outputs: `claudedocs/runtime_logs/grasp_track/g0a_d322/` ...
  `g0a_d328/` (+ `viz_infra_d324/`). D329 is an audit session with no sim run.

## Active Case: G0a (redefined 2026-07-10, user-approved)

- Object: **cylinder D34 x H90** (radius `0.017m`, height `0.090m`), replacing
  the 10cm cube. Mass stays `0.72kg` for D330 (single-variable change:
  geometry only; real-mass calibration is a G0b prep item).
- 이번 case의 신규 변수: 물체 기하 (cube -> cylinder). D330 introduces no other
  variable.
- Invariants:
  - Fixed object position `(x=0.30m, y=0.00m)`; friction `static=1.5`,
    `dynamic=1.2`; state-only; no render (single-frame viz only); no RL/PPO;
    no gripper close; no grasp; no lift; no randomization.
  - Pose family: D325 `position_only_tangent_minus1` (link5 `+x` = tangent
    `-1` within `15deg`, tool axis free).
  - Alignment standoff fixed `2mm`: tangent offset `D/2 - 8mm + 2mm = 11mm`;
    radial offset `D/2 - 10mm = 7mm`; TCP z = cylinder center height
    (env-local `+0.0329`). Future grasp flush formula stays `D/2 - 8mm`.
  - Gates (structure unchanged, reparameterized to D34/H90): TCP `<=5mm`;
    tangent `<=15deg`; gap `[0, 5mm]`; no penetration; contact `>=15mm` below
    object top (`+0.0779` env-local); displacement `<5mm`; 10/10 trials.
  - Contact witness repaired: probe-local
    `env_cfg.robot.spawn.activate_contact_sensors = True` (reactive repair,
    D328-justified).
- Next runtime output path: `claudedocs/runtime_logs/grasp_track/g0a_d330/`.

## Latest Result

- D329 verdict: `D329_G0A_WRONG_OBJECT_CASE_REDEFINE` (audit, no sim run;
  failable experiment deferred to D330 with explicit justification).
- D328 verdict (superseded next-step, still-valid evidence):
  `D328_G0A_COLLISION_DRIVE_REPAIR_FAIL` — cube removed TCP error `1.512mm`
  vs cube present `72.178mm` (commanded `0.927mm` both); far_side_slide repair
  final retest `0/10`, TCP errors `58.656-59.379mm`. The cube-present
  collision finding stands; the "audit the cube collision sweep" next step is
  superseded by the object redefinition.
- D322-D327 history: see `claudedocs/EXPERIMENT_LEDGER.md` rows and the
  session docs listed above (D322 96.63mm fail -> D323 strict family
  infeasible -> D324 viz infra -> D325 tangent family adopted, runtime fail
  58.096mm -> D326 teleport-static penetration 0.151mm stop -> D327 2mm
  standoff static pass, effort repair runtime fail).

## Next Concrete Action

Do **not** resume cube-targeted G0a work, waypoint search, blind drive/effort
tuning, G0b grasp/lift, gripper close, RL/PPO, render, randomization, VLA,
RoArm, or B200.

1. User confirms two D330 design assumptions (session doc `## D330 최소 패치
   설계` item 7): cylinder mass `0.72kg` hold, and alignment z at cylinder
   center height.
2. Implement `sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py`
   per the D329 session-doc design (new file; no edits/renames of existing
   probes or `roarm_rl` env files; probe-local overrides only).
3. Run the D330 10-trial gate (failable). PASS 10/10 ->
   `D330_G0A_CYL_ALIGNMENT_PASS`, discuss G0b entry. FAIL -> per-gate counts +
   working contact-force trace + viz/Rerun; same stall signature would falsify
   the wrong-object explanation for the runtime stall and reopen
   collision/drive audit on the correct object.
4. Keep Visualization DoD (D324) and the IsaacLab package rule (D326).

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` (tail: D329)
4. `claudedocs/EXPERIMENT_LEDGER.md` (tail rows)
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260710_grasp_g0a_d329_object_mismatch_audit.md`
7. `claudedocs/session_20260710_grasp_g0a_d328_collision_path_repair_fail.md`
8. `sim_scripts/cube10cm_top_view_d327_grasp_g0a_standoff_execution_probe.py`
   (target/eval machinery that D330 reparameterizes)

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
- Visualization DoD (D324~) and IsaacLab package rule (D326~; verify/restore
  `numpy==1.26.0`, `psutil==5.9.8` after any install) stay in force.

## Frozen / Background Assets

- Tap track frozen at D321: `1920/2000` accepted (96.0%), dataset under
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/lerobot_dataset`;
  session `claudedocs/session_20260708_cube10cm_top_view_d321_physicality_gate_low_mid_production.md`;
  design draft `claudedocs/design_d321_goal_conditioned_primitive.md`.
- D319 RL data-factory direction is background:
  `claudedocs/direction_20260708_rl_data_factory.md`.
