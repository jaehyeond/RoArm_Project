# START_HERE.md

Last updated: 2026-07-11 KST (D332 current truth: a default PhysX mirror recook
of the live-stage link5 mesh overlaps at the frozen pre-step pose, while the
first runtime sample couples a gripper_link event with 12.117mm global-ground
depenetration. The static causal discriminator is mixed, not confirmed.)

## Current Truth

- Active pivot is **grasp track G0a (redefined: cylinder D34 x H90)**, not cube
  repairs, cube collision audits, tap expansion, PPO, VLA, Track A, RoArm
  deployment, or B200.
- D332 verdict (session
  `claudedocs/session_20260711_grasp_g0a_d332_static_collision_discriminator.md`):
  `D332_G0A_PRESTEP_MIRROR_HULL_OVERLAP_RUNTIME_GRIPPER_CONTACT_SCENE_CONFOUNDED_MIXED`.
  - Frozen canonical target `[0.293, 0.011, 0.032883]m`; deterministic
    HOME-seeded IK command
    `[2.1487,31.1085,112.8005,10.4847,0,0]deg`; commanded TCP error
    `0.817812mm`. Reset jitter was removed and the exact command was logged.
  - Raw link5 STL is clear by `+4.273819mm`; the unrestricted mathematical
    hull overlaps by `6.363467mm`. A default PhysX mirror recook of the exact
    live-stage source mesh (`35` vertices, `48` polygons) overlaps by
    `6.236272mm`. GJK/EPA agree and AABB is not used, but the live articulation
    collider cook/owner was not directly extracted or fully parity-checked.
  - The cylinder sensor tensor/path/reporter hard contract passed. Unfiltered
    baseline net force `7.0632007N` matches weight and validates net reporting
    posthoc; a positive gripper event validates that channel. The frozen
    filtered-support positive control failed, and link4/link5 `0N` channels
    lack independent positive controls.
  - Scene-domain failure: each phase resets the cylinder bottom to
    `z=-0.012117m` while active global ground is at `z=0`, embedding it
    `12.117mm`. Baseline/target first post-step z corrections are
    `+12.256849/+12.707490mm`; target net force is `125.033206N`.
  - First observed post-step sample contains a `gripper_link` filtered event
    (`66.866266N`) and object motion, while sampled link5/link4 are `0N`.
    Ground depenetration and robot contact are coupled; this is not clean onset
    timing or robot-only causal attribution.
  - Final/max object XY displacement `10.282285/10.452925mm`; final/max tilt
    `9.235161/9.439981deg`; final TCP error `3.413499mm`; final joint tracking
    error `0.009325rad`.
  - Critical limit: pre-step mirror overlap strongly supports gap-fill, but it
    is not a direct live-collider result and cannot be joined causally to the
    post-step gripper event. Do not claim the collision class or link5-only
    cause confirmed; do not re-author collision yet.
  - `/World/ground` root filtering failed, but the exact collision-plane path
    was never tested and raw warning output was not retained. Do not generalize
    this to a GPU support-filter limitation.
  - Wrist null-space scan and 10-trial gate were not run. G0a was not promoted.
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
- D331/D330 context remains binding: D330 was 5 low + 1 intermediate + 4 stall
  with object displacement `10/10`; D331 established reset jitter, incomplete
  per-env logging, convexHull stage evidence, tipping-vs-sliding ambiguity, and
  the unmeasured `100-120g` correction.
- Grasp-track session docs (evidence trail, all under `claudedocs/`):
  `session_20260709_grasp_g0a_d322_alignment_fail.md` ... d323, d324, d325,
  d326, d327, d328, `session_20260710_grasp_g0a_d329_object_mismatch_audit.md`,
  `session_20260710_grasp_g0a_d330_cyl_alignment_fail.md`,
  `session_20260711_grasp_g0a_d331_critique_audit_d332_design.md`,
  `session_20260711_grasp_g0a_d332_static_collision_discriminator.md`.
- Runtime outputs: `claudedocs/runtime_logs/grasp_track/g0a_d322/` ...
  `g0a_d332/` (+ `viz_infra_d324/`). D329/D331 are audit sessions, no sim run.

## Active Case: G0a (redefined 2026-07-10, user-approved)

- Object: **cylinder D34 x H90** (radius `0.017m`, height `0.090m`). Mass is
  the `0.72kg` placeholder; **real mass unmeasured — the committed "100-120g
  spec" note is an estimate, measure before G0b** (D331 correction).
- D332 신규 변수: `[]` - measurement/diagnostic state only.
- D333 사전등록 신규 변수: `[support_domain_global_ground_collision_disabled]`
  - remove only the redundant ground collision so the already-frozen TapTable
  frame becomes the sole support; object-relative target/gates stay unchanged.
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
- Contact witness status: D330's post-PLAY/manual robot-link sensors remain an
  articulation-body/view-domain lifecycle contract failure (reporter paths did
  resolve; lifecycle vs articulation support was not separately isolated).
  D332's one cylinder-owned, pre-PLAY sensor passes structural hard checks;
  net reporting and the gripper channel are observed working. The frozen
  support-filter gate failed, exact ground-collider filtering was not tested,
  and complete link4/link5 negative attribution is not validated.
- Latest runtime output path:
  `claudedocs/runtime_logs/grasp_track/g0a_d332/`.

## Latest Result

- D332 verdict:
  `D332_G0A_PRESTEP_MIRROR_HULL_OVERLAP_RUNTIME_GRIPPER_CONTACT_SCENE_CONFOUNDED_MIXED`
  - mirror gap-fill is strongly supported, but the runtime static discriminator
  is scene-confounded; G0a remains incomplete and G0b remains blocked.
- D331 verdict: `D331_G0A_ANALYSIS_AUDIT_D332_DESIGN` (audit, no sim run).
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

D333 is one scene-domain contract repair plus the same one-env static retest:
disable only the redundant global-ground collision so the existing TapTable
top at `TABLE_Z` becomes the sole support, restore a valid TapTable filtered-
force positive control, and keep the object pose, relative target formula,
joint reset, mass/friction, thresholds, and 200+200 steps unchanged. Stop after
deciding whether object motion and a gripper/link event remain without the
`12.117mm` depenetration. Do not re-author collision, tune the target/family,
run ownership search, or advance the alignment ladder in the same session.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` (tail: D331, D332)
4. `claudedocs/EXPERIMENT_LEDGER.md` (tail rows)
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260711_grasp_g0a_d332_static_collision_discriminator.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d332/g0a_d332_static_collision_summary.json`
8. `sim_scripts/cyl34_top_view_d332_grasp_g0a_static_collision_discriminator.py`
9. `claudedocs/session_20260711_grasp_g0a_d331_critique_audit_d332_design.md`
10. `claudedocs/session_20260710_grasp_g0a_d330_cyl_alignment_fail.md`

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
- D332 durable rule: distinguish raw mesh, unrestricted mathematical hull,
  default mirror recook, and directly extracted live collision. Sensor
  initialization and posthoc net reporting cannot replace a frozen filtered-
  support positive control. A scene-domain or body-attribution mismatch blocks
  body-specific repair.
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
