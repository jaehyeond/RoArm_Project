# START_HERE.md

Last updated: 2026-07-12 KST (D334 current truth: the actual gripper tool
geometry overlaps the cylinder at the frozen canonical pose. The step-0
contact is a target-family placement error, not a cook/proxy artifact.
Awaiting the user's repair-direction choice.)

## Current Truth

- Active pivot is **grasp track G0a on cylinder D34 x H90**. Cube repairs,
  G0b close/lift, PPO/RL, VLA, randomization, RoArm real, and B200 remain out
  of scope.
- D334 verdict: `D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED`.
  - Zero new physical variables: all 200+1 physics steps were exact D333
    replays. Replay parity was bit-exact (object/TCP/contact-point deltas
    `0.000000mm`; gripper `76.4128N`, relative delta `0.00e+00`), so every
    frozen-pose conclusion is licensed.
  - Live PhysX ownership is clean 1:1: link5 and gripper_link each own exactly
    their own STL convex hull; cross-body attachments `[]`. The D332
    owner-mismatch hypothesis is refuted.
  - **Gripper raw collision STL itself overlaps the cylinder**: pose A
    (commanded) `-5.9567mm` (EPA `5.863mm`), pose B (post-step-0) `-1.7216mm`
    (EPA `1.722mm`). The recorded contact point sits `-5.3834mm` inside the
    cylinder surface and `0.549mm` on the gripper cooked surface.
  - link5 cooked hull (certified, volume parity `0.0498%`) overlaps only at
    pose A (`-6.2367mm`, reproducing D332's mirror to `0.4um`) and is clear
    (`+3.0438mm`, recorded `0N`) at pose B - never the runtime cause.
  - Gripper cook parity FAILED (`1.46%` > `0.5%`): cooked hull inflates
    `~3.5-9.4mm` beyond raw. Secondary finding, not the primary cause.
  - Direct live cook on instance-proxy prims fails
    (`RESULT_ERROR_COOKING_FAILED` / `RESULT_ERROR_INVALID_PARSING`) -
    recorded; mirror cook gated by live collider volume is the cook route.
- Visualization DoD passed: three inspected PNGs (pose A, pose B, zoomed
  contact map) + one non-empty RRD.

## Interpretation

- The canonical D325 `position_only_tangent_minus1` target places the
  physical tool inside the object. Any approach/10-trial rerun under this
  target family cannot pass regardless of collision-representation fixes.
- Repair precedence: **target-family repair first** (make the commanded pose
  clear the actual tool surface); gripper hull inflation is a separate,
  secondary collision-representation case (variable ladder: do not bundle).
- D330 swept results remain scene-confounded and unreattributed.

## Active Case: G0a

- Object: cylinder radius `0.017m`, height `0.090m`, fixed `(0.300,0.000)`;
  mass placeholder `0.72kg` (real mass unmeasured - required before G0b).
- Friction `static=1.5`, `dynamic=1.2`; no material/mass tuning.
- Target family D325 `position_only_tangent_minus1` (tangent `11mm`, radial
  `7mm`, TCP z at cylinder center) - **now shown to collide; repair pending
  user choice**. Grasp flush formula reference stays `D/2-8mm`.
- G0a gates unchanged: TCP `<=5mm`, tangent `<=15deg`, jaw gap `[0,5mm]`, no
  penetration, contact `>=15mm` below top, displacement `<5mm`, `10/10`.
- Latest runtime output: `claudedocs/runtime_logs/grasp_track/g0a_d334/`.
- Latest detailed session:
  `claudedocs/session_20260712_grasp_g0a_d334_live_collision_shape_ownership_audit.md`.

## Next Concrete Action

**USER DECIDED (2026-07-12, at the D334 pre-registered stop): option 1 —
target-family repair, to run as D335 in a fresh session.**

D335 scope (one new variable: `[target_family_geometry]`): redesign the
canonical target so the actual gripper surface clears the cylinder at the
commanded pose (recompute radial/tangent offsets against the audited gripper
geometry), then gate with the D334 signed-distance harness (tool surface
clear at commanded pose) BEFORE any settle/physics evaluation.

Deferred (separate case, only after D335): gripper collision-representation
repair for the `1.46%` cook-parity / `~3.5-9.4mm` hull inflation.

## Must Read First

1. `AGENTS.md` (then `CLAUDE.md` for Claude-specific workflow)
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D332-D334)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260712_grasp_g0a_d334_live_collision_shape_ownership_audit.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json`
8. `sim_scripts/cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit.py`
9. `claudedocs/session_20260711_grasp_g0a_d333_sole_support_static_retest.md`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale. Memory is an index, not evidence.
- Git commit/push only on explicit user request.
- B200/JHPark/SSH/pull/.ssh and `/half-clone` remain forbidden.
- Variable Ladder: one or two new variables per case; future ideas go to
  `claudedocs/BACKLOG.md`; grasp outputs only under
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`; paths forward-only.
- Unmeasured estimates are not specs. Unrecorded pilots are not decision
  evidence.
- Distinguish raw mesh, mathematical hull, mirror recook, and live-volume-
  gated cook. AABB-only reasoning is forbidden; property-query AABB is
  informational only (frame/scale conventions unverified).
- Visualization DoD and Isaac package pins remain binding:
  `numpy==1.26.0`, `psutil==5.9.8` after any install.

## Frozen Background

- Professor direction: finish one graspable cylinder case first, then
  G0b -> G1a grid -> G1b standalone PPO
  (`claudedocs/direction_20260708_grasp_pivot.md:3-32`).
- Tap track frozen at D321: `1920/2000` accepted (`96.0%`).
- G0b prerequisites: real cylinder mass measurement and BACKLOG
  `tool_surface_union`.
