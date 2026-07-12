# START_HERE.md

Last updated: 2026-07-12 KST (D335 current truth: no raw-tool-clear target was
found in the pre-registered finite grasp-semantic radial/tangent candidate set;
the pre-physics gate correctly prevented all settle physics.)

## Current Truth

- Active pivot remains **grasp track G0a on cylinder D34 x H90**. Cube repair,
  G0b close/lift, PPO/RL, VLA, randomization, real RoArm, and B200 remain out of
  scope.
- D335 verdict: `D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP`.
  - D334 old-target negative control was bit-exact with controlled physics `0`:
    link5 raw `+4.2726455336mm` / CLEAR, gripper raw `-5.9566769497mm` /
    OVERLAP, both deltas `0.000000mm`.
  - Deterministic bounded search evaluated `2,629` unique candidates (`1,449`
    coarse + `1,180` refinement). Frozen alignment gates passed `2,560`;
    complete raw-tool clearance and full pass were both `0`.
  - link5 raw was CLEAR `2,629/2,629`; gripper raw was OVERLAP `2,422`,
    BORDERLINE `207`, CLEAR `0`.
  - Best registered ranking row was `(r,t)=(14.6,13.9)mm`: link5
    `+7.787464mm` / CLEAR, gripper BVH ranking scalar `-0.000121945mm` /
    OVERLAP; TCP/tangent/gap/height gates passed. The scalar is not an EPA
    penetration depth or exact near-miss magnitude.
  - Pre-physics contract PASS, candidate gate FAIL, sim counter `0->0`,
    `physics_licensed=false`, controlled physics steps `0`. No baseline or
    target-settle artifacts exist, as required.
  - Visualization DoD passed: one inspected decision PNG, six marker frames,
    and one non-empty `2,480,172`-byte RRD.
- D334 causal truth remains: the old D325 command overlaps the actual gripper
  raw mesh; ownership is clean 1:1. Gripper cook parity `1.46%` FAIL is a
  separate secondary representation finding.

## Interpretation

- D335 licenses a **finite executed-set** conclusion, not a proof that every
  continuous r/t value is infeasible. Do not overclaim mathematical
  impossibility from the sampled grid.
- Do not repeat offset-only tuning, approach, or 10-trial runs in the current
  HOME-seeded position-only family. No target was selected.
- Do not expand beyond the anti-retreat boundary `r<=17mm` without explicitly
  redefining grasp-depth/bracketing semantics; clearance by moving away from
  the cylinder is not a G0a repair.
- Collision-representation repair cannot replace an actual raw-tool-clear
  command. It remains reserve, not the current critical-path fix.
- D330 swept results remain scene-confounded and unreattributed.

## Active Case: G0a

- Object: cylinder radius `0.017m`, height `0.090m`, fixed `(0.300,0.000)`;
  mass placeholder `0.72kg` (real mass required before G0b).
- Friction `static=1.5`, `dynamic=1.2`; no material/mass tuning.
- Old target D325 `(radial,tangent)=(7,11)mm` is collision-invalid. D335 produced
  no replacement target in the registered non-retreat scalar-offset set.
- G0a gates remain: TCP `<=5mm`, tangent `<=15deg`, jaw gap `[0,5mm]`, no
  penetration, contact `>=15mm` below top, displacement `<5mm`, `10/10`.
- Latest runtime output: `claudedocs/runtime_logs/grasp_track/g0a_d335/`.
- Latest detailed session:
  `claudedocs/session_20260712_grasp_g0a_d335_target_family_repair.md`.

## Next Concrete Action

**STOP for user case choice. Recommended active candidate: D336 finite-grid
caveat discriminator** — no new physical variable; use the same audited raw
tool geometry, frozen HOME-seeded r/t family, anti-retreat/alignment gates, and
zero-step pre-physics contract, but apply a separately pre-registered
continuous/finer feasibility method around the top basins. It must be capable
of finding a `>=+0.1mm` raw-clear candidate; otherwise it stops without physics.

Reserve choices (not active without user approval):

1. If finer feasibility also fails, add exactly one reachable
   wrist/tool-orientation variable; do not resurrect the unreachable D323
   strict-axis family.
2. Explicitly redefine grasp-depth semantics before permitting `r>17mm`.
3. Gripper collision-representation repair stays deferred until actual
   raw-clear target feasibility exists.

All future ideas are recorded in `claudedocs/BACKLOG.md`; none is implemented
automatically. `g0a_pass=false`; G0b/RL/ladder promotion remain blocked.

## Must Read First

1. `AGENTS.md` (then `CLAUDE.md` only for Claude-specific workflow)
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D334-D335)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260712_grasp_g0a_d335_target_family_repair.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d335/g0a_d335_target_family_repair_summary.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_candidate_scan.csv`
9. `sim_scripts/cyl34_top_view_d335_grasp_g0a_target_family_repair.py`
10. `claudedocs/session_20260712_grasp_g0a_d334_live_collision_shape_ownership_audit.md`
11. `claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale. Memory is an index, not evidence.
- Git commit/push only on explicit user request.
- B200/JHPark/SSH/pull/.ssh and `/half-clone` remain forbidden.
- Variable Ladder: one or two new variables per case; future ideas go to
  `claudedocs/BACKLOG.md`; grasp outputs only under
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`; paths forward-only.
- Unmeasured estimates are not specs. Unrecorded pilots are not evidence.
- Distinguish raw mesh, mathematical hull, mirror cook, and live-volume-gated
  cook. AABB-only reasoning is forbidden.
- Visualization DoD and Isaac pins remain binding: `numpy==1.26.0`,
  `psutil==5.9.8` after any install.

## Frozen Background

- Professor direction: finish one graspable cylinder case first, then
  G0b -> G1a grid -> G1b standalone PPO
  (`claudedocs/direction_20260708_grasp_pivot.md:3-32`).
- Tap track frozen at D321: `1920/2000` accepted (`96.0%`).
- G0b prerequisites: real cylinder mass measurement and BACKLOG
  `tool_surface_union`.
