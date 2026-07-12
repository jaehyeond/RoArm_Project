# START_HERE.md

Last updated: 2026-07-12 KST (D336 current truth: the finite-grid caveat on
D335 is discharged — the position-only r/t target family penetrates the
cylinder at millimeter scale at every evaluated point, including the
continuous optimum; zero physics steps ran.)

## Current Truth

- Active pivot remains **grasp track G0a on cylinder D34 x H90**. Cube repair,
  G0b close/lift, PPO/RL, VLA, randomization, real RoArm, and B200 remain out
  of scope.
- D336 verdict: `D336_G0A_FINITE_GRID_CAVEAT_DISCHARGED_NO_CLEAR_STOP`.
  - Zero new physical variables; method-only change (exact contact-level EPA
    ranking + continuous refinement inside the frozen D335 family/domain).
  - Controls bit-exact: old target link5 `+4.2726455336mm`/CLEAR, gripper
    `-5.9566769497mm`/OVERLAP, deltas `0.000000mm` vs D334; grid parity vs
    the pinned D335 CSV `0.000000mm`; exact layer `6.4606mm >= 5.8630mm`.
  - `3,181` unique evaluations (2,629 exact rescore + 322 Nelder-Mead + 230
    micro-grid): raw-clear `0`, full-pass `0`. Sim counter `0 -> 0`,
    `physics_licensed=false`, controlled physics steps `0`.
  - **Ranking-bias finding**: D335's BVH scalar was not a proximity measure.
    D335 best `(14.6,13.9)mm` (scalar `-0.000122mm`) is actually `-7.830mm`
    deep by certified EPA. True best basin: `(15.3897,9.0000)mm` at
    `-4.285mm` (alignment-passing best `-4.396mm` at `(15.2774,9.0446)mm`);
    worst `-11.299mm`. EPA 64-contact cap saturated everywhere → depths are
    lower bounds of solid penetration.
- D334 causal truth stands: the actual gripper raw mesh overlaps the
  cylinder under this family; ownership is clean 1:1. Gripper cook parity
  `1.46%` FAIL remains a separate secondary finding.

## Interpretation

- The HOME-seeded position-only radial/tangent family is **closed at
  millimeter scale**: no offset-only tuning, finer grid, or optimizer pass in
  this family may be attempted again (D336). A `>=4.29mm` certified
  penetration at the continuous optimum cannot become `+0.1mm` clearance by
  position offsets.
- Still an executed-set statement, not a continuum impossibility proof — but
  the "maybe the grid missed a pocket" branch is gone.
- The BVH distance scalar of a colliding mesh must never be used as a
  near-miss/proximity measure; ranking uses contact-level EPA enumeration
  (D336 method), judgment stays the exact clear rule.
- Collision-representation repair cannot replace a raw-tool-clear command; it
  remains reserve.

## Active Case: G0a

- Object: cylinder radius `0.017m`, height `0.090m`, fixed `(0.300,0.000)`;
  mass placeholder `0.72kg` (real mass required before G0b).
- Friction `static=1.5`, `dynamic=1.2`; no material/mass tuning.
- Old target D325 `(7,11)mm` is collision-invalid (D334). D335 found no clear
  scalar-offset target; D336 closed the whole position-only family at
  millimeter scale.
- G0a gates remain: TCP `<=5mm`, tangent `<=15deg`, jaw gap `[0,5mm]`, no
  penetration, contact `>=15mm` below top, displacement `<5mm`, `10/10`.
- Latest runtime output: `claudedocs/runtime_logs/grasp_track/g0a_d336/`.
- Latest detailed session:
  `claudedocs/session_20260712_grasp_g0a_d336_finite_grid_caveat_discriminator.md`.

## Next Concrete Action

**STOP for user case choice** (D336 discharged the caveat; no candidate
exists to evaluate):

- **(A) Recommended: add exactly one new reachable wrist/tool-orientation
  variable** to the target family (reuse the same bounded r/t domain,
  anti-retreat `r<=17mm`, alignment gates, and the D336 exact raw-tool
  pre-physics gate). Quantitatively motivated: the jaw region needs
  `>=~4.4mm` effective clearance change that position offsets cannot
  produce. The unreachable D323 strict-axis family must not be resurrected.
- (B) Explicitly redefine grasp-depth semantics before permitting `r>17mm`.
- (C) Gripper collision-representation repair stays deferred until an actual
  raw-clear target family exists.

All future ideas go to `claudedocs/BACKLOG.md`; none is implemented
automatically. `g0a_pass=false`; G0b/RL/ladder promotion remain blocked.

## Must Read First

1. `AGENTS.md` (then `CLAUDE.md` only for Claude-specific workflow)
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D334-D336)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260712_grasp_g0a_d336_finite_grid_caveat_discriminator.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d336/g0a_d336_finite_grid_caveat_summary.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_exact_clearance_map.png`
9. `sim_scripts/cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator.py`
10. `claudedocs/session_20260712_grasp_g0a_d335_target_family_repair.md`
11. `claudedocs/session_20260712_grasp_g0a_d334_live_collision_shape_ownership_audit.md`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale. Memory is an index, not evidence.
- Git commit/push only on explicit user request.
- B200/JHPark/SSH/pull/.ssh and `/half-clone` remain forbidden.
- Variable Ladder: one or two new variables per case; future ideas go to
  `claudedocs/BACKLOG.md`; grasp outputs only under
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`; paths forward-only.
- Unmeasured estimates are not specs. Unrecorded pilots are not evidence.
- Distinguish raw mesh, mathematical hull, mirror cook, and live-volume-gated
  cook. AABB-only reasoning is forbidden. BVH distance scalars of colliding
  meshes are ranking-invalid (D336); use contact-level EPA enumeration.
- Visualization DoD and Isaac pins remain binding: `numpy==1.26.0`,
  `psutil==5.9.8` after any install.

## Frozen Background

- Professor direction: finish one graspable cylinder case first, then
  G0b -> G1a grid -> G1b standalone PPO
  (`claudedocs/direction_20260708_grasp_pivot.md:3-32`).
- Tap track frozen at D321: `1920/2000` accepted (`96.0%`).
- G0b prerequisites: real cylinder mass measurement and BACKLOG
  `tool_surface_union`.
