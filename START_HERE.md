# START_HERE.md

Last updated: 2026-07-13 KST (D337 current truth: the q5 gripper convention
error is repaired — the target family is feasible again with the jaw open
(2,560/2,629 raw-clear); the sole remaining G0a blocker is the cooked-hull
collision representation.)

## Current Truth

- Active pivot remains **grasp track G0a on cylinder D34 x H90**. Cube repair,
  G0b close/lift, PPO/RL, VLA, randomization, real RoArm, and B200 remain out
  of scope.
- D337 verdict: `D337_G0A_STATIC_RUNTIME_MIXED_STOP` — with a fully resolved
  causal story:
  - **q5 convention error (durable fact)**: URDF gripper `q5=0` = CLOSED,
    `1.571rad` = OPEN (D322 mapping `real 88.3deg <-> 1.571rad`). The
    D325-family "open gripper q5=0" was wrong; D330-D336 all wrote a closed
    moving jaw into the grasp volume. D334-D336 conclusions stand but are
    scoped to the closed-jaw sub-family.
  - With `q5=1.5413rad` (98.1% of open limit, ~86.6deg real): open-jaw scan
    passes `2,560/2,629`; selected target = **the original D325 `(7,11)mm`**
    (link5 `+4.2726mm` / gripper `+11.1751mm` raw-clear); all controls
    bit-exact; design scoping validated on the live stage (`+11.175088` vs
    predicted `+11.175mm`).
  - Conditional 200-step settle: raw meshes clear at every reading (min
    `+7.498/+9.595mm`), gripper `0N`, final alignment PASS, final
    displacement `2.754mm < 5mm` — but link5 hit a `38.861N` step-0 impulse
    and the object was disturbed (max XY `5.418mm`, tilt `4.208deg`) before
    resting against link5 at `~1.70N`. Attribution-timing gate failed on an
    onset-metric limitation (recorded onset `19` missed the step-0 impulse
    row) -> MIXED, no verdict override.
  - **Causal attribution (evidence)**: physics collides with the cooked
    convex hulls, not the raw meshes — D334 certified link5's cooked hull at
    `-6.2367mm` overlap at exactly this pose. The moving-jaw and target
    problems are solved; collision representation is the one blocker left.
- **USD/URDF divergence (durable fact)**: the robot USD (5/13) embeds full
  `gripper_link.stl` as the moving-jaw collision mesh; the URDF was changed
  5/14 to a 4mm-box proxy (`g2a`). The USD is stale vs the URDF but more
  physical; it remains the audited truth. Any regeneration must decide the
  moving-jaw representation explicitly.

## Active Case: G0a

- Object: cylinder r `0.017m`, h `0.090m`, fixed `(0.300,0.000)`; mass
  placeholder `0.72kg` (real mass required before G0b); friction `1.5/1.2`.
- Feasible open-jaw target family exists: `position_only_tangent_minus1` +
  `q5=1.5413rad`; canonical candidate `(7,11)mm`.
- G0a gates remain: TCP `<=5mm`, tangent `<=15deg`, jaw gap `[0,5mm]`, no
  penetration, contact `>=15mm` below top, displacement `<5mm`, `10/10`.
- Latest runtime output: `claudedocs/runtime_logs/grasp_track/g0a_d337/`
  (incl. `design_scoping/` and the first full-trajectory 200-step RRD).
- Latest detailed session:
  `claudedocs/session_20260713_grasp_g0a_d337_open_jaw_target_gate.md`.

## Next Concrete Action

**STOP for user case choice. Recommended: D338 collision-representation
repair** — the D334/D335/D336 deferral condition ("wait until a raw-clear
target exists") is now satisfied. Scope: regenerate/replace the robot
collision representation so the cooked hulls match the raw meshes within a
registered tolerance (primary offender: link5 cooked `-6.2367mm` at the
canonical pose; secondary: gripper cook parity `1.46%` FAIL, and the
g2a-vs-full-mesh moving-jaw decision). This changes a collision asset =
explicit user approval + its own pre-registered case with the D337 controls
as parity anchors. After it passes, re-run the frozen `(7,11)` open-jaw
settle, then the 10-trial alignment gate (G0a 본 게이트).

Reserve choices:

1. Onset-metric hardening (record impulse-row onsets) — REACTIVE fix allowed
   only as part of the next settle case, not standalone.
2. `r>17mm` grasp-depth redefinition — unchanged, not needed now.

All future ideas go to `claudedocs/BACKLOG.md`. `g0a_pass=false`;
G0b/RL/ladder promotion remain blocked.

## Must Read First

1. `AGENTS.md` (then `CLAUDE.md` only for Claude-specific workflow)
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D334-D337)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260713_grasp_g0a_d337_open_jaw_target_gate.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d337/g0a_d337_open_jaw_target_gate_summary.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d337/design_scoping/d337_design_scoping_results.md`
9. `claudedocs/session_20260712_grasp_g0a_d336_finite_grid_caveat_discriminator.md`
10. `claudedocs/session_20260712_d336_posthoc_setup_rerun_plan_audit.md`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale. Memory is an index, not evidence.
- Git commit/push only on explicit user request.
- B200/JHPark/SSH/pull/.ssh and `/half-clone` remain forbidden.
- Variable Ladder: one or two new variables per case; grasp outputs only
  under `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`; forward-only.
- **q5 convention: `q5=0` = CLOSED; sim "open" = `~1.541-1.571rad`** (D337).
- BVH distance scalars of colliding meshes are ranking-invalid (D336); use
  contact-level EPA enumeration. AABB-only reasoning is forbidden.
  Distinguish raw mesh, mathematical hull, mirror cook, live cook.
- Visualization DoD and Isaac pins binding: `numpy==1.26.0`, `psutil==5.9.8`.

## Frozen Background

- Professor direction: finish one graspable cylinder case first, then
  G0b -> G1a grid -> G1b standalone PPO
  (`claudedocs/direction_20260708_grasp_pivot.md:3-32`).
- Tap track frozen at D321: `1920/2000` accepted (`96.0%`).
- G0b prerequisites: real cylinder mass measurement and BACKLOG
  `tool_surface_union` (now folded into the D338 representation decision).
