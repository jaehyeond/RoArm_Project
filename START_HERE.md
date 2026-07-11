# START_HERE.md

Last updated: 2026-07-11 KST (D333 current truth: sole-support repair passed,
but the frozen correct-cylinder final pose still causes immediate
gripper_link-attributed contact and object disturbance. Body/collision-shape
ownership remains mixed.)

## Current Truth

- Active pivot is **grasp track G0a on cylinder D34 x H90**. Cube repairs,
  G0b close/lift, PPO/RL, VLA, randomization, RoArm real, and B200 remain out of
  scope.
- D333 verdict:
  `D333_G0A_CLEAN_STATIC_BODY_ATTRIBUTION_MIXED_STOP`.
  - Only new physical variable:
    `[support_domain_global_ground_collision_disabled]`.
  - Exact global-ground collider changed enabled to disabled before PLAY and
    remained disabled. TapTable collider stayed enabled; top error was
    `0.000000297mm`; robot was fixed-base with zero root drift.
  - Cylinder-owned ContactSensor passed its one-body/four-filter contract and
    resolved TapTable/link4/link5/gripper_link one-to-one. Reporter and sleep
    thresholds were zero.
  - Sole-support baseline passed every gate: first z correction
    `0.000003354mm`, last-50 TapTable Fz `7.063635349N`, tail bottom/table gap
    `0.000134554mm`, max XY/tilt `0.003773945mm/0.003364521deg`, and robot
    filters `0N`.
  - Clean target still disturbed the cylinder immediately: gripper_link onset
    `0`, peak `76.412754919N`; link4/link5 onset `-1/-1`; object disturbance
    onset `0`; max/final XY `12.598178941/9.298849201mm`; max/final tilt
    `8.074518/3.881523deg`.
  - Gripper force persisted in `180/200` rows. Filtered-force sum closed the net
    force within `6.7417e-6N`; step-0 gripper force/object-displacement XY
    cosine was `0.999981`.
  - G0a PASS is false, ladder promotion is false, collision repair is not
    authorized, and D330 swept approach is not reattributed.
- Visualization DoD passed: three inspected PNGs, Isaac markers, and one
  non-empty 200-step RRD.

## Interpretation

- D332's `12.117mm` global-ground embedding was real and is now removed. It was
  not the sole cause of the D332 gripper event or object motion: target
  first-step z changed `12.707490 -> 0.480719mm`, but gripper peak remained
  `66.866266 -> 76.412755N` and final XY remained
  `10.282285 -> 9.298849mm`.
- The prior `pop-into-gripper only` explanation is refuted.
- D332's default mirror-recooked link5 hull still overlaps by `6.236272mm`, but
  clean runtime sampled gripper_link rather than link5. The link5 gap-fill
  hypothesis is downgraded to an unresolved mirror/live-cook/rigid-body-owner
  mismatch. Do not rewrite collision or change the target yet.
- Link5 `0N` is not a complete no-contact proof because that filter lacks an
  independent positive control. Conversely, the gripper_link attribution is
  strong at the rigid-body ContactSensor level; it is not exact collision-shape
  identity.
- D330 remains a confounded swept run. Its executed `0/10` and XY displacement
  are historical facts, but its target-height and causal interpretation are not
  a clean cylinder result. D333 only resolves the frozen final-pose question.
- Correct-cylinder substitution alone is not sufficient. Clean static geometry
  must be repaired before any approach/10-trial rerun can change a decision.

## Active Case: G0a

- Object: cylinder radius `0.017m`, height `0.090m`, fixed position
  `(0.300,0.000)`; mass remains the `0.72kg` placeholder. Real mass is
  unmeasured and must be measured before G0b.
- Friction remains `static=1.5`, `dynamic=1.2`; no material/mass tuning.
- Pose family remains D325 `position_only_tangent_minus1`; open gripper;
  tangent offset `11mm`, radial offset `7mm`, alignment standoff `2mm`, TCP z
  at reset cylinder center. Future grasp flush formula remains `D/2-8mm`.
- G0a gates remain TCP `<=5mm`, tangent `<=15deg`, fixed-jaw gap `[0,5mm]`, no
  penetration, contact point at least `15mm` below top, displacement `<5mm`,
  and `10/10`. D333 is a discriminator, not a gate promotion.
- Latest runtime output:
  `claudedocs/runtime_logs/grasp_track/g0a_d333/`.
- Latest detailed session:
  `claudedocs/session_20260711_grasp_g0a_d333_sole_support_static_retest.md`.

## Next Concrete Action

D334 is one decision-changing, frozen-pose collision-asset audit with no new
physical variable:

1. At the D333 frozen pre-step command pose and recorded post-step-0 pose,
   enumerate link5 and gripper_link live collision prims and their nearest
   rigid-body owners.
2. Record source mesh, approximation/cook attributes, and default-cooked shape
   parity for both bodies.
3. Compute non-AABB signed distance from each relevant raw/cooked collision
   shape to the cylinder and map the recorded D333 gripper contact point to the
   candidate shapes and actual tool surface.
4. Stop for user choice:
   - proxy/cook artifact -> collision-representation repair candidate;
   - actual tool geometry overlap -> target-family repair candidate;
   - unresolved parity -> `MIXED_STOP`.

Do not run a clean D330 sweep/10-trial yet: the frozen static target already
fails, so it cannot change the current repair choice. D334 must not rewrite a
mesh, tune a target, run ownership search beyond link5/gripper_link, or advance
the ladder in the same session.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D331-D333)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260711_grasp_g0a_d333_sole_support_static_retest.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d333/g0a_d333_sole_support_static_summary.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_postrun_csv_reanalysis.json`
9. `sim_scripts/cyl34_top_view_d333_grasp_g0a_sole_support_static_retest.py`
10. `claudedocs/session_20260711_grasp_g0a_d332_static_collision_discriminator.md`
11. `claudedocs/session_20260711_d332_verification_briefing_d333_prechecks.md`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale. Memory is an index, not evidence.
- Existing dirty/untracked/ahead state must not be reverted. Git commit/push
  only on explicit user request.
- B200/JHPark/SSH/pull/.ssh and `/half-clone` remain forbidden.
- Variable Ladder: one or two new variables per case; future ideas go to
  `claudedocs/BACKLOG.md`; grasp outputs only under
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`; paths are forward-only.
- Use an object class consistent with the ladder target. Do not reintroduce the
  cube without explicit user approval.
- Unmeasured estimates are not specs. Unrecorded pilots are not decision
  evidence until reproduced as artifacts.
- Distinguish raw mesh, mathematical hull, mirror recook, and directly
  extracted live collision. ContactSensor initialization/net closure does not
  prove every body-filter negative channel or exact collision shape.
- Scene/support contract and body/shape attribution must pass before a
  body-specific repair.
- Visualization DoD and Isaac package pins remain binding:
  `numpy==1.26.0`, `psutil==5.9.8` after any install.

## Frozen Background

- Professor direction remains: finish one graspable cylinder case first, then
  G0b -> G1a grid -> G1b standalone PPO. The original lab-meeting instructions
  are `claudedocs/direction_20260708_grasp_pivot.md:3-32`; later sections are
  project decisions, not direct professor quotes.
- Tap track is frozen at D321: `1920/2000` accepted (`96.0%`).
- G0b prerequisites remain real cylinder mass measurement and BACKLOG
  `tool_surface_union`; current 4mm moving-jaw collision proxy cannot establish
  grasp physics fidelity.
