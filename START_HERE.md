# START_HERE.md

Last updated: 2026-05-14 PM KST (post-δ.5 contact diagnostic)

This is the rolling project dashboard. It is overwritten as the project moves.
Do not use it as the full experiment history. Durable lessons live in
`claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

## Current Truth

Latest verified state:

- `claudedocs/session_20260514_alpha_prime_delta_topdown.md`
  - Original sections: `(alpha') basin sweep PASS`, `(delta) top-down chain partial`,
    `(delta.2) gripper -10° BLOCKED by URDF clamp`.
  - APPENDIX (5/14 PM): `(δ.2) doc errata` correcting L138-139 Skill 0/1a step counts
    (actual 200/150 step max, not 21/14) — append-only.
  - Section (5/14 PM): `(δ.4) Result` — Skill 1b multi-stage diagnostic.
    Stall localized to TCP z ≈ +51.8mm INVARIANT under target depth (D008).
  - New section (5/14 PM): `(δ.5) Result` — contact geometry diagnostic.
    Settled sponge top ≈ +47.0mm; gripper_link mesh min-z ≈ +47.4mm at stall
    (D009). Root cause now top-contact/collision geometry.

Active pivot: **Hierarchical chain skills with P6v14a as learned release sub-skill**.
(D003 contamination avoided — P6v14a is used as a primitive, NOT as BC training source.)

Reserve pivot (still valid, from prior Path D failure analysis):
Procedural release-only demos → train release BC → CLEAN eval.

## Current Status

RoArm + Isaac Lab sponge stacking research. B200 = headless state-only learning/eval
(D004). Rendering off by default (HARD RULE #17/#26).

### Path D (chronologically prior pivot — FAILED)

- Nominal: `175/256 = 68.36%`
- CLEAN `gripper_q@success < 0.4 rad`: `24/256 = 9.38%`
- DIRTY/counter-path: `151/256 = 58.98%`
- Counter-path inflation: `7.3x`
- Decision: Path D FAIL under `<10% CLEAN` gate. Don't repeat BC-on-RL-rollout (D003).

### Hierarchical chain (current pivot — top-contact localized, geometry fix next)

`roarm_rl/chain_skills.py` = scripted Skill 0/1/2/4 (IK + force-set per
D007) + LEARNED Skill 3 (P6v14a/model_499.pt release primitive).

Run-by-run summary:

- 5/13: 7-iter B200 debugging. Skill 0 PD limit cycle fixed via `robot_dof_targets`
  force-set (200→21 steps, D007). Skill 1 descent stall first observed
  (sponge -44mm Y push). [`session_20260513_chain_skills_hierarchical_pivot.md`]
- 5/14 (alpha'): P6v14a basin sweep 6-point grid (dx 0-45mm, dz 0-20mm).
  release_step=13 consistent 5/6 runs. P6v14a release primitive portable
  within 30-45mm xy / +20mm z.
- 5/14 (delta): Top-down chain v1 (q_high TCP +150mm, GRIPPER_OPEN_DEG=0).
  Lateral knock 7× improved (Y -44→-6mm). BUT vertical descent stall: tcp_after1
  z = +51.9mm vs target +33mm (= 19mm gap). `grasped=False`,
  `CHAIN_FINAL_SUCCESS=NO`.
- 5/14 (delta.2): GRIPPER_OPEN_DEG `0 → -10` widen-jaw test. URDF
  `lower="0"` clamped target. Physics bit-identical to (delta.1). Also (per
  errata APPENDIX): max_full_err was gripper-inflated to ~10°, so Skill 0/1a
  forced to 200/150 step max (NOT the 21/14 originally written in L138-139).
  Decision: D006 (do not modify URDF; future scripted skills use
  `GRIPPER_OPEN_DEG = 0.0` and `exclude_gripper=True`).
- 5/14 (delta.4): Skill 1b multi-stage z descent diagnostic
  (hover→+50→+40→+33mm) + `exclude_gripper=True` + revert to `GRIPPER_OPEN_DEG=0`.
  - Skill 0 / 1a early-break restored (21 / 14 steps) — exclude_gripper fix verified.
  - Skill 1b1 target +50mm → TCP +51.79mm, arm_err 0.32°.
  - Skill 1b2 target +40mm → TCP +51.72mm, arm_err 2.96° (stall confirmed).
  - Skill 1b3 target +33mm → TCP +51.80mm, arm_err 4.90° (stall confirmed).
  - TCP z INVARIANT at ≈ +51.8mm across all targets and across 5/13 + (δ.1) +
    (δ.2) + (δ.4) sessions. xy precision 0.6-0.8mm. Sponge xy unmoved across
    sub-stages → pure z contact equilibrium.
  - Conclusion (D008): contact equilibrium at +51.8mm; sub-stage splitting and
    step-ramping CANNOT bypass it. Fix is in geometry/contact space.
- 5/14 (delta.5): 5-D/5-E/5-F contact geometry diagnostic.
  - Initial write: sponge root z +11.4mm OK, but after Skill 0/1a settling root
    z is +23.5mm → actual sponge top +47.0mm.
  - At Skill 1b stall, TCP stays +51.72~+51.80mm while gripper_link mesh min-z
    is +47.4mm, only +0.4mm above sponge top. XY AABB overlap ≈ 60mm × 22mm.
  - `_grasped=False`, `_was_grasped=False` through Skill 1b, so not latch artifact.
  - Conclusion (D009): current top-down pose is conceptually right, but collision
    mesh contacts the sponge top before TCP can reach +33mm grasp. Fix geometry
    before four-sponge stacking demo generation.

## Current Direction

(δ.6) top-down pick geometry fix before four-sponge well/hash stacking:

1. **G1 — Quick geometry patch (recommended)**: raise/redefine `TCP_GRASP_Z`,
   shift TCP relative to sponge center, or rotate wrist/gripper so gripper_link
   collision mesh clears the settled sponge top while closing around the 22mm width.
2. Re-run B200 chain with the existing 5-D/5-E diagnostics. PASS only if Skill 1b
   reaches grasp pose without top stall, `grasped=True`, and `CHAIN_FINAL_SUCCESS=YES`.
3. Only after pick primitive succeeds: resume four-sponge well/hash workspace
   planning/rendering (white table, black robot, gray background, pink edge-stand
   sponges) and sequential pick/place/stack demo generation.

Then branch on outcome:
- geometry patch succeeds → proceed to four-sponge source layout and stacking demos.
- geometry patch cannot avoid top contact → inspect/modify gripper collision mesh
  or train (γ) PPO Skill 1 primitive.

Reserve: (γ) PPO Skill 1 descend+grasp primitive (1-2 weeks, paper-quality).
Procedural release-only demos → release BC (Codex prior).

## Must Read First

1. `claudedocs/DECISIONS.md` D006, D007, D008, **D009 (new 5/14 PM)**
2. `claudedocs/EXPERIMENT_LEDGER.md` — 2026-05-14 (α'), (δ), (δ.2), (δ.4), (δ.5) rows
3. `claudedocs/session_20260514_alpha_prime_delta_topdown.md` — full (δ.4)/(δ.5)
   sections + APPENDIX errata for (δ.2) L138-139
4. `claudedocs/session_20260513_chain_skills_hierarchical_pivot.md` (architecture
   origin, force-set discovery, sponge collision)
5. `roarm_rl/chain_skills.py` (current code — multi-stage Skill 1b + 5-D/5-E diagnostics)
6. `roarm_rl/roarm_stack_env.py` L408-412 (`_pre_physics_step` clamp — D006/D007 anchor)
7. `assets/roarm_m3/urdf/roarm_m3.urdf` (`link5_to_gripper_link` joint — D006 + 5-F target)

## Source Files To Verify Before Coding

- `roarm_rl/chain_skills.py` — md5 `03169d005c4d39fa10583047e8957961` (Lenovo + B200)
- `launch_chain_topdown.sh` — md5 `2d52c0efd2ca1d5bab78f3e029185a47` (Lenovo + B200)
- `launch_basin_sweep.sh` — md5 `a361aa673f82a663be8cd39ff4d0dee6` (Lenovo + B200, unchanged)
- `roarm_rl/roarm_stack_env.py` (env physics — `_pre_physics_step` L408-412)
- `assets/roarm_m3/urdf/roarm_m3.urdf` (gripper + finger joint limits — D006 + 5-F)
- `logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt` (Skill 3 source on B200)
- B200 `/tmp/chain_topdown_v4.{out,err}` ((δ.5) contact geometry diagnostic)

## Do Not Trust As Current State

- `HANDOFF.md`: March-era handoff, stale.
- `TASKS.md`: February-era task list, stale.
- Path D v1 `68.36%` nominal success without CLEAN split.
- Memory-only metrics that are not verified from referenced logs/data.
- (δ.2) result table L138-139 in `session_20260514_alpha_prime_delta_topdown.md`
  Skill 0 "21 step" / Skill 1a "14 step" — INCORRECT, actual 200/150 step max
  per APPENDIX errata.
- "(δ.3) full ramping" as a viable next experiment — D008 predicts same failure.
- "more sub-stages (5/6/N)" as a viable next experiment — D008 same prediction.
- Four-sponge stacking demos before top-down pick succeeds — D009 says pick
  primitive currently stalls on gripper-link top contact.

## Active Reserve Decision (HARD RULE #26 territory)

B200 is currently locked-in for physics-only state-only RL (Annotator/RTX bypass)
through 2026-05-19. Hierarchical chain experiments run in this regime. After
5/19, HARD RULE #22/#24/#25 (4-axis matrix / v7 collection / 3-track parallel)
may resume. See memory topic file
`project_b200_physics_only_rl_priority_20260507_night.md`.

## Context Safety Rule

If the active chat context is approaching 95%, stop new implementation work.
Before continuing, update:

1. `START_HERE.md` (this file — overwrite with current truth)
2. `claudedocs/EXPERIMENT_LEDGER.md` (append new experiment rows)
3. `claudedocs/DECISIONS.md` (append new durable lessons only)
4. A new `claudedocs/session_YYYYMMDD_short_title.md` (append-only detail log) —
   or append-only into the existing day's session doc if same-day continuation.

Then start a new session using the boot prompt in `CLAUDE.md` `## Current-State Protocol`.
Do NOT use `/half-clone` or `/handoff` (HARD RULE #11).
