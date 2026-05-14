# START_HERE.md

Last updated: 2026-05-14 PM KST

This is the rolling project dashboard. It is overwritten as the project moves.
Do not use it as the full experiment history. Durable lessons live in
`claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

## Current Truth

Latest verified state:

- `claudedocs/session_20260514_alpha_prime_delta_topdown.md` (5/14 PM update)
  - Sections: `(alpha') basin sweep PASS`, `(delta) top-down chain partial`,
    `(delta.2) gripper -10° BLOCKED by URDF clamp`.

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

### Hierarchical chain (current pivot — IN PROGRESS)

`roarm_rl/chain_skills.py` = scripted Skill 0/1/2/4 (IK + force-set per D007) +
LEARNED Skill 3 (P6v14a/model_499.pt release primitive).

Run-by-run summary (`claudedocs/session_20260513_chain_skills_hierarchical_pivot.md`
and `claudedocs/session_20260514_alpha_prime_delta_topdown.md`):

- 5/13: 7-iter B200 debugging. Skill 0 PD limit cycle fixed via `robot_dof_targets`
  force-set (200→21 steps, D007). Skill 1 descent stall first observed
  (sponge -44mm Y push).
- 5/14 (alpha'): P6v14a basin sweep 6-point grid (dx 0-45mm, dz 0-20mm) on
  release primitive. release_step=13 consistent 5/6 runs. Post-release sponge
  knock-away → Skill 3 must early-terminate at release+buffer steps. ✓ portable
  primitive confirmed (30-45mm xy / +20mm z).
- 5/14 (delta): Top-down chain v1 (TCP +150mm clearance, GRIPPER_OPEN_DEG=0).
  Lateral knock 7× improved (Y -44→-6mm). BUT vertical descent stall persists:
  tcp_after1 z = +51.9mm vs target +33mm (= 19mm gap, bit-identical to 5/13).
  `grasped=False`, `CHAIN_FINAL_SUCCESS=NO`.
- 5/14 (delta.2): GRIPPER_OPEN_DEG `0 → -10` to widen jaw past 22mm sponge
  width. **TEST INVALID**: URDF gripper `lower="0"` clamps target. Gripper
  actual `-0.0°` identical to delta.1. Descent stall bit-identical.
  Hypothesis untestable in current sim. Decision: D006 (do not modify URDF).

## Current Direction

Skill 1b descent stall fix (gripper-state independent per delta.2):

1. **(delta.4) NEW — RECOMMENDED**: Skill 1b multi-stage z-target
   (+63 → +50 → +40 → +33mm). ETA ~2hr. MEDIUM confidence.
2. **(delta.3) FULL**: Multi-stage target ramping (`target = current + delta_step`
   instead of one-shot q_grasp). ETA half day. HIGH confidence.
3. **(gamma)**: PPO Skill 1 descend+grasp primitive. ETA 1-2 weeks. paper-quality.
4. **(beta)**: Sim physics tuning (effort_limit↑, friction). ETA 1-2 days. LOW
   confidence — geometry, not physics.

Recommended sequence: (delta.4) → if FAIL → (delta.3) → if FAIL → (gamma) or
geometry deep-dive.

## Must Read First

1. `claudedocs/DECISIONS.md` (especially D006, D007 — new lessons from 5/14)
2. `claudedocs/EXPERIMENT_LEDGER.md` (all entries, especially 2026-05-13 late
   onward for current hierarchical pivot context)
3. `claudedocs/session_20260514_alpha_prime_delta_topdown.md` (current truth)
4. `claudedocs/session_20260513_chain_skills_hierarchical_pivot.md` (architecture
   origin, force-set discovery, sponge collision)
5. `roarm_rl/chain_skills.py` (current code)
6. `roarm_rl/roarm_stack_env.py` L408-412 (`_pre_physics_step` clamp — D006/D007 anchor)

## Source Files To Verify Before Coding

- `roarm_rl/chain_skills.py` — md5 `6d7c06623b12c9ad005174c3851164c5` (Lenovo + B200)
- `launch_chain_topdown.sh` — md5 `a5208ac9c0530bc40c3bfff85c8118be` (Lenovo + B200)
- `launch_basin_sweep.sh` — md5 `a361aa673f82a663be8cd39ff4d0dee6` (Lenovo + B200)
- `roarm_rl/roarm_stack_env.py` (env physics — `_pre_physics_step` L408-412)
- `assets/roarm_m3/urdf/roarm_m3.urdf` (gripper joint limit — D006 source)
- `logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt` (Skill 3 source on B200)

## Do Not Trust As Current State

- `HANDOFF.md`: March-era handoff, stale.
- `TASKS.md`: February-era task list, stale.
- Path D v1 `68.36%` nominal success without CLEAN split.
- Memory-only metrics that are not verified from referenced logs/data.
- `chain_skills.py` "696 LOC" claim from earlier continuation prompts — actual
  is 908 LOC after 5/14 work.

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
4. A new `claudedocs/session_YYYYMMDD_short_title.md` (append-only detail log)

Then start a new session using the boot prompt in `CLAUDE.md` `## Current-State Protocol`.
