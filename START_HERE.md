# START_HERE.md

Last updated: 2026-05-13 KST

This is the rolling project dashboard. It is overwritten as the project moves.
Do not use it as the full experiment history. Durable lessons live in
`claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

## Current Truth

Latest verified state:

- `claudedocs/session_20260517_pathD_v2_clean_rate_fail.md`

Important: the filename is a continuation-session label. Use the file content and
local metrics as source of truth, not the filename date alone.

## Current Status

RoArm + Isaac Lab sponge stacking research is in the post-Path-D-failure stage.
B200 remains the headless state-only learning/eval machine. Rendering is for
later replay/extraction unless explicitly testing render capability.

Path D v2 result:

- Nominal success: `175/256 = 68.36%`
- CLEAN success, `gripper_q_at_success < 0.4 rad`: `24/256 = 9.38%`
- DIRTY/counter-path: `151/256 = 58.98%`
- Counter-path inflation: `7.3x`
- Decision: Path D FAIL under the user-specified `<10% CLEAN` gate

## Current Direction

Pivot from RL-rollout-derived release BC to procedural/SkillGen/MimicGen-style
clean release demonstrations.

Recommended next branch:

1. Generate procedural release-only demos.
2. Use IK/pregrasp init near target, scripted gripper-open command, gravity settle.
3. Filter by direct CLEAN success, not nominal stage4 alone.
4. Train release BC on clean procedural demos.
5. Evaluate with CLEAN/DIRTY split and `gripper_q_at_success`.

## Must Read First

1. `claudedocs/DECISIONS.md`
2. `claudedocs/EXPERIMENT_LEDGER.md`
3. `claudedocs/session_20260517_pathD_v2_clean_rate_fail.md`
4. `claudedocs/session_20260514_evening_rpl_sweep_fail_pathD_entry.md`
5. `claudedocs/path_d_design_20260514.md`

## Source Files To Verify Before Coding

- `roarm_rl/eval_release_bc.py`
- `roarm_rl/gen_release_demos_from_rollout.py`
- `roarm_rl/train_release_bc.py`
- `launch_pathD_eval_bc.sh`
- `claudedocs/pathD_data/analyze_eval_v2.py`
- `claudedocs/pathD_data/eval_metrics_v2.pt`

## Do Not Trust As Current State

- `HANDOFF.md`: March-era handoff, stale for current Isaac Lab Path D/P6 work.
- `TASKS.md`: February-era task list, stale for current Isaac Lab Path D/P6 work.
- Path D v1 `68.36%` nominal success without CLEAN split.
- Any memory-only metric that is not verified from referenced logs/data.

## Context Safety Rule

If the active chat context is approaching 95%, stop new implementation work.
Before continuing, update:

1. `START_HERE.md`
2. `claudedocs/EXPERIMENT_LEDGER.md`
3. `claudedocs/DECISIONS.md` only if a durable lesson changed
4. a new `claudedocs/session_YYYYMMDD_short_title.md`

Then start a new session from this file.
