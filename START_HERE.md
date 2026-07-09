# START_HERE.md

Last updated: 2026-07-09 KST (D322 current truth: grasp track pivot installed; G0a alignment runtime failed 0/10.)

## Current Truth

- Active pivot is now **grasp track G0a**, not cube tap expansion, PPO, VLA, Track A, RoArm deployment, or B200.
- Tap track is frozen at D321 as a useful producer/data-pipeline asset:
  `1920/2000` accepted low/mid script-v2 episodes (`96.0%`), LeRobot append passed, but it is +x-only, not direction-diverse, and not learned-policy success.
- New direction doc:
  `claudedocs/direction_20260708_grasp_pivot.md`.
- D322 prompt source:
  `claudedocs/D322_PROMPT.md`.
- D322 session doc:
  `claudedocs/session_20260709_grasp_g0a_d322_alignment_fail.md`.
- D322 primary runtime output:
  `claudedocs/runtime_logs/grasp_track/g0a_d322/g0a_alignment_summary.json`.
- D322 long-hold diagnostic sidecar:
  `claudedocs/runtime_logs/grasp_track/g0a_d322_longtrack_check/g0a_alignment_summary.json`.

## Active Case: G0a

- New variable: `grasp pose geometry` only: base yaw alignment + asymmetric fixed-jaw/moving-jaw TCP offset.
- Invariants:
  - Existing 10cm cube, mass `0.72kg`.
  - Fixed object position `(x=0.30m, y=0.00m)`.
  - Friction stays `static=1.5`, `dynamic=1.2`.
  - State-only, no render, no RL/PPO, no gripper close, no grasp, no lift.
  - 10cm cube is too wide for the measured practical gripper opening (~40-45mm), so G0a is alignment-only.
- Pre-registered success criteria:
  1. TCP pose error `<=5mm` and base-yaw error `<=3deg`.
  2. Fixed-jaw grasp face to cube face gap `<=3mm` with no penetration.
  3. Cube XY displacement `<5mm`.
  4. All 10 trials pass the same condition.
- Output path: `claudedocs/runtime_logs/grasp_track/g0a_d322/`.

## Latest Result

- D322 primary G0a runtime verdict: `D322_G0A_ALIGNMENT_FAIL`.
- Primary 10-trial result:
  - pass all criteria: `0/10`.
  - failure counts: TCP pose `10/10`, fixed-jaw gap `10/10`, fixed-jaw penetration `10/10`, cube displacement `0/10`, base yaw `0/10`.
  - mean TCP pose error: `96.63mm`.
  - mean fixed-jaw signed face gap: `-56.00mm` (penetration by proxy definition).
  - mean cube XY displacement: `0.019mm`.
  - mean max arm joint tracking error: `0.174rad`.
- Long-hold diagnostic (`500` approach + `500` hold, `12s`) also failed:
  - pass all criteria: `0/10`.
  - mean TCP pose error: `96.49mm`.
  - mean fixed-jaw signed face gap: `-55.78mm`.
  - mean max arm joint tracking error: `0.174rad`.
- Interpretation:
  - Base yaw alignment itself is fine.
  - The cube is not being pushed/moved; this is not a tap-style overshoot failure.
  - The current low side-alignment TCP target is not being reached under the live RoArm actuator/pose contract, or the fixed-jaw/TCP proxy definition is not yet aligned with the USD tool frame.

## Next Concrete Action

Do **not** start G0b, cylinder spawn, gripper close, lift, RL/PPO, render, VLA, RoArm, B200, friction/material changes, or position randomization.

Next session should repair G0a only:

1. Verify actual USD link/jaw frame semantics for TCP, fixed jaw, and moving jaw.
2. Query live link poses for link5 / gripper link / collision proxy at the failed final state.
3. Decide whether the G0a target should be expressed as TCP, EEF, fixed-jaw face, or named tool-surface proxy.
4. Keep the same fixed cube/friction/no-close/no-render setup and rerun the 10-trial alignment criteria.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md`
4. `claudedocs/EXPERIMENT_LEDGER.md`
5. `claudedocs/D322_PROMPT.md`
6. `claudedocs/direction_20260708_grasp_pivot.md`
7. `claudedocs/session_20260709_grasp_g0a_d322_alignment_fail.md`
8. `sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale; do not use them as current truth.
- Memory is only a helper index. Repo docs/logs/code are current truth.
- B200/JHPark/SSH/pull/.ssh copy are forbidden for this branch unless the user explicitly changes the rule.
- Existing dirty/untracked/ahead state must not be reverted.
- Variable Ladder Protocol (D322~):
  - Each case may introduce only one or two new variables.
  - Session docs must state `이번 case의 신규 변수: [...]`.
  - Future ideas go to `claudedocs/BACKLOG.md`, not into implementation.
  - `START_HERE.md` Active Case is the single source of truth.
  - New grasp outputs go under `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`.

## Frozen / Background Assets

- D321 tap dataset:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/lerobot_dataset`.
- D321 session:
  `claudedocs/session_20260708_cube10cm_top_view_d321_physicality_gate_low_mid_production.md`.
- D321 design draft:
  `claudedocs/design_d321_goal_conditioned_primitive.md`.
- D319 RL data-factory direction is now background, not the active case:
  `claudedocs/direction_20260708_rl_data_factory.md`.
