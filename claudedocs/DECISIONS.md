# DECISIONS.md

Durable project decisions and lessons. Append new decisions; do not delete old
ones. If a decision is superseded, mark it as superseded and link the newer
decision. Detailed evidence belongs in `claudedocs/session_*.md`.

## D001 - PPO-only reward shaping is not the main path

Evidence:

- P6v6 through P6v14 repeatedly found reward-farming basins.
- P6v14c started with useful release behavior but collapsed from high initial
  stage4 to near-zero by early PPO updates.
- RPL alpha sweep P6v16/P6v16b/P6v16c was alpha-invariant in early collapse.

Implication:

- PPO-only remains useful as a baseline/ablation, not the primary route for the
  long-horizon sponge stacking policy.
- If RL is reintroduced, use demonstrations, KL/BC regularization, frozen prior
  steering, or a similarly constrained method.

Sources:

- `claudedocs/session_20260513_p6v14c_failure_analysis.md`
- `claudedocs/session_20260514_evening_rpl_sweep_fail_pathD_entry.md`

## D002 - Nominal stage4 success is insufficient

Evidence:

- Path D v1 nominal success was `175/256 = 68.36%`.
- Path D v2 exact-step gripper capture showed CLEAN success was only
  `24/256 = 9.38%`.
- Counter-path inflated nominal success by about `7.3x`.

Implication:

- Always report CLEAN/DIRTY split for placement success.
- Required metrics: `gripper_q_at_success`, success-step histogram, nominal
  success, CLEAN success, DIRTY/counter-path count.
- Do not call a run successful from nominal `_place_success_flag` alone.

Source:

- `claudedocs/session_20260517_pathD_v2_clean_rate_fail.md`

## D003 - RL-rollout-derived release BC is contaminated for Path D

Evidence:

- D.1 demos came from P6v14a rollout on the same environment.
- That source included counter-path artifacts where `_place_counter >= 50`
  could fire without direct gripper-open release.
- BC learned a safer hover/closed-gripper mode that inflated nominal success but
  did not produce enough CLEAN release success.

Implication:

- Do not collect more release BC data from the same P6v14a rollout source as the
  main fix.
- Prefer procedural release demos with scripted gripper-open and direct CLEAN
  success filtering.

Source:

- `claudedocs/session_20260517_pathD_v2_clean_rate_fail.md`

## D004 - B200 is headless state-only learning/eval by default

Evidence:

- Current Isaac Lab workflows on B200 use 28-dim state-only observations and
  headless eval/training.
- Rendering belongs to replay/extraction workflows unless explicitly testing
  render capability.

Implication:

- Do not spend B200 cycles on visual rendering by default.
- Keep B200 launch scripts fail-fast guarded with root, user, host, and md5
  checks.

Sources:

- `claudedocs/phase1_step_abc_complete_20260507.md`
- `claudedocs/session_20260517_pathD_v2_clean_rate_fail.md`

## D005 - New-session boot must be file-grounded

Evidence:

- `HANDOFF.md` and `TASKS.md` are stale but still present.
- Recent project state lives across session docs, data files, launch scripts, and
  agent memories.
- Relying on memory alone can miss the Path D v2 CLEAN failure and overstate v1
  nominal success.

Implication:

- New Claude Code/Codex sessions must read `START_HERE.md`,
  `claudedocs/DECISIONS.md`, and `claudedocs/EXPERIMENT_LEDGER.md` before making
  current-state claims or edits.
- Metrics must be verified from referenced logs/data before being cited.

Source:

- `START_HERE.md`
