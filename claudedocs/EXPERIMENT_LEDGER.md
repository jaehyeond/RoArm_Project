# EXPERIMENT_LEDGER.md

Append-only index of major experiments and decisions. Keep entries short and
link to detailed logs. Do not use this as the only source for metrics; verify
from the linked session/data files before making claims.

| Date/Label | Run/Path | Goal | Key Result | Verdict | Source |
|---|---|---|---|---|---|
| 2026-05-13 | P6v14a | Phase 0a pregrasp release bootstrap | Stage4 success `77.8%`; first strong release bootstrap | Narrow success | `claudedocs/phase1_balpha_p6v14a_sanity_20260513.md` |
| 2026-05-13 | P6v14c | Resume P6v14a into bridge/full-chain PPO | Iter 0 stage4 `36.5%`, then early PPO collapse to near-zero | Fail; PPO destroys useful release behavior | `claudedocs/session_20260513_p6v14c_failure_analysis.md` |
| 2026-05-14 | P6v15 Path A | Reset gripper actor bias / PPO safeguards | Metrics near bit-identical to P6v14c | Fail; close-bias hypothesis rejected | `claudedocs/session_20260514_path_a_reject_path_b_rpl.md` |
| 2026-05-14 | P6v16/P6v16b/P6v16c Path B | Residual Policy Learning alpha sweep | Alpha `0.30/0.05/0.10` all collapse at iter 10 around `0.003-0.004` stage4 | Fail; RPL framework rejected for this PPO setup | `claudedocs/session_20260514_evening_rpl_sweep_fail_pathD_entry.md` |
| 2026-05-14/17 label | Path D v1 | P6v14a rollout demos -> release BC -> pregrasp BC eval | Nominal success `175/256 = 68.36%` | Ambiguous; required CLEAN audit | `claudedocs/pathD_data/eval_metrics.pt`, `claudedocs/pathD_data/analyze_eval.py` |
| 2026-05-17 label | Path D v2 | Exact-step gripper audit of Path D eval | CLEAN `24/256 = 9.38%`, DIRTY `151/256 = 58.98%`, nominal `68.36%` | Fail under `<10% CLEAN` gate; pivot recommended | `claudedocs/session_20260517_pathD_v2_clean_rate_fail.md` |

## Current Next Experiment Candidate

Procedural release-only demonstrations:

- IK/pregrasp near target.
- Scripted gripper-open action.
- Gravity settle.
- Direct CLEAN success filter with `gripper_q_at_success < 0.4 rad`.
- Train release BC and evaluate nominal/CLEAN/DIRTY split.
