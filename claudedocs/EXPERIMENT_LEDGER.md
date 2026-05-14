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
| 2026-05-13 late | chain_skills.py 7-iter B200 | Hierarchical chain pivot (scripted Skill 0/1/2 + P6v14a Skill 3) | Skill 0 PD limit cycle fix via `robot_dof_targets` force-set (200→21 steps), Skill 1 sponge side-collision (-44mm Y push) | Partial fail; reach OK, descent+grasp fail | `claudedocs/session_20260513_chain_skills_hierarchical_pivot.md` |
| 2026-05-14 (α') | P6v14a basin-of-attraction sweep | 6-point grid (dx 0-45mm, dz 0-20mm) on P6v14a release primitive | release_step=13 consistent 5/6 runs (memorization-free), (15,0) outlier=197 brittle, all final_d_xy 147-227mm post-release sponge knock-away | Partial PASS; release primitive portable 30-45mm xy / +20mm z, needs early-terminate at release+buffer | `claudedocs/session_20260514_alpha_prime_delta_topdown.md` |
| 2026-05-14 (δ) | Top-down chain v1 (GRIPPER_OPEN_DEG=0) | Re-architected Skill 0/1a/1b/1c/2/3/4 with q_high TCP +150mm clearance | Lateral knock 7× improved (Y -44→-6mm), BUT vertical descent stall tcp_after1 z=+51.9mm (target +33mm, bit-identical to 5/13), grasped=False, CHAIN_FINAL_SUCCESS=NO | Fail; geometry redesign insufficient | `claudedocs/session_20260514_alpha_prime_delta_topdown.md` |
| 2026-05-14 (δ.2) | GRIPPER_OPEN_DEG 0→-10° widen-jaw test | Test hypothesis that finger-tip width<22mm sponge width caused Skill 1b stall | URDF gripper limit `lower="0"` → env `soft_joint_pos_limits` clamps target -10° to actual 0° (verified L408-412 `_pre_physics_step` torch.clamp); tcp_after1 z=+51.9mm = δ.1 bit-identical | TEST INVALID (untestable in current sim); URDF mod = P6v14a retrain risk = NO-GO | `claudedocs/session_20260514_alpha_prime_delta_topdown.md` |
| 2026-05-14 (δ.4) | Skill 1b multi-stage z diagnostic (hover→+50→+40→+33mm) + exclude_gripper option + GRIPPER_OPEN_DEG=0 revert | Locate exact z where descent stalls; gripper-state independent (D006) | All 3 sub-stages stall at **TCP z ≈ +51.8mm INVARIANT under target depth**; arm_err grew (0.32°→2.96°→4.90°) but z fixed; xy precision 0.6-0.8mm; sponge xy unmoved across sub-stages → pure z contact equilibrium, NOT step-size issue. Skill 0/1a early-break restored (21/14 steps) via `exclude_gripper=True`. CHAIN_FINAL_SUCCESS=NO | Diagnostic SUCCESS / performance FAIL; sub-stage splitting REJECTED; (δ.3) full ramping ALSO predicted useless; pivot to (δ.5) contact/geometry deep-dive (5-D/5-E/5-F) | `claudedocs/session_20260514_alpha_prime_delta_topdown.md` (APPENDIX + δ.4 sections); B200 `/tmp/chain_topdown_v3.{out,err}` |
| 2026-05-14 (δ.5) | 5-D/5-E/5-F contact geometry diagnostic | Verify sponge top, gripper collision envelope, and jaw proxy at Skill 1b stall | Spawn initially root z=+11.4mm, but after settling sponge root z=+23.5mm → top z=+47.0mm. At all Skill 1b stall states TCP z=+51.7~+51.8mm; gripper_link mesh min-z=+47.4mm, only +0.4mm above sponge top, with XY overlap ~60mm×22mm. `grasped=False`, so not latch artifact. | Diagnostic SUCCESS; root cause now top-contact/collision geometry. Next: redesign top-down pick geometry before 4-sponge well/hash stacking demos. | `claudedocs/session_20260514_alpha_prime_delta_topdown.md`; B200 `/tmp/chain_topdown_v4.{out,err}` |

## Current Next Experiment Candidate

**Active pivot (2026-05-14 PM, post-δ.4)**: Hierarchical chain skills with P6v14a as learned release sub-skill.
(D003 contamination avoided — P6v14a used directly as primitive, NOT as BC training source.)

(δ.5) result CONFIRMED the D008 contact equilibrium is gripper-link top-contact: settled sponge top ≈ +47.0mm, gripper_link collision mesh min-z ≈ +47.4mm at the +51.8mm TCP stall (D009).

Decision pending — Skill 1b top-contact fix before generating four-sponge stacking demos:

1. **G1** — Geometry redesign: raise/redefine `TCP_GRASP_Z`, shift approach relative to sponge center, or rotate wrist/gripper so collision mesh clears sponge top; 1-2h; HIGH.
2. **G2** — Collision simplification: inspect/modify gripper_link collision mesh or USD convex hull if overly conservative; 2-4h; MEDIUM.
3. **G3** — If geometry cannot solve cleanly, train (γ) PPO Skill 1 descend+grasp primitive; 1-2 weeks; paper-quality.

Recommended order: G1 quick geometry patch + B200 chain rerun. Do not proceed to four-sponge dataset/demo generation until top-down pick succeeds.

Outcome branches:
- finger_tip presses sponge TOP → redefine TCP_GRASP_Z or finger collision mesh.
- finger presses SIDE + jaw < 22mm → URDF gripper limit change requires P6v14a retrain (~1-2 weeks) → (γ) PPO Skill 1 primitive likely cheaper path.
- effort_limit bottleneck → URDF effort↑ (mild retrain risk).

Reserve / fallback:
- **(γ)** PPO Skill 1 descend+grasp primitive; 1-2 weeks; paper-quality.
- Procedural release-only demos → train release BC → CLEAN eval (Codex prior pivot, still valid).
