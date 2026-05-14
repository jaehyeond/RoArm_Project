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

## D006 - URDF gripper joint `lower="0"` clamps sim gripper open targets

Evidence:

- `assets/roarm_m3/urdf/roarm_m3.urdf` defines
  `<joint name="link5_to_gripper_link" ...><limit lower="0" upper="1.571" effort="2.5" velocity="3.14"/></joint>`.
- Isaac Lab loads this into `_robot.data.soft_joint_pos_limits`, which
  `_pre_physics_step` clamps via
  `torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)`
  (`roarm_rl/roarm_stack_env.py` ~L408-412).
- (delta.2) experiment 2026-05-14: set `GRIPPER_OPEN_DEG = -10.0` in
  `chain_skills.py`. Sim trace showed gripper actual stuck at `-0.0 deg`
  while `max_err_deg = 10.00` exactly matched the unattained `-10 deg` target.
- (delta.1) vs (delta.2) Skill 1b end-state was bit-identical:
  `tcp_after1 z = +51.9 mm` and `tcp_err_pre_close = 18.88 mm` in both runs.
  Conclusion: jaw-width hypothesis was UNTESTABLE in current sim.

Implication:

- Hardware range `[-10°, +100°]` in `CLAUDE.md` is HARDWARE ONLY. The URDF was
  authored with `lower=0` for grasp convention. Sim gripper cannot open beyond
  0° without URDF modification.
- DO NOT modify URDF `link5_to_gripper_link` lower limit casually. P6v14a
  was trained against this URDF; changing it invalidates the trained policy
  and forces re-training (~1-2 weeks).
- If "open wider than 0°" is required, the proper paths are: (a) PPO retrain
  on modified URDF (expensive), or (b) accept current jaw geometry and fix
  the underlying issue differently (descent ramping, sim physics tuning).
- For future scripted-skill design: gripper open target = `0.0 deg`, NOT
  negative. Negative target produces meaningless `max_joint_err` noise without
  affecting physics.

Source:

- `claudedocs/session_20260514_alpha_prime_delta_topdown.md` (delta.2 section)
- `assets/roarm_m3/urdf/roarm_m3.urdf` line with `link5_to_gripper_link`

## D007 - Scripted skill closed-loop in Isaac Lab needs `robot_dof_targets` force-set to avoid PD limit cycle

Evidence:

- `roarm_stack_env._pre_physics_step` accumulates targets every step:
  `targets = self.robot_dof_targets + self.cfg.action_scale * self.actions`
  with `action_scale = 0.1 rad/step` and actions clamped to `±1`.
- For scripted Skill 0 (HOME -> hover), naive closed-loop with saturated
  action produced PD limit-cycle: target accumulation pushed joint to its
  limit and PD response oscillated, requiring 200 max steps and producing
  `max_err 6.95°` with `tcp_err 35.2 mm` (5/13 iter #4-#5).
- Skill 1 trace showed elbow target overshoot `+7.7°` past q_grasp at s=40,
  consistent with PD limit cycle (sponge_y -44mm push observed).
- Force-set fix (`base_env.robot_dof_targets[:] = target_t` + null action
  through `env._pre_physics_step`) reduced Skill 0 to 21 steps with
  `max_err 1.38°` and `tcp_err 8.2 mm` (5/13 iter #7, then 5/14 also).

Implication:

- Scripted skills in this Isaac Lab env MUST force-set
  `base_env.robot_dof_targets` directly per-step, then call `env.step(null_action)`
  to advance physics. Do NOT rely on the action interface for scripted skills.
- This pattern is REQUIRED for: Skill 0/1a/1b/2/4 in `chain_skills.py`.
- Skill 3 (P6v14a learned policy) uses the action interface as designed
  (stochastic during training, deterministic during inference) — force-set is
  only for the deterministic scripted skills.
- PPO training does NOT exhibit this limit cycle because stochastic actions
  do not saturate, so the limit cycle is purely a deterministic-script artifact.

Source:

- `claudedocs/session_20260513_chain_skills_hierarchical_pivot.md` (7-iter trace, force-set fix at iter #7)
- `roarm_rl/chain_skills.py` (`_force_set_p6v14a_entry`, `_run_single_basin`, `run_chain_isaac` force-set pattern)
- `roarm_rl/roarm_stack_env.py` L408-412 (`_pre_physics_step` clamp logic)

## D008 — Skill 1b descent stall is contact equilibrium at TCP z ≈ +51.8mm, target-invariant; sub-stage splitting and step-ramping cannot bypass it

Evidence:

- 5/13 baseline + (δ.1) + (δ.2) + (δ.4) all report `tcp_after1 z = +51.8~+51.9mm`,
  bit-identical across 4 sessions, across approach geometry (top-down vs direct
  descent), across gripper target (0° vs URDF-clamped from -10°), across
  sub-stage splitting (one-shot q_grasp vs 4-waypoint hover→+50→+40→+33mm).
- (δ.4) 1b1 target +50mm: TCP reached +51.79mm, arm_err 0.32°, xyz_err 1.98mm.
  1b2 target +40mm: TCP +51.72mm, arm_err 2.96°, xyz_err 11.74mm. 1b3 target +33mm:
  TCP +51.80mm, arm_err 4.90°, xyz_err 18.81mm. TCP z INVARIANT across all 3
  sub-stages despite target depth varying 17mm.
- arm_err scaled with target depth (0.32° → 2.96° → 4.90°) → PD torque ↑, yet
  TCP z fixed → contact reaction force ↑ at same rate → equilibrium maintained.
- xy precision 0.6-0.8mm (xyz_err minus z_err); sponge xy unmoved across sub-stages
  (+266, -34.5)mm; pure z-axis contact equilibrium, not lateral collision or
  step-size limitation.
- 200 step max per sub-stage with TIGHT tol=0.005 rad (0.29°) — physics had ample
  settling time; stall is true equilibrium, not transient.

Implication:

- DO NOT attempt sub-stage splitting variations (5-stage, 6-stage, etc.) — same
  +51.8mm equilibrium predicted regardless of stage count.
- (δ.3) full ramping (`target = current + Δ`) ALSO REJECTED a priori — PD step
  increments would converge to the same contact equilibrium because the limiter
  is reaction force, not target overshoot.
- Root cause is in `(finger_geometry × gripper_jaw_width × sponge_collision_mesh
  × effort_limit)` space, NOT in trajectory parameterization. Fix must come from
  contact / geometry investigation, not from waypoint smoothing.
- Next-step priority: `(δ.5)` deep-dive — cheap diagnostics first (5-D sponge z
  measure, 5-E finger_tip FK at +51.8mm, 5-F gripper jaw separation at q=0°,
  ~1.5h total), then branch:
  - finger tip presses sponge TOP → redefine TCP_GRASP_Z or finger collision.
  - finger presses SIDE + jaw width < 22mm → URDF gripper limit change → P6v14a
    retrain risk (1-2 weeks) → (γ) PPO Skill 1 primitive likely the cheaper path.
  - effort_limit (2.5 Nm) bottleneck → URDF effort ↑ (mild retrain risk).

Source:

- `claudedocs/session_20260514_alpha_prime_delta_topdown.md` (`(δ.4) Result` section, errata APPENDIX)
- B200 logs `/tmp/chain_topdown_v3.out` (Skill 1b multi-stage descent trace)
- `claudedocs/EXPERIMENT_LEDGER.md` (2026-05-14 (δ.4) row)

## D009 — Skill 1b stall is gripper-link top-contact against settled sponge top; fix grasp geometry before stacking demos

Evidence:

- (δ.5) B200 diagnostic run `/tmp/chain_topdown_v4.out` added 5-D/5-E/5-F
  logging to `chain_skills.py`.
- 5-D: spawn write initially sets sponge root z to `+11.4mm` as expected, but after
  Skill 0/1a settling the sponge root is `+23.5mm`; with 47mm edge-stand height,
  actual sponge top is `+47.0mm`.
- 5-E: at all Skill 1b sub-stage equilibria, TCP remains `+51.72~+51.80mm`,
  while gripper_link collision mesh world bbox min-z is `+47.4mm`; this is only
  `+0.4mm` above the measured sponge top. XY AABB overlap is large
  (`~60.0mm × 22.0mm`), so the gripper collision envelope sits directly over
  the sponge top/width.
- `grasped=False` and `_was_grasped=False` through Skill 1b; this is not a
  grasp latch artifact. It is collision/contact geometry before grasp.

Implication:

- The immediate bottleneck is not target depth, step-size, or action interface.
  The current top-down TCP/gripper pose drives the gripper_link collision mesh
  onto the sponge top before the TCP can reach `+33mm`.
- Next fixes should test grasp geometry changes first: redefine `TCP_GRASP_Z`,
  shift TCP relative to sponge center, change wrist/gripper approach pose, or
  modify gripper collision mesh. Do not resume staged descent/ramping variants.
- For the four-sponge well/hash stacking task, do not generate large demos until
  the top-down pick primitive can clear the settled sponge top and close on the
  sponge without top-contact stall.

Source:

- `claudedocs/session_20260514_alpha_prime_delta_topdown.md` (`(δ.5) Result` section)
- B200 logs `/tmp/chain_topdown_v4.out`
