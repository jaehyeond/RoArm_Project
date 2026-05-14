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
