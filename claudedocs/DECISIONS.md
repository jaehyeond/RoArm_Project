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

## D010 — Collision proxy proves top-contact was real; after latch, close must terminate immediately

Evidence:

- G2-A replaced only `gripper_link` collision with a tiny `4mm × 4mm × 4mm`
  proxy (`gripper_link_collision_g2a.stl`) while preserving visual mesh and
  regenerated B200 USD.
- `/tmp/chain_topdown_g2a_v2.out`: 5-F confirmed the proxy bbox, Skill 1b reached
  the +47mm target in `(14,8,8)` steps with `stall_signature=FALSE`, and 5-E
  showed `mesh_min_z_minus_sponge_top=+56.5mm` at final 1b instead of the D009
  `+0.4mm/-0.2mm` top-contact regime. This validates the collision-envelope
  diagnosis.
- In the same run, close was stable and close-end `d_sponge_tcp=23.6mm` was inside
  the env's 25mm grasp distance, but `gripper_q=21.98°` was below the 0.4rad
  (`22.9°`) threshold, so `grasped=False`.
- `/tmp/chain_topdown_g2a_v3.out`: increasing latch target to 26° produced
  `grasped=True`, but the close loop continued after latch; the sponge attach
  path then destabilized the arm (`tcp_after1=(-99.1,+12.7,+636.0)mm`,
  `tcp_err=694.6mm`) and the chain still failed.

Implication:

- G2-A confirms top-contact was not a trajectory artifact: simplifying the
  collision envelope removes the Skill 1b z stall.
- Do not keep closing/holding at the grasp pose after `_grasped=True`. Skill 1c
  must terminate immediately when the latch fires, then hand off to Skill 2
  lift/transport. Continuing the close loop after latch can inject a physics
  explosion via kinematic sponge attach and table/object contacts.
- Next concrete patch: in Skill 1c, use a custom loop or callback break condition:
  target gripper just above threshold, step until `_grasped=True`, then break
  immediately and begin Skill 2. Keep the G2-A collision proxy for this test.

Sources:

- B200 `/tmp/chain_topdown_g2a_v2.out`
- B200 `/tmp/chain_topdown_g2a_v3.out`
- `local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl`

## D011 — Skill 1c latch early-stop is necessary but not sufficient; Skill 2 attached transport now owns the failure

Evidence:

- G2-A v4 patched Skill 1c to force-set `q_grasp`, command `GRIPPER_LATCH_DEG=26.0`,
  and break immediately after the first env step where `base_env._grasped[0]`
  becomes True.
- `/tmp/chain_topdown_g2a_v4.out`: Skill 1b remained clean with proxy collision:
  `(14,8,8)` steps, `stall_signature=FALSE_at_b3`, and final
  `mesh_min_z_minus_sponge_top=+56.5mm`.
- Skill 1c early-stop worked: latch detected after 15 steps; close ended with
  `gripper_q=23.02°`, `d_sponge_tcp=23.6mm`, `grasped=True`, and
  `tcp_after1=(+265.2,-34.4,+47.1)mm` (no repeat of the v3 `z=+636mm`
  Skill-1c explosion).
- The next failure moved to Skill 2: immediate attached transport ran to 120 max
  steps with `max_arm_err=66.99°`, `tcp_err=481.7mm`, `grasped=True`, and
  sponge z `378.9mm`; final `CHAIN_FINAL_SUCCESS=NO`.
- Env mechanics explain why this is the correct surface to inspect next:
  `_apply_action()` calls `_update_grasp_attach()` whenever `_grasped.any()`
  (`roarm_rl/roarm_stack_env.py` L413-420), and `_update_grasp_attach()` writes
  the sponge pose to current TCP with zero velocity (L950-964).

Implication:

- Do not declare the pick primitive fixed just because Skill 1c can latch without
  exploding. The failure boundary has moved downstream to attached Skill 2
  transport/lift semantics.
- Next experiments should instrument Skill 2 and `_update_grasp_attach` timing /
  attached-object dynamics. Keep G2-A proxy for this branch, but do not proceed
  to four-sponge stacking demos until Skill 2 can lift/transport without the
  attached-object explosion.

Sources:

- B200 `/tmp/chain_topdown_g2a_v4.out`
- `roarm_rl/chain_skills.py` Skill 1c latch early-stop patch
- `roarm_rl/roarm_stack_env.py` L413-420, L950-964

## D012 — Stable attached transport conflicts with P6v14a release-entry distribution

Evidence:

- G2-A v5 added dense Skill 2 diagnostics. `/tmp/chain_topdown_g2a_v5_skill2diag.out`
  showed the original Skill 2 target starts with a `+90°` arm error entirely from
  wrist_r `+90° -> 0°`; TCP/sponge then drift upward together under `_grasped=True`.
- G2-A v6 held wrist_r at `+90°`. Skill 2 no longer exploded and stopped after
  one arm step with `tcp_err=7.9mm`, but the handoff was not a P6v14a entry:
  `gripper_q=24.82°`, `sponge_z=40.1mm`, wrist_r `+90°`.
- G2-A v7/v8 tried to wait for gripper/full-error convergence while attached.
  Both reintroduced runaway: v7 reached `max_arm_err=261.79°`, v8 reached
  `max_arm_err=272.81°`. This happened even with wrist_r held at `+90°`, so
  attached dwell/post-latch close itself is unsafe.
- G2-A v9 used the stable pattern: wrist_r held `+90°`, gripper target held at
  latch `26°`, arm-only break. Skill 2 stayed short/stable (`steps=1`,
  `tcp_err=8.0mm`, no z explosion), but P6v14a Skill 3 failed:
  post-Skill3 `d_xy=508.1mm d_z=365.0mm`; final `CHAIN_FINAL_SUCCESS=NO`
  with `final_d_xy=704.8mm`.

Implication:

- Do not rotate wrist_r from `+90°` to `0°` while the sponge is attached.
- Do not continue closing or dwell for convergence after latch while attached.
- The stable post-pick handoff state is outside P6v14a's learned release-entry
  distribution. Continuing to force P6v14a Skill 3 after this altered handoff is
  not a valid path to success without either a new release primitive trained for
  the stable entry or a different scripted/physics release bridge.

Sources:

- B200 `/tmp/chain_topdown_g2a_v5_skill2diag.out`
- B200 `/tmp/chain_topdown_g2a_v6_holdwrist.out`
- B200 `/tmp/chain_topdown_g2a_v7_holdwrist_fullerr.out`
- B200 `/tmp/chain_topdown_g2a_v8_holdwrist_latchgrip.out`
- B200 `/tmp/chain_topdown_g2a_v9_holdwrist_latchgrip_armbreak.out`

## D013 — Minimal scripted release bridge succeeds from stable G2-A handoff, but remains a diagnostic bridge

Evidence:

- G2-A v10 replaced P6v14a Skill 3 with a minimal scripted release bridge from
  the v9 stable handoff distribution: wrist_r held `+90°`, gripper held near
  latch, sponge below TCP after one stable attached transport step.
- `/tmp/chain_topdown_g2a_v10_scripted_release_bridge.out` verified the upstream
  pass gates remained intact: Skill 1b no top-contact stall, Skill 1c latch at
  step 15, and Skill 2 short/stable (`steps=1`, `tcp_err=8.0mm`,
  `sponge_z=40.1mm`).
- The bridge opened the gripper below `grasp_gripper_thresh`; `_grasped` cleared
  at `release_step=1`. The sponge then settled with minimal robot motion:
  post-release `d_xy=22.3mm`, `d_z=12.1mm`, `CHAIN_SETTLED=YES`.
- Retreat did not disturb the placed sponge: final `d_xy=22.3mm`, `d_z=12.1mm`,
  `CHAIN_FINAL_SUCCESS=YES`.

Implication:

- The stable G2-A handoff is physically release-compatible if release is kept
  minimal: open gripper, let `_grasped` clear, and avoid wrist rotation,
  post-latch close, or attached dwell.
- This is enough to proceed to a careful top-down pick/release primitive
  diagnostic and then four-sponge planning, but it should not be overclaimed as a
  learned release solution. For paper-quality autonomy, a learned release
  primitive trained from the stable G2-A handoff distribution remains the cleaner
  path.
- Do not add random scripted release variants if v10 fails in a broader layout.
  Either keep the minimal bridge semantics or train a distribution-correct
  release primitive.

Sources:

- B200 `/tmp/chain_topdown_g2a_v10_scripted_release_bridge.out`
- `roarm_rl/chain_skills.py` Skill 3 scripted release bridge

## D014 — v10 release bridge does not solve four-source layout; long attached transport reintroduces Skill 2 runaway before release

Evidence:

- G2-A v11 added a seed0 four-source layout diagnostic for the existing
  single-sponge env. Because the current env has no L1 support bodies, L2 stack
  targets were intentionally skipped; S1 was tested as a floor placement from
  source `(+0.2137,-0.1957)` to L1.sp1.
- `/tmp/chain_topdown_g2a_v11_layout_source_sweep.out` line 137 confirmed Skill
  1c still latched cleanly at step 15; line 138 had `gripper_q=23.02deg`,
  `d_sponge_tcp=21.2mm`, `grasped=True`.
- The failure occurred before release: Skill 2 long attached transport ran to
  max steps with `max_arm_err=253.21deg`, `tcp_err=486.5mm`, `sponge_z=66.7mm`
  (line 263). The bridge then released a sponge already about half a meter from
  target, yielding post-release `d_xy=555.0mm`, `CHAIN_SETTLED=NO` (line 280)
  and final `CHAIN_FINAL_SUCCESS=NO` (line 283).

Implication:

- v10 proves a stable short handoff can physically release, but it does not solve
  source-to-target transport from the four-source layout.
- Do not spend time adding scripted release variants for this failure. Release is
  downstream; the current failing surface is long attached transport under
  `_update_grasp_attach`.
- Four-sponge progression needs a transport-compatible strategy: either a learned
  primitive trained from realistic source-to-target attached states, a proper
  physics gripper/constraint model, or a redesigned staged primitive that avoids
  unsafe long attached dwell. The existing minimal release bridge remains useful
  only after a stable handoff near the target.

Sources:

- B200 `/tmp/chain_topdown_g2a_v11_layout_source_sweep.out`
- `roarm_rl/chain_skills.py` `--layout_source_sweep`

## D015 — Quick SurfaceGripper retrofit is not a drop-in fix for four-source transport

Evidence:

- Isaac Lab's installed `SurfaceGripper` is CPU-only. The local source states:
  "SurfaceGripper is only supported on CPU for now" and instructs `--device cpu`
  (`IsaacLab/source/isaaclab/isaaclab/assets/surface_gripper/surface_gripper.py`
  lines 48-50 and 260-265). The official tutorial repeats this CPU-only
  constraint.
- Added `sim_scripts/surface_gripper_transport_probe.py`, a separate CPU-only
  diagnostic that does not use `_update_grasp_attach` and dynamically creates an
  Isaac Sim `SurfaceGripper` prim before sim reset.
- Probe v2 attached the gripper under `Robot/link5/SurfaceGripper` with TCP
  offset. B200 `/tmp/roarm_surface_gripper_transport_probe_v2.out` line 143
  showed `close_detect_step=-1`; line 152 showed the robot reached the transport
  TCP (`tcp_err=7.9mm`) but the sponge stayed at the source with
  `d_xy_pre_release=166.1mm`; line 164 ended `SURFACE_PROBE_SUCCESS=NO`.
- Probe v3 moved the gripper under `Robot/gripper_link/SurfaceGripper`, zero
  local offset, and increased `grip_distance=0.200`. B200
  `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.out` line 144 again
  showed `close_detect_step=-1`; line 153 again showed `tcp_err=7.9mm` while the
  sponge remained at the source (`d_xy_pre_release=166.1mm`); line 165 ended
  `SURFACE_PROBE_SUCCESS=NO`.

Implication:

- SurfaceGripper is a plausible physics-gripper direction, but not as a quick
  dynamic prim retrofit on the current RoArm USD. It needs proper asset authoring
  of gripper pose/axis/API semantics before it can replace `_update_grasp_attach`.
- Do not interpret v2/v3 as proof that a physical constraint cannot transport the
  sponge. The probes did not reach a closed/attached state.
- Do not keep trying arbitrary SurfaceGripper parent/offset variants. The next
  useful branches are either:
  - properly author and validate a gripper asset/constraint in USD, with a unit
    test that reaches `state=Closed` on the sponge before chain integration; or
  - train a learned transport/release primitive from realistic G2-A four-source
    attached distributions.

Sources:

- `sim_scripts/surface_gripper_transport_probe.py`
- B200 `/tmp/roarm_surface_gripper_transport_probe_v2.{out,err}`
- B200 `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.{out,err}`

## D016 — P7 learned attached transport improves XY but does not solve release/upright placement

Evidence:

- Added `curriculum_attached_transport_release` and `reward_phase=7` so B200 PPO
  starts from realistic G2-A seed0 attached handoff states instead of re-learning
  Skill 1b/1c.
- B200 reset probe `/tmp/p7v1_attached_reset_probe_v2.out` verified the intended
  distribution: `grasped_frac=1.000` (line 65), `was_grasped_frac=1.000` (line
  66), `d_sponge_tcp_mean_mm=0.00` (line 67), initial mean `d_xy=175.80mm`
  (line 68).
- P7v1 diagnostic showed the initial reward still allowed closed/high hold:
  `/tmp/p7v1_diag20.out` line 584 had `p7_xy_offset_mean=0.2391`, line 589
  `p7_gripper_open_rate=0.0631`, and line 596 `p7_sponge_height_m=0.1437`.
- P7v3 fixed reward/latch contamination and improved transport. In the full B200
  run `/tmp/p7v3_transport_release.out`, iteration 496 had
  `p7_xy_offset_mean=0.0512` (line 14984), `p7_release_z_offset_mean=0.0328`
  (line 14985), and `p7_settled_z_offset_mean=0.0138` (line 14986).
- However the same run did not solve the primitive: line 14991
  `p7_on_target_rate=0.0005`, line 14993 `p7_upright_rate=0.0576`, and line
  14994 `p7_place_success_rate=0.0007`.

Implication:

- B200 PPO is useful here: it reduced mean XY from about `176mm` to about `51mm`.
  But this is only partial transport improvement, not a valid pick-place
  primitive.
- Do not claim learned transport/release is solved.
- Do not run more blind reward variants without first inspecting P7 rollout
  failure modes, especially object orientation/upright loss after release.
- The SurfaceGripper/constraint path remains separate and still requires an
  authored unit test that reaches `Closed` before chain integration.

Sources:

- `roarm_rl/roarm_stack_env.py` P7 curriculum/reward
- `launch_p6v17_transport_release.sh`
- B200 `/tmp/p7v1_attached_reset_probe_v2.{out,err}`
- B200 `/tmp/p7v1_diag20.{out,err}`
- B200 `/tmp/p7v3_diag20.{out,err}`
- B200 `/tmp/p7v3_transport_release.{out,err}`

## D017 — P7 model_499 fails mainly by object orientation collapse during attached transport/release

Evidence:

- Added `sim_scripts/p7_rollout_failure_diag.py`, a state-only B200 rollout
  diagnostic for
  `$ROARM_B200_ROOT/logs/roarm_rl/roarm_stack_p7v3_g2a_attached_transport_release/model_499.pt`.
  It records reset, pre-release, release, post-settle, and final sponge/TCP/
  target states, including quaternion-derived
  `sz_world_z = 1 - 2(qx^2 + qy^2)`.
- B200 `/tmp/p7v3_rollout_failure_diag.out` line 42 verified the intended
  checkpoint path; line 43 ran `num_envs=256 episodes=2`; line 93 captured
  `completed_episodes=512`.
- Line 95 classified all 512 episodes as
  `C_tips_during_attached_transport`.
- The reset distribution was upright and realistic: line 97 had
  `d_xy=0.1732`, `release_z_offset=0.0069`, `settled_z_offset=0.0359`,
  `sz_world_z=1.0000`.
- By pre-release, line 98 showed the upright signal had already collapsed:
  `d_xy=0.0783`, `release_z_offset=0.0770`, `settled_z_offset=0.1060`,
  `sz_world_z=0.2667`.
- At release, line 99 showed the object was still far from a clean release
  state: `d_xy=0.0739`, `release_z_offset=0.0788`,
  `settled_z_offset=0.1078`, `sz_world_z=0.2851`.
- Final z error was deceptively small only because the sponge was lying flat:
  line 101 had `settled_z_offset=0.0006` but `sz_world_z=0.0156`.

Implication:

- P7v3's mean XY improvement is not a solved primitive and not merely a
  reporting artifact. The policy/attach dynamics produce orientation collapse
  before or at release.
- Do not start another blind scalar reward variant from `model_499.pt` before
  inspecting the attached transport action/path/quaternion dynamics.
- A useful learned-branch fix must preserve or explicitly control object
  orientation through attached transport and release. A useful physics branch
  still requires a proper authored gripper/constraint unit test before replacing
  `_update_grasp_attach`.

Sources:

- `sim_scripts/p7_rollout_failure_diag.py`
- `claudedocs/session_20260515_p7_rollout_failure_diag.md`
- B200 `/tmp/p7v3_rollout_failure_diag.{out,err}`

## D018 — P7 upright collapse starts while still kinematically attached, before gripper open/release

Evidence:

- Added `sim_scripts/p7_action_tcp_quat_trace.py`, a state-only step trace for
  P7 `model_499.pt` that logs policy action, gripper action/joint/open flag,
  TCP path/delta/velocity, sponge quaternion, `sz_world_z`, `_grasped`,
  `_was_grasped`, sponge-TCP distance, and rigid-object velocity norms.
- Direct source inspection confirmed `_update_grasp_attach` in
  `roarm_rl/roarm_stack_env.py` lines 1096-1110 writes the sponge position to
  TCP, preserves the current sponge quaternion at line 1107, and zeroes root
  velocity at line 1110.
- B200 `/tmp/p7v3_action_tcp_quat_trace.out` line 99 showed reset starts were
  upright and attached: `d_xy=0.1722`, `sz=1.0000`,
  `d_sponge_tcp=0.00000`, `grasped=1.000`.
- Lines 100-103 showed three of four sampled envs already below upright
  threshold at step 1 while `open=0` and `grasped=1`, with large quaternion
  deltas and high angular velocity.
- Aggregate lines 245-253 showed the same for all 256 envs:
  `first_tip_while_grasped=256/256`, `tip_before_or_at_open=256/256`,
  `tip_while_grasped_before_or_at_release=256/256`.
- Aggregate lines 254-260 showed timing: mean first open/release step `20.21`,
  but mean first tip while grasped step `1.72`.
- Line 253 showed no large one-step TCP jump above `0.030m`; line 264 showed
  `max_tcp_delta_max=0.0246` but saturated actions
  (`max_abs_action_mean=1.0000`).
- Line 263 showed final z/XY remain misleading: final `d_xy=0.0238` and
  `settled_z_abs=0.0201`, but `sz=0.0759`.

Implication:

- The P7 failure is not primarily the gripper-open transition. Upright collapse
  begins during attached motion while `_grasped=True`.
- Because `_update_grasp_attach` preserves the current sponge quaternion, once
  physics/contact tips the object during an attached step, later attached steps
  keep carrying that tipped orientation.
- Do not change the P7 reward scalar first. The next useful learned/attach
  branch is an attach quaternion reset/constraint diagnostic to test whether
  transport becomes mechanically meaningful when attached orientation is
  controlled. The physics branch remains a properly authored gripper/constraint
  unit test, not another arbitrary SurfaceGripper parent/offset variant.

Sources:

- `sim_scripts/p7_action_tcp_quat_trace.py`
- `claudedocs/session_20260515_p7_action_tcp_quat_trace.md`
- B200 `/tmp/p7v3_action_tcp_quat_trace.{out,err}`

## D019 — Attach quaternion constraint suppresses immediate P7 tipping but does not solve transport/release

Evidence:

- Added `sim_scripts/p7_attach_quat_constraint_probe.py`, a runtime-only
  monkey-patch diagnostic for `_update_grasp_attach`. It does not edit reward,
  `roarm_stack_env.py`, `chain_skills.py`, scripted release, or assets.
- Baseline from `/tmp/p7v3_action_tcp_quat_trace.out`: lines 245-253 had
  `first_tip_while_grasped=256/256` and `tip_before_or_at_open=256/256`; lines
  261-264 had release `sz=0.2983`, final `sz=0.0759`, and final
  `d_xy=0.0238`.
- `preserve+keep` tested whether velocity zeroing alone caused the collapse.
  B200 `/tmp/p7v3_attach_quat_preserve_keep.out` lines 141-149 still had
  `first_tip_while_grasped=256/256` and `tip_before_or_at_open=256/256`; lines
  151-160 had mean first tip while grasped `1.67`, release `sz=0.1561`, final
  `sz=0.0101`.
- `identity+zero` tested a hard upright quaternion with the old velocity
  zeroing behavior. B200 `/tmp/p7v3_attach_quat_identity_zero.out` lines
  141-149 improved but did not eliminate attached tipping
  (`first_tip_while_grasped=189/256`, `tip_before_or_at_open=128/256`); lines
  151-160 had release `sz=0.9664` and final `sz=0.9113`, but final
  `d_xy=0.2487`.
- `identity+keep` best suppressed pre-release attached tipping. B200
  `/tmp/p7v3_attach_quat_identity_keep.out` lines 141-149 had
  `first_tip_while_grasped=77/256` and `tip_before_or_at_open=11/256`; lines
  151-160 had release `sz=0.9921`, but final `sz=0.6434` and final
  `d_xy=0.2604`.

Implication:

- Preserving the current sponge quaternion during kinematic attach is a major
  failure amplifier. Once the sponge tips, the old attach model keeps carrying
  the tipped pose.
- Velocity zeroing alone is not the primary cause; removing it while preserving
  quaternion still fails like baseline.
- A simple upright quaternion constraint is not a solved primitive. It improves
  upright release but exposes poor transport/placement with the old P7 policy.
  The old policy was trained under broken attach semantics and should not be
  judged as solved under the altered semantics.
- Next valid branches are:
  - implement a controlled env-level attach orientation semantic change and
    retrain/evaluate under that semantic; or
  - build a properly authored physics gripper/constraint unit test before chain
    integration.
- Do not claim attach reset solved P7, and do not jump to reward scalar tuning
  before choosing the mechanics branch.

Sources:

- `sim_scripts/p7_attach_quat_constraint_probe.py`
- `claudedocs/session_20260515_p7_attach_quat_constraint_probe.md`
- B200 `/tmp/p7v3_attach_quat_identity_zero.{out,err}`
- B200 `/tmp/p7v3_attach_quat_preserve_keep.{out,err}`
- B200 `/tmp/p7v3_attach_quat_identity_keep.{out,err}`

## D020 — Env-level identity+keep attach semantics are active, but do not solve P7 without a release/transport redesign

Evidence:

- Implemented gated env config, defaulting to original behavior:
  `attach_quat_mode="preserve"` and `attach_velocity_mode="zero"`.
- B200 `/tmp/p7_attach_semantics_identity_keep.out` line 64 confirmed
  `attach_quat_mode=identity attach_velocity_mode=keep`; line 66 showed a
  deliberately tipped attached sponge was reset upright (`sz_mean=1.0000`) while
  velocity was kept (`vel_norm_mean=3.0020`).
- B200 `/tmp/p7_attach_semantics_preserve_zero.out` line 64 confirmed the default
  `preserve+zero` path; line 66 preserved the 60deg tipped orientation
  (`sz_mean=0.5000`) and zeroed velocity (`vel_norm_mean=0.0000`).
- Fresh P7v4 identity+keep diagnostic training did not solve or even trend
  cleanly. `/tmp/p7v4_attach_identity_keep_diag20.out` lines 44-45 confirmed the
  enabled semantics. Line 105 had initial `p7_xy_offset_mean=0.1904`, but line
  586 worsened to `0.3620`; line 596 still had
  `p7_place_success_rate=0.0000`.
- Evaluating `model_19.pt` under the same env-level semantics confirmed a
  no-release failure: `/tmp/p7v4_attach_identity_keep_model19_trace.out` line 44
  had `identity keep`, line 94 confirmed `_update_grasp_attach` used that mode,
  lines 338-340 had `first_open=0/256` and `release_or_open=0/256`, and line 355
  ended at `final d_xy=0.1488`, `sz=0.9036`.

Implication:

- The controlled env-level mechanics switch is valid and should be kept as a
  diagnostic/training knob.
- `identity+keep` improves upright mechanics relative to the immediate-collapse
  old baseline, but it is not a P7 solution. Under a short fresh PPO diagnostic,
  it produced closed/no-release behavior and poor transport.
- Do not claim attach reset solved P7. Next work must either redesign the P7
  controller/reward/curriculum under the new mechanics to force release and
  target transport, or move to the authored physics gripper/constraint unit-test
  branch.

Sources:

- `roarm_rl/roarm_stack_env.py` attach semantics config and `_update_grasp_attach`
- `sim_scripts/p7_attach_semantics_env_probe.py`
- `sim_scripts/p7_action_tcp_quat_trace.py`
- `claudedocs/session_20260517_p7_attach_semantics_env_experiment.md`
- B200 `/tmp/p7_attach_semantics_identity_keep.{out,err}`
- B200 `/tmp/p7_attach_semantics_preserve_zero.{out,err}`
- B200 `/tmp/p7v4_attach_identity_keep_diag20.{out,err}`
- B200 `/tmp/p7v4_attach_identity_keep_model19_trace.{out,err}`

## D021 — P7 release guidance breaks no-release but exposes early/tipped release; do not threshold-tune blindly

Evidence:

- Added gated `p7_release_guidance` diagnostics with default off, leaving P7v3/P7v4
  reward unchanged when disabled.
- P7v5 with identity+keep and release-guidance xy `0.12` enabled open/release:
  `/tmp/p7v5_identity_keep_release_guidance_model19_trace.out` lines 239-241
  had `first_open=256/256`, `release_or_open=256/256`; lines 242-245 showed
  pre-open attached tip nearly suppressed (`first_tip_while_grasped=1/256`).
  But line 255 released far from target (`d_xy=0.1522`) and line 256 ended flat
  (`final d_xy=0.1260`, `sz=0.4126`).
- P7v6 tightened the release-guidance xy threshold to `0.08`. B200
  `/tmp/p7v6_identity_keep_release_guidance_xy08_model19_trace.out` line 354
  improved release XY to `0.0849`, but lines 341-344 showed attached tip before
  open returned (`118/256`), and line 355 still ended flat
  (`final d_xy=0.1055`, `sz=0.2840`).

Implication:

- P7v4's no-release failure was real and a local open signal can break it.
- However, open-signal threshold tuning alone trades one failure for another:
  early/far release at xy `0.12`, and delayed/tipped release at xy `0.08`.
- Do not continue blind P7 scalar/threshold tuning. The next useful learned branch
  needs a structured low-motion release/settle curriculum under identity+keep, or
  the project should move to Branch B authored physics gripper/constraint unit
  testing.

Sources:

- `roarm_rl/roarm_stack_env.py` gated `p7_release_guidance`
- `claudedocs/session_20260517_p7_release_guidance_diagnostics.md`
- B200 `/tmp/p7v5_identity_keep_release_guidance_diag20.{out,err}`
- B200 `/tmp/p7v5_identity_keep_release_guidance_model19_trace.{out,err}`
- B200 `/tmp/p7v6_identity_keep_release_guidance_xy08_diag20.{out,err}`
- B200 `/tmp/p7v6_identity_keep_release_guidance_xy08_model19_trace.{out,err}`

## D022 — Structured near-target P7 release smoke is active but fails post-release upright settle; do not long-train this A configuration

Evidence:

- Added default-off `p7_structured_release_curriculum` as a falsifiable A-branch
  test, not as another scalar/xy-threshold tuning run. Defaults remain unchanged:
  old attach behavior is still `attach_quat_mode="preserve"` and
  `attach_velocity_mode="zero"`, and the structured curriculum is disabled unless
  explicitly enabled.
- B200 `/tmp/p7v7_structured_release_smoke.out` line 68 confirmed the new
  mechanism was active: `attach_quat_mode=identity attach_velocity_mode=keep
  structured_release=True`.
- Line 69 confirmed a perfect near-target reset:
  `d_xy=0.0000`, `rel_z_abs=0.0000`, `sz=1.0000`,
  `grasped=1.000`, `open=0.000`.
- Lines 74-76 showed the hand-authored low-motion gripper-open smoke broke
  no-release without attached tipping:
  `first_open=64/64`, `release_or_open=64/64`,
  `tip_while_grasped_before_release=0/64`.
- Line 78 showed the release itself was close and upright:
  `release sz=0.9720`, `d_xy=0.0089`.
- Line 79 showed the actual failure after settle:
  `final d_xy=0.0411`, `settled_z_abs=0.0041`, but `sz=0.2484` and
  `success_rate=0.2344`.
- Lines 80-81 summarized the branch test: `MECHANISM_ACTIVE=YES` and
  `EARLY_KILL=YES`.

Implication:

- The structured A-branch gate is real, but the tested A configuration should not
  be long-trained. If a perfect near-target, arm-still, identity+keep release
  already settles flat, PPO is being asked to learn around a contact/settle
  mechanics failure rather than a policy problem.
- Do not continue P7 release-guidance scalar/threshold tuning, and do not simply
  extend this structured curriculum into a long run.
- The stronger next branch is now Branch B: properly author and validate a
  physics gripper/constraint unit test that reaches stable `Closed` before chain
  integration. Any future learned release work must first pass a comparable
  policy-free upright-settle smoke.

Sources:

- `roarm_rl/roarm_stack_env.py` default-off `p7_structured_release_curriculum`
- `sim_scripts/p7_structured_release_curriculum_probe.py`
- `claudedocs/session_20260517_p7_structured_release_curriculum_smoke.md`
- B200 `/tmp/p7v7_structured_release_smoke.{out,err}`

## D023 — Canonical SurfaceGripper+sponge unit test fails the Closed gate; do not use gripped-object count alone

Evidence:

- Added `sim_scripts/p7_branch_b_surface_gripper_unit_probe.py`, a CPU-only
  Branch B unit probe that uses Isaac Lab's canonical
  `Tests/SurfaceGripper/test_gripper.usd` rig and the project RoArm sponge. It
  does not edit `roarm_stack_env.py`, `train_ppo.py`, `chain_skills.py`, reward
  scalars, scripted release variants, or RoArm SurfaceGripper parent/offsets.
- B200 `/tmp/p7_branch_b_surface_gripper_unit_smoke.out` line 89 verified the
  canonical SurfaceGripper asset and sponge prim.
- Line 90 showed the reset state: sponge at `z=0.4986`, SurfaceGripper state
  open (`state=-1.0`).
- Lines 91-103 showed the close phase never reached `Closed`; state remained
  `0.0` or `-1.0`.
- Line 121 reported `closed_detect_step=-1`, `closed_frac=0.0000`,
  `gripped_positive_frac=1.0000`, and `max_drift=0.37595`.
- Line 122 showed the final state was still not closed (`state=+0.0`) and the
  sponge had drifted/fallen to `z=0.1235`.
- Line 123 ended `SURFACE_UNIT_SUCCESS=NO`. Final stderr lines 1-13 were
  NVML/cpufreq warnings only, with no Python traceback.

Implication:

- The first concrete Branch B SurfaceGripper+sponge hypothesis is killed before
  transport or chain integration.
- Do not chain-integrate SurfaceGripper while the unit probe cannot reach stable
  `Closed` on the sponge.
- Do not return to arbitrary RoArm parent/offset SurfaceGripper search. The next
  valid SurfaceGripper diagnostic must be a single controlled axis/object-size
  test against the canonical rig, or the branch should switch to an explicitly
  authored fixed/D6 constraint unit.
- `get_gripped_objects` / positive gripped-object count is not sufficient
  evidence of attach in this setup; the gate must use `state=Closed` plus low
  drift/hold stability.

Sources:

- `sim_scripts/p7_branch_b_surface_gripper_unit_probe.py`
- `claudedocs/session_20260517_p7_branch_b_surface_gripper_unit_probe.md`
- B200 `/tmp/p7_branch_b_surface_gripper_unit_smoke.{out,err}`

## D024 — B200 Isaac 5.1 runtime requires per-run matching NVIDIA userspace overrides; do not change system symlinks

Evidence:

- Plain B200 `nvidia-smi` failed before Isaac probe logic with an NVML
  driver/library mismatch: userspace NVML reported `580.159`, while
  `/proc/driver/nvidia/version` reported kernel module `580.95.05`.
- The matching library was already present at
  `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`, but
  `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` pointed to
  `libnvidia-ml.so.580.159.03`.
- Vulkan had the same mismatch shape: `/etc/vulkan/icd.d/nvidia_icd.json`
  pointed through `libGLX_nvidia.so.0` (`580.159.03`), while
  `/usr/share/vulkan/icd.d/nvidia_icd.json` directly pointed to
  `/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.95.05`.
- The non-destructive runtime fix was:
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.
- B200 `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out` lines 55-63
  then reported `Driver Version: 580.95.05`, Vulkan, and NVIDIA B200 devices,
  and the script reached the authored probe header at lines 40-41.
- B200 `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.out` lines 40-49
  reached the fixed-constraint probe and created the joint rather than crashing
  in Isaac startup.

Implication:

- Do not edit system NVIDIA symlinks as part of this project.
- For B200 Isaac 5.1 probes in this environment, use the per-run NVML preload
  and Vulkan ICD override above until the cluster image is repaired.
- Treat logs from runs without these overrides as suspect if they crash during
  startup or GLX/Vulkan initialization.

Sources:

- `claudedocs/session_20260517_p7_branch_b_fixed_constraint_unit.md`
- B200 `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.{out,err}`
- B200 `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.{out,err}`

## D025 — Controlled SurfaceGripper axis/object fails commonly; explicit fixed-constraint close/hold/release passes only as a pre-transport unit

Evidence:

- Added `sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py`, a
  controlled canonical-rig diagnostic comparing Isaac Lab's canonical cuboid and
  the project RoArm sponge in the same authored SurfaceGripper setup. It does
  not attach SurfaceGripper to the RoArm chain and does not alter P7 reward,
  release guidance, scripted release, env defaults, or launch defaults.
- B200 `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out` lines 78-79
  verified the canonical rig and the two object cases.
- Lines 80-113 showed the canonical cuboid never reached `Closed`: line 111
  reported `closed_detect_step=-1`, `closed_frac=0.0000`,
  `gripped_positive_frac=1.0000`, `max_drift=0.11145`; line 113 reported
  `success=NO`.
- Lines 114-147 showed the RoArm sponge also never reached `Closed`: line 145
  reported `closed_detect_step=-1`, `closed_frac=0.0000`,
  `gripped_positive_frac=1.0000`, `max_drift=0.34692`; line 147 reported
  `success=NO`.
- Lines 148-149 ended with `diagnosis=COMMON_SURFACE_GRIPPER_FAIL` and
  `SURFACE_AXIS_OBJECT_SUCCESS=NO`.
- Added `sim_scripts/p7_branch_b_fixed_constraint_unit_probe.py`, an explicit
  fixed-joint close/release API unit with a kinematic anchor and RoArm sponge.
  It is CPU-only, has no transport, and is not chain-integrated.
- B200 `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v2.out` lines 49-66 proved
  stable hold (`rel=0`, `drift=0`, `speed_norm=0`), but lines 67-87 showed
  deleting the joint prim alone did not wake release (`release_ok=NO`).
- B200 `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.out` line 49 created the
  joint; lines 50-66 proved 120-step stable attached hold before any transport;
  line 67 removed the joint; lines 68-84 showed release/fall after a wake
  velocity; lines 85-87 reported `hold_ok=YES`, `release_ok=YES`,
  `FIXED_UNIT_SUCCESS=YES`.

Implication:

- SurfaceGripper failure is not just RoArm sponge geometry/material/scale; the
  canonical cuboid fails too in the controlled rig. Do not chain-integrate
  SurfaceGripper.
- `gripped_count` remains invalid as attach evidence without `state=Closed` plus
  low drift/hold stability.
- The fixed-constraint unit is a pre-transport mechanics PASS, not a P7 success
  and not a chain success. The next valid Branch B step is a controlled
  fixed-constraint micro-move/hold/release unit before any RoArm chain
  integration.

Sources:

- `sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py`
- `sim_scripts/p7_branch_b_fixed_constraint_unit_probe.py`
- `claudedocs/session_20260517_p7_branch_b_fixed_constraint_unit.md`
- B200 `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.{out,err}`
- B200 `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v2.{out,err}`
- B200 `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.{out,err}`

## D026 — Static fixed-joint hold is not attach-transport evidence; micro-move kills the current fixed-constraint API

Evidence:

- Added `sim_scripts/p7_branch_b_fixed_constraint_micro_move_probe.py`, a
  pre-chain CPU unit that reuses the explicit fixed-joint close/release idea but
  adds a tiny scripted anchor motion before release. It does not use the RoArm
  chain, SurfaceGripper, P7 training, reward tuning, release guidance, or launch
  default changes.
- B200 `/tmp/p7_branch_b_fixed_constraint_micro_move_smoke.out` lines 40-41
  confirmed the probe scope: CPU, no chain, no transport, no SurfaceGripper, no
  P7 training.
- Line 48 confirmed the intended micro-move:
  `move_delta=([0.020, 0.0, 0.010])`.
- Line 49 confirmed the joint was created and closed at `rel=0.000000`.
- Lines 50-58 confirmed the initial static hold still looked perfect:
  `rel=0.000000`, `drift=0.000000`, `speed_norm=0.000000`.
- Lines 59-71 showed the failure during motion: the anchor moved to
  `[0.020, 0.0, 0.360]`, but the sponge stayed at `[0.0, 0.0, 0.350]`;
  `rel` grew to `0.022361`, and `speed_norm` stayed `0.000000`.
- Lines 72-84 showed post-move hold remained separated at `rel=0.022361`.
- Lines 85-102 showed release still worked after joint removal plus wake
  velocity, so the run reached the intended release phase.
- Line 103 reported `max_move_rel=0.022361`,
  `max_post_move_rel=0.022361`, and `release_drop=0.326499`.
- Lines 104-105 reported the gate failure:
  `close_ok=YES`, `initial_hold_ok=YES`, `move_ok=NO`,
  `post_move_ok=NO`, `release_ok=YES`, `FIXED_MICRO_MOVE_SUCCESS=NO`.

Implication:

- The previous static fixed-constraint unit PASS was only a zero-motion
  close/release smoke. It must not be cited as evidence that the authored
  constraint can transport an attached sponge.
- Do not chain-integrate the current fixed-joint API.
- The next valid Branch B step, if continuing constraints, must redesign and
  retest actuation semantics in isolation: the attached object must move with the
  driven body under a falsifiable micro-move gate before any RoArm chain work.

Sources:

- `sim_scripts/p7_branch_b_fixed_constraint_micro_move_probe.py`
- `claudedocs/session_20260517_p7_branch_b_fixed_constraint_micro_move.md`
- B200 `/tmp/p7_branch_b_fixed_constraint_micro_move_smoke.{out,err}`

## D027 — Dynamic velocity-driven anchor rescues fixed-joint motion, but target tracking is not calibrated yet

Evidence:

- Added `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_probe.py`, a
  pre-chain CPU unit that uses a dynamic, gravity-disabled anchor with mass
  `100.0`, driven by `write_root_velocity_to_sim`. It does not use the RoArm
  chain, SurfaceGripper, P7 training, reward tuning, release guidance, or launch
  default changes.
- Full-command B200 run
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_smoke.out`:
  - lines 40-41 confirmed CPU/no chain/no transport/no SurfaceGripper/no P7
    training;
  - line 48 requested `move_delta=([0.020, 0.0, 0.010])`,
    `move_velocity=([0.050, 0.0, 0.025])`;
  - line 49 closed at `rel=0.000000`;
  - lines 59-71 showed anchor and sponge positions identical through motion with
    `rel=0.000000`;
  - lines 72-84 showed post-move hold remained coupled;
  - lines 85-102 showed release/fall still worked;
  - line 103 reported `max_move_rel=0.000000`,
    `max_post_move_rel=0.000000`, `move_norm=0.022361`,
    `anchor_moved=0.044707`, `sponge_moved=0.044707`,
    `release_drop=0.346430`;
  - lines 104-105 reported all gates YES and
    `FIXED_DYNAMIC_ANCHOR_SUCCESS=YES`.
- Half-command cross-check
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_halfcmd_smoke.out`:
  - line 48 requested `move_delta=([0.010, 0.0, 0.005])`;
  - lines 59-71 again showed anchor and sponge positions identical with
    `rel=0.000000`;
  - line 103 reported `max_move_rel=0.000000`,
    `max_post_move_rel=0.000000`, `move_norm=0.011180`,
    `anchor_moved=0.022349`, `sponge_moved=0.022349`,
    `release_drop=0.336436`;
  - lines 104-105 again reported all gates YES and
    `FIXED_DYNAMIC_ANCHOR_SUCCESS=YES`.

Implication:

- D026's kinematic pose-write failure does not kill fixed-joint semantics in
  general. It kills that actuation method.
- Dynamic, gravity-disabled, velocity-driven anchor actuation is the current
  surviving Branch B constraint mechanism: it can move the attached sponge with
  `rel=0` and release afterward in an isolated unit test.
- It is not chain-ready. The actual displacement is about 2x the command in both
  B200 runs, so target tracking/calibration is unresolved.
- Before any RoArm chain integration, run an isolated target-tracking unit using
  this dynamic-anchor mechanism with an explicit final displacement error gate.

Sources:

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_probe.py`
- `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_constraint.md`
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_smoke.{out,err}`
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_halfcmd_smoke.{out,err}`

## D028 — Closed-loop dynamic-anchor velocity target tracking passes in isolation, but remains pre-chain

Evidence:

- Added `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_target_probe.py`,
  a pre-chain CPU unit that keeps the D027 dynamic, gravity-disabled anchor and
  USD fixed joint, but replaces open-loop velocity duration with a measured
  closed-loop target servo:
  `velocity = clamp(target_kp * (target_pos - anchor_pos), max_cmd_speed)`.
  It does not use the RoArm chain, SurfaceGripper, P7 training, reward tuning,
  release guidance, scripted release variants, or launch default changes.
- Full-command B200 run
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_smoke.out`:
  - lines 40-41 confirmed CPU/no chain/no transport/no SurfaceGripper/no P7
    training;
  - line 48 requested target delta `([0.020, 0.0, 0.010])`;
  - line 49 closed at `rel=0.000000`;
  - lines 59-68 showed closed-loop motion with anchor and sponge positions
    identical and `rel=0.000000`;
  - line 83 reported post-hold `final_anchor_target_error=0.001426` and
    `final_sponge_target_error=0.001426`;
  - line 102 reported `max_move_rel=0.000000`,
    `max_post_move_rel=0.000000`, `target_error_threshold=0.003000`,
    `release_drop=0.335825`;
  - lines 103-104 reported all gates YES and
    `FIXED_DYNAMIC_ANCHOR_TARGET_SUCCESS=YES`.
- Half-command B200 cross-check
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_halfcmd_smoke.out`:
  - line 48 requested target delta `([0.010, 0.0, 0.005])`;
  - line 81 reported post-hold `final_anchor_target_error=0.001429` and
    `final_sponge_target_error=0.001429`;
  - line 100 reported `max_move_rel=0.000000`,
    `max_post_move_rel=0.000000`, `target_error_threshold=0.003000`,
    `release_drop=0.330823`;
  - lines 101-102 reported all gates YES and
    `FIXED_DYNAMIC_ANCHOR_TARGET_SUCCESS=YES`.

Implication:

- D027's target-calibration caveat is resolved only for this closed-loop isolated
  dynamic-anchor unit. The open-loop dynamic-anchor probe remains overshooting
  evidence and should not be cited as calibrated target tracking.
- This is not P7 success and not RoArm chain integration evidence.
- Before chain integration, the next Branch B work must still be pre-chain:
  define and test the smallest controller/interface mapping from target-tracked
  anchor motion toward a future TCP/anchor command surface.

Sources:

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_target_probe.py`
- `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_target_tracking.md`
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_smoke.{out,err}`
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_halfcmd_smoke.{out,err}`

## D029 — Mock-TCP interface wrapper passes pre-chain, but real RoArm integration risks remain untested

Evidence:

- Added `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_interface_probe.py`,
  a pre-chain CPU unit that wraps the D028 target-tracked dynamic-anchor mechanism
  in a mock TCP command surface. The interface mapping is
  `anchor_target = tcp_target + tcp_to_anchor_offset`. It does not use the RoArm
  chain, IK, SurfaceGripper, P7 training, reward tuning, release guidance,
  scripted release variants, or launch default changes.
- Default-offset B200 run
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_smoke.out`:
  - lines 40-41 confirmed CPU/no chain/no transport/no SurfaceGripper/no P7
    training;
  - line 48 closed at `rel=0.000000` with three waypoints;
  - lines 58, 75, and 93 reported `transform_error=0.000000`;
  - lines 67, 85, and 102 reported waypoint target-stop errors `0.001411`,
    `0.001464`, and `0.001394`;
  - line 128 reported `max_move_rel=0.000000`, `max_hold_rel=0.000000`,
    `max_final_anchor_target_error=0.001468`,
    `max_final_sponge_target_error=0.001468`, `release_drop=0.338178`;
  - lines 129-130 reported all gates YES and
    `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.
- Nonzero-offset B200 cross-check
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_offset_smoke.out`:
  - line 48 activated `tcp_to_anchor_offset=([0.015, 0.0, -0.010])`;
  - lines 58, 75, and 93 showed mock TCP targets distinct from anchor targets
    while `transform_error=0.000000`;
  - line 128 repeated `max_move_rel=0.000000`, `max_hold_rel=0.000000`,
    max final target errors `0.001468`, and `release_drop=0.338178`;
  - lines 129-130 again reported all gates YES and
    `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.

Implication:

- A thin mock-TCP command wrapper around the target-tracked dynamic anchor is
  viable in isolation.
- This still does not validate RoArm chain integration. IK, articulation dynamics,
  controller latency, TCP estimation, contact, and attach/release timing remain
  untested.
- Do not present this as P7 success or chain-readiness. Any movement from mock TCP
  to actual RoArm chain integration needs an explicit transition decision and a
  new falsifiable gate.

Sources:

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_interface_probe.py`
- `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_interface_probe.md`
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_smoke.{out,err}`
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_offset_smoke.{out,err}`

## D030 — Mock chain-command contract passes pre-chain; remaining blocker is real chain signal generation

Evidence:

- Added `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`,
  a pre-chain CPU unit that wraps the D029 mock-TCP interface in a stricter
  command/state contract. It does not use the RoArm chain, IK, SurfaceGripper, P7
  training, reward tuning, release guidance, scripted release variants, or launch
  default changes.
- The contract allows `CLOSE` only before attach/release, `MOVE`/`HOLD` only
  while attached, and `RELEASE` only after target-reached state.
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.out`:
  - lines 40-41 confirmed CPU/no chain/no transport/no SurfaceGripper/no P7
    training;
  - line 42 showed negative checks all passed:
    move-before-close, release-before-close, double-close, early-release, and
    move-after-release were all rejected;
  - line 49 accepted `CLOSE` with `rel=0.000000`, joint exists, nonzero
    `tcp_to_anchor_offset=([0.015, 0.0, -0.010])`, and `waypoints=3`;
  - lines 59, 76, and 94 accepted the three `MOVE` commands with
    `transform_error=0.000000`;
  - lines 68, 86, and 103 reported target-stop errors `0.001411`,
    `0.001464`, and `0.001394`;
  - line 111 accepted `RELEASE` after target-reached state and removed the joint;
  - line 129 reported `contract_negative_ok=YES`, `max_attached_rel=0.000000`,
    `max_final_anchor_target_error=0.001468`,
    `max_final_sponge_target_error=0.001468`, and `release_drop=0.338178`;
  - lines 130-131 reported all gates YES and
    `DYNAMIC_ANCHOR_CHAIN_CONTRACT_SUCCESS=YES`.

Implication:

- The isolated Branch B constraint path now has a tested minimal command contract.
- The remaining problem is no longer isolated constraint coupling, target
  tracking, TCP-offset mapping, or command ordering. The remaining problem is
  whether the **actual RoArm chain** can generate reliable TCP/IK/timing signals
  that satisfy this contract under articulation dynamics and contact.
- This is not P7 success and not chain-readiness. Do not integrate into the RoArm
  chain implicitly; any chain transition needs explicit approval and a new narrow
  falsifiable gate.

Sources:

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`
- `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`
- B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.{out,err}`

## D031 — RoArm planner kinematics can satisfy the command contract only with explicit TCP resampling; raw waypoint gaps are too coarse

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_contract_dryrun_probe.py`, a
  local/numpy-only chain-side diagnostic. It imports the existing
  `TrajectoryPlanner` and FK/IK code, but does not run Isaac, insert constraint
  prims, use SurfaceGripper, run P7 training, or edit env/train/chain defaults.
- Local `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe.out` and B200
  `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe_b200.out` matched:
  - line 2 confirmed `chain_side_only=YES`, `isaac_chain_integration=NO`,
    `constraint_prim_insertion=NO`, `surface_gripper=NO`, `p7_training=NO`,
    `env_default_edits=NO`, and `chain_defaults_edits=NO`;
  - lines 12-17 showed all six planner waypoints under the `0.003m` FK TCP
    error gate, with max waypoint FK error `0.000551m`;
  - lines 19-23 showed the raw planner waypoint gaps fail the conservative
    `0.010m` command-step gate: `0.073074m`, `0.018075m`, and `0.022913m`
    are over the gate;
  - lines 24-29 showed a resampled contract stream with `CLOSE`, three `MOVE`
    commands, `HOLD`, and `RELEASE`; all three MOVE commands had
    `ik_converged=YES`;
  - line 30 reported `contract_move_steps=3`,
    `max_contract_tcp_step_m=0.007648`,
    `max_contract_fk_error_m=0.000649`, and
    `final_transport_target_error_m=0.000231`;
  - lines 31-32 reported contract-stream gates YES and
    `ROARM_CHAIN_CONTRACT_DRYRUN_SUCCESS=YES`, while `raw_planner_gap_ok=NO`.

Implication:

- The current RoArm planner/kinematics are not the immediate blocker for a
  contract-compatible TCP stream if an explicit chain-side command/timing layer
  resamples attached transport into small TCP steps.
- The existing raw planner targets must not be treated as directly compatible
  with the D030 command contract under a `0.010m` step gate.
- This remains pre-integration evidence only. It does not validate articulation
  dynamics, controller latency, TCP estimation in Isaac, contact, or
  attach/release timing. Do not integrate fixed/dynamic constraints into the
  RoArm chain without explicit user approval and a new falsifiable gate.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_contract_dryrun_probe.py`
- `claudedocs/session_20260517_p7_branch_b_roarm_chain_contract_dryrun.md`
- Local `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe_b200.{out,err}`

## D032 — Chain-side TCP resampling needs safety margin; exact 10mm spacing can still violate a 10mm realized-step gate

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_timing_resample_probe.py`, a
  local/numpy-only dry-run that validates HOME→grasp pre-close TCP resampling,
  attached grasp→transport resampling, true final stream FK error, command
  ordering, and release-after-target timing. It does not run Isaac, insert
  constraint prims, use SurfaceGripper, run P7 training, or edit env/train/chain
  defaults.
- Local and B200 outputs were sha256-identical for both the no-margin failure
  and the conservative-resampling pass.
- No-margin run (`--resample_fraction 1.0`) failed under the unchanged `0.010m`
  realized-step gate:
  - line 3 showed `max_tcp_step_m=0.010000` and `resample_fraction=1.000`;
  - line 31 showed a pre-close `PRE_MOVE` with `tcp_step_m=0.010351` and
    `ok=NO`;
  - line 69 reported `max_preclose_tcp_step_m=0.010351`;
  - lines 70-71 reported `preclose_stream_ok=NO`, `command_order_ok=NO`, and
    `ROARM_CHAIN_TIMING_RESAMPLE_SUCCESS=NO`.
- Conservative run (`--resample_fraction 0.9`, default) passed:
  - line 3 showed `max_tcp_step_m=0.010000` with `resample_fraction=0.900`;
  - lines 11-16 still showed raw planner gaps fail, with
    `raw_max_gap_m=0.211271` later reported on line 73;
  - line 65 accepted `CLOSE` only after target reached;
  - lines 67-69 showed attached `MOVE` commands with IK convergence YES and
    max attached TCP step `0.007691`;
  - lines 71-72 accepted `HOLD` and `RELEASE` after target reached;
  - line 73 reported `preclose_cmds=38`, `attached_cmds=3`,
    `max_preclose_tcp_step_m=0.009525`,
    `max_attached_tcp_step_m=0.007691`,
    `max_preclose_fk_error_m=0.000997`,
    `max_attached_fk_error_m=0.000655`,
    `transport_final_error_m=0.000655`, and zero IK failures;
  - lines 74-75 reported all gates YES and
    `ROARM_CHAIN_TIMING_RESAMPLE_SUCCESS=YES`.

Implication:

- Future chain-side command/timing logic must not emit raw planner waypoints
  directly, and should not target exactly the maximum allowed TCP step. It needs
  conservative sub-step spacing to absorb realized FK/IK error while preserving
  the external `0.010m` gate.
- This is still only pre-integration chain-side evidence. It does not validate
  articulation dynamics, controller latency, contact, attach/release timing in
  Isaac, or dynamic-anchor/fixed-constraint insertion in the RoArm chain.
- Do not present this as P7 success or chain-readiness. Constraint integration
  still requires explicit user approval and a new falsifiable gate.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_timing_resample_probe.py`
- `claudedocs/session_20260517_p7_branch_b_roarm_chain_timing_resample.md`
- Local `/tmp/p7_branch_b_roarm_chain_timing_resample_probe*.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_timing_resample_probe*_b200.{out,err}`

## D033 — Real RoArm articulation timing requires realized-TCP gating; one-step command assumptions are unsafe

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`, an
  Isaac/RoArm articulation-only diagnostic around the conservative command
  stream. It keeps the env sponge far away, so `CLOSE` and `RELEASE` are
  marker/gripper timing checks only. It does not insert constraint prims,
  integrate fixed/dynamic constraints, attach SurfaceGripper, run P7 training,
  or edit env/train/chain defaults.
- B200 `/tmp/p7_branch_b_roarm_chain_dynamics_timing_probe_b200.out`:
  - line 40 confirmed articulation-only/no-constraint/no-SurfaceGripper/no-P7
    scope and `release_marker_only=YES`;
  - line 42 reused the conservative stream:
    `events_total=44`, `pre_move_cmds=38`, `move_cmds=3`,
    `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`;
  - line 70 reported sim HOME TCP vs analytic FK error `0.000870m`, below the
    `0.003m` gate;
  - lines 71-85 showed sampled events reached under gated execution, including
    `CLOSE` in 16 sim steps, MOVE marker events in 9/10/10 steps, `HOLD` in 2,
    and `RELEASE` marker in 3;
  - line 86 reported aggregate `total_sim_steps=311`, `max_event_steps=16`,
    `event_timeouts=0`, `max_first_step_target_error_m=0.009291`,
    `one_step_target_ok=NO`, `max_final_target_error_m=0.002705`,
    `max_sim_tcp_step_m=0.001947`, `max_cache_fresh_delta_m=0.000000`,
    `grasped_seen=NO`, and `release_gripper_open_ok=YES`;
  - lines 87-88 reported controller/target/sim-step/cache/no-attach/release
    marker gates YES and `ROARM_CHAIN_DYNAMICS_TIMING_SUCCESS=YES`, while
    `one_step_target_ok=NO`.

Implication:

- The conservative command stream is not sufficient as a blind one-command-per-
  sim-step schedule. A future chain-side executor must wait on realized TCP (or
  an equivalent measured state gate) before advancing `PRE_MOVE`, `CLOSE`,
  `MOVE`, `HOLD`, or `RELEASE`.
- In this no-contact/no-attach diagnostic, the real articulation/controller can
  follow the stream within the `0.003m` target gate and the realized per-sim-step
  TCP motion stayed below `0.010m`.
- This still does not validate contact, object attachment, release physics, or
  constraint insertion. It is not P7 success and not constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`
- `claudedocs/session_20260517_p7_branch_b_roarm_chain_dynamics_timing.md`
- B200 `/tmp/p7_branch_b_roarm_chain_dynamics_timing_probe_b200.{out,err}`

## D034 — Current env `_grasped` kinematic latch is not a stable post-close handoff surface

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`,
  a narrow Isaac/RoArm diagnostic that executes only `PRE_MOVE* -> CLOSE`, then
  holds the same grasp pose briefly. It does not insert constraint prims,
  integrate fixed/dynamic constraints, attach SurfaceGripper, execute attached
  transport, run release, run P7 training, or edit env/train/chain defaults.
- `roarm_rl/roarm_stack_env.py` lines 1184-1195 show `_grasped` is latched from
  a distance+gripper threshold condition. Lines 1216-1236 show the current
  `_update_grasp_attach` writes the sponge root pose to the TCP and, by default,
  zeroes velocity. This is an env kinematic pose-write boundary, not authored
  constraint physics.
- B200 `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_b200.out`:
  - line 41 confirmed post-close latch-boundary scope only, with no constraint
    insertion, no fixed/dynamic integration, no SurfaceGripper, no attached
    transport, no release marker, and explicitly
    `attach_physics_validated=NO`, `release_physics_validated=NO`;
  - line 43 confirmed the conservative source stream was truncated before MOVE:
    `source_events_total=44`, `executed_events=39`, `move_cmds_executed=0`,
    `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`;
  - line 81 confirmed CLOSE still reached cleanly in 15 steps with
    `gripper_q_deg=+23.02`, `d_tcp_sponge_m=0.023599`,
    `sponge_xy_drift_m=0.000005`, `min_upright_z=1.000000`, and latch seen;
  - line 82 showed the latch step itself was quiet:
    `pose_jump_m=0.000000`, `d_tcp_sponge_jump_m=0.000000`,
    `quat_angle_deg=0.000`, latch step and gripper-threshold step both `275`;
  - line 83 killed the first stationary post-latch hold step:
    `target_error_m=0.015684`, `tcp_step_m=0.016131`,
    `pose_drift_m=0.017552`, `xy_drift_m=0.006564`,
    `sponge_speed_mps=1.696947`, `sponge_ang_speed_rps=17.195574`,
    `quat_angle_deg=21.267`, `early_kill=YES`;
  - lines 84-86 reported `hold_early_kill=YES`, `target_error_ok=NO`,
    `sim_step_ok=NO`, `post_latch_hold_ok=NO`, and
    `ROARM_POST_CLOSE_LATCH_BOUNDARY_SUCCESS=NO`.

Implication:

- The previous passive-contact close result remains useful only up to the latch
  marker. The first post-latch stationary hold shows the current kinematic attach
  boundary is unstable under the RoArm articulation/contact setup.
- Do not proceed from CLOSE into attached transport using the current env
  `_grasped` kinematic attach boundary as a valid handoff surface.
- This is not P7 success, not constraint integration, and not attach/release
  physics validation. Future work must either analyze/redesign this env boundary
  in isolation or, with explicit user approval, test an authored constraint path
  under a separate falsifiable gate.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
- `claudedocs/session_20260518_p7_branch_b_post_close_latch_boundary.md`
- B200 `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_b200.{out,err}`

## D035 — Post-latch failure is driven by env pose-write attach, not by quaternion or velocity mode alone

Evidence:

- Extended `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
  to run an attribution matrix without editing env/train/chain defaults:
  `attach_quat_mode`, `attach_velocity_mode`, and a marker-only
  `--disable_attach_posewrite` control that no-ops `_update_grasp_attach` inside
  the diagnostic only.
- All runs kept the same strict scope: no constraint prim insertion, no
  fixed/dynamic constraint integration, no SurfaceGripper, no attached
  transport, no release marker, no P7 training, and no default edits.
- B200 pose-write enabled variants all failed on the first stationary
  post-latch hold step:
  - default `preserve+zero`, `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_default_b200.out`
    lines 83-86: `target_error_m=0.015684`, `tcp_step_m=0.016131`,
    `post_latch_hold_ok=NO`;
  - `preserve+keep`, `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_keep_b200.out`
    lines 83-86: `target_error_m=0.013359`, `tcp_step_m=0.013831`,
    `post_latch_hold_ok=NO`;
  - `identity+zero`, `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_identity_zero_b200.out`
    lines 83-86: `target_error_m=0.015831`, `tcp_step_m=0.016265`,
    `post_latch_hold_ok=NO`;
  - `identity+keep`, `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_identity_keep_b200.out`
    lines 83-86: `target_error_m=0.012996`, `tcp_step_m=0.013450`,
    `post_latch_hold_ok=NO`.
- The marker-only/no-posewrite control passed the same stationary hold:
  `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_no_posewrite_b200.out`
  lines 83-88 show steps 1-20 with negligible drift and no early kill; line 89
  reports `post_latch_hold_steps_done=20`, `hold_max_target_error_m=0.000817`,
  `max_sim_tcp_step_m=0.001947`, `hold_max_pose_drift_m=0.000000`,
  `hold_max_speed_mps=0.000604`; lines 90-91 report `post_latch_hold_ok=YES`
  and `ROARM_POST_CLOSE_LATCH_BOUNDARY_SUCCESS=YES`.

Implication:

- The proximate trigger for the D034 post-latch failure is the env kinematic
  `_update_grasp_attach` pose-write to TCP, not merely velocity zeroing or
  quaternion preservation.
- Do not try to rescue current RoArm chain handoff by only toggling
  `attach_quat_mode` or `attach_velocity_mode`; those variants remain killed.
- The no-posewrite pass is only a marker-only negative control. It does not
  validate attach physics, release physics, attached transport, SurfaceGripper,
  or constraint insertion.
- The next valid work is to redesign/test a local handoff model in isolation, or
  wait for explicit user approval before testing authored constraint insertion.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
- `claudedocs/session_20260518_p7_branch_b_post_close_latch_boundary.md`
- B200 `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_{default,keep,identity_zero,identity_keep,no_posewrite}_b200.{out,err}`

## D036 — CLOSE handoff failure is center-snap geometry; offset-preserving pose-write is only a local stationary candidate

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`, a
  diagnostic-local CLOSE handoff matrix. It executes only `PRE_MOVE* -> CLOSE`
  and stationary post-close hold. It does not insert constraint prims, integrate
  fixed/dynamic constraints, attach SurfaceGripper, execute attached transport,
  run release, run P7 training, or edit env/train/chain defaults.
- B200 v2 logs
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_{posewrite_tcp,marker_only,delayed_posewrite,oneshot_align,offset_preserve_posewrite}_v2_b200.out`
  line 41 confirm the strict scope and explicitly keep
  `attach_physics_validated=NO`, `release_physics_validated=NO`, and
  `claim_attach_success=NO`.
- Line 43 in each v2 stdout confirms the same conservative source stream was
  truncated before MOVE: `source_events_total=44`, `executed_events=39`,
  `pre_move_cmds=38`, `move_cmds_executed=0`, `raw_max_gap_m=0.211271`,
  `raw_gap_ok=NO`.
- Line 82 in the baseline and offset-preserve v2 logs again showed the latch
  marker step itself was quiet: `pose_jump_m=0.000000`,
  `d_tcp_sponge_jump_m=0.000000`, `quat_angle_deg=0.000`, and latch/threshold
  global step `275`.
- Current TCP-center pose-write baseline
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_posewrite_tcp_v2_b200.out`
  line 83 reproduced D034: `target_error_m=0.015684`,
  `tcp_step_m=0.016131`, `pose_drift_m=0.017552`,
  `sponge_speed_mps=1.696947`, `quat_angle_deg=21.267`,
  `early_kill=YES`; lines 84-86 reported `post_latch_hold_ok=NO` and success
  `NO`.
- Marker-only
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_marker_only_v2_b200.out`
  lines 83-88 passed sampled hold steps; line 89 reported
  `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
  `hold_max_pose_drift_m=0.000000`; lines 90-91 reported
  `post_latch_hold_ok=YES` and success `YES`.
- Delayed TCP-center pose-write
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_delayed_posewrite_v2_b200.out`
  lines 83-85 passed the first 3 stationary env steps with no pose-write
  failure. Line 86 failed as soon as the center-snap pose-write began:
  `target_error_m=0.015686`, `tcp_step_m=0.016133`,
  `pose_drift_m=0.017553`, `quat_angle_deg=21.266`, `early_kill=YES`.
  Lines 87-89 reported `post_latch_hold_ok=NO` and success `NO`.
- One-shot TCP-center align
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_oneshot_align_v2_b200.out`
  line 83 still failed the first hold step, though less violently:
  `target_error_m=0.005097`, `pose_drift_m=0.005682`,
  `sponge_speed_mps=0.823018`, `sponge_ang_speed_rps=16.806622`,
  `quat_angle_deg=7.080`; lines 84-86 reported `post_latch_hold_ok=NO`.
- Continuous TCP-offset-preserving pose-write
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_offset_preserve_posewrite_v2_b200.out`
  lines 83-88 passed sampled hold steps; line 89 reported
  `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
  `hold_max_pose_drift_m=0.000000`, `hold_max_offset_error_m=0.000001`,
  `hold_max_speed_mps=0.000869`, `posewrite_calls=40`, and
  `offset_initialized=YES`; lines 90-91 reported `post_latch_hold_ok=YES` and
  success `YES`.
- All v2 stderr files had only the known cpufreq/NVML/Fabric warnings on lines
  1-4 and no Python traceback.

Implication:

- The local post-CLOSE failure is not merely "any pose-write" and not merely
  quaternion or velocity mode. The killed operation is specifically snapping the
  sponge center to the TCP after latch.
- Waiting before the same TCP-center snap only delays the same failure, and a
  one-shot center snap is still too disruptive for the stationary hold gate.
- Preserving the latch-time TCP-to-sponge offset is the first local kinematic
  handoff model that survives the stationary post-close hold while still using
  continuous pose-write. This is a candidate for further isolated handoff
  diagnostics only.
- Do not claim object attachment physics, attached transport, release physics,
  SurfaceGripper success, or constraint integration from the marker-only or
  offset-preserving stationary PASS. No MOVE/transport or RELEASE was executed,
  and no constraint prim was inserted.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
- `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_{posewrite_tcp,marker_only,delayed_posewrite,oneshot_align,offset_preserve_posewrite}_v2_b200.{out,err}`

## D037 — Post-CLOSE target buffers are delivered, but the 5deg grasp-pose nudge is not realized even before CLOSE

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`,
  a diagnostic-local target-delivery probe. It wraps
  `_robot.set_joint_position_target()`, snapshots Articulation target fields, and
  compares the same 5deg shoulder nudge before CLOSE, after CLOSE/latch through
  `env.step(null_action)`, and after CLOSE/latch through direct
  `set_joint_position_target()+sim.step`. It does not insert constraint prims,
  integrate fixed/dynamic constraints, attach SurfaceGripper, execute attached
  transport, go to the transport target, run release, run P7 training, or edit
  env/train/chain defaults.
- B200 v3
  `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.out`:
  - line 41 confirms the strict diagnostic scope and explicitly reports
    `attach_physics_validated=NO`, `release_physics_validated=NO`, and
    `claim_attach_success=NO`;
  - line 43 confirms the run remains pre-transport:
    `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`;
  - line 44 proves the watched target is a nonzero 5deg shoulder nudge with
    `expected_tcp_delta_m=0.024271`, and the before-CLOSE comparison uses
    `target_q_open_deg` with gripper open;
  - lines 83-85 show the before-CLOSE target reaches
    `_robot.set_joint_position_target()` and Articulation target fields including
    `joint_pos_target` with best diff `0.00000004rad`;
  - line 94 reports the before-CLOSE open-gripper nudge still is not realized:
    `final_joint_error_max_deg=5.046748`, `final_target_tcp_error_m=0.023923`,
    `tcp_target_reduced=NO`, `joint_error_reduced=NO`,
    `target_realized=NO`, and `grasped=NO`;
  - lines 110-112 show the post-latch env-step target is also delivered into
    `_robot.set_joint_position_target()` and Articulation target fields with best
    diff `0.00000004rad`;
  - line 121 reports post-latch env-step still fails to realize the nudge:
    `final_joint_error_max_deg=5.044317`, `final_target_tcp_error_m=0.023842`,
    `target_realized=NO`;
  - line 134 reports direct set+sim-step after latch also does not rescue:
    `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000004`,
    `max_realized_tcp_delta_m=0.000098`,
    `final_joint_error_max_deg=5.046912`, and `target_realized=NO`;
  - line 135 aggregates `before_target_realized=NO`,
    `after_env_target_realized=NO`, `after_direct_target_realized=NO`,
    `before_vs_after_split=NO`, `direct_rescues=NO`, and
    `general_grasp_pose_target_delivery_blocker=YES`.
- B200 stderr lines 1-4 contain only the known cpufreq/NVML/Fabric warnings and
  no Python traceback.

Implication:

- Do not describe the current blocker as specifically offset-preserve moving
  failure. Offset-preserve moving behavior is still untested because the robot
  did not realize the commanded nudge.
- Do not describe the current blocker as only post-CLOSE/latch-specific. The
  same 5deg shoulder nudge at the grasp pose with gripper open and `_grasped=NO`
  also fails to realize, despite target delivery into Articulation buffers.
- The next valid diagnostic should move earlier in the approach/HOME/high
  sequence with the same target-delivery instrumentation to isolate whether this
  is a local grasp-pose/drive/limit issue, a controller-state issue, or a broader
  command-realization issue.
- This is not P7 success, not attach physics, not release physics, not attached
  transport, and not constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
- `claudedocs/session_20260518_p7_branch_b_post_latch_target_delivery.md`
- B200 `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.{out,err}`

## D038 — The 5deg shoulder-nudge blocker is local to the grasp-before-CLOSE pose, not broad articulation target realization

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`,
  a diagnostic-local approach-stage target-delivery probe. It compares the same
  +5deg shoulder nudge at settled HOME, early PRE_MOVE, high, hover, and
  grasp-before-CLOSE/open-gripper stages. It does not insert constraint prims,
  integrate fixed/dynamic constraints, attach SurfaceGripper, execute attached
  transport, go to the transport target, run release, run P7 training, or edit
  env/train/chain defaults.
- B200 v2
  `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.out`:
  - line 41 confirms the strict diagnostic scope and explicitly reports
    `attach_physics_validated=NO`, `release_physics_validated=NO`, and
    `claim_attach_success=NO`;
  - line 43 confirms the run remains pre-transport:
    `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`;
  - line 72 reports `action_scale=0.100000`, `null_action_max_abs=0.000000`,
    and the articulation soft limits;
  - line 75 proves the HOME nudge is nonzero and within limits
    (`expected_tcp_delta_m=0.035677`, limits OK), and line 87 reports
    `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000004`,
    `final_nudge_joint_error_deg=0.109396`, and `target_realized=YES`;
  - line 103 proves the early PRE_MOVE nudge is nonzero and within limits
    (`expected_tcp_delta_m=0.035105`, limits OK), and line 115 reports
    `target_realized=YES` with final nudge error `0.106804deg`;
  - line 131 proves the high-pose nudge is nonzero and within limits
    (`expected_tcp_delta_m=0.023524`, limits OK), and line 143 reports
    `target_realized=YES` with final nudge error `0.084780deg`;
  - line 159 proves the hover-pose nudge is nonzero and within limits
    (`expected_tcp_delta_m=0.023692`, limits OK), and line 171 reports
    `target_realized=YES` with final nudge error `0.105476deg`;
  - line 187 proves the grasp-before-CLOSE/open-gripper nudge is nonzero and
    within soft/analytic limits (`expected_tcp_delta_m=0.024271`, limits OK);
  - line 199 reports the grasp-before-CLOSE env-step failure despite target
    delivery: `set_target_seen=YES`,
    `best_data_target_attr_diff_rad=0.00000004`,
    `final_target_tcp_error_m=0.023947`,
    `final_nudge_joint_error_deg=5.042476`,
    `tcp_target_reduced=NO`, `nudge_joint_error_reduced=NO`,
    `target_realized=NO`, and `grasped=NO`;
  - line 211 reports direct set+sim-step at the same grasp-before-CLOSE pose
    also fails: `set_target_seen=YES`, `max_realized_tcp_delta_m=0.000108`,
    `final_target_tcp_error_m=0.023927`,
    `final_nudge_joint_error_deg=5.044027`, and `target_realized=NO`;
  - lines 213-214 aggregate the split:
    `env_realized_stages=['settled_home', 'early_pre_move', 'high', 'hover']`,
    `env_failed_stages=['grasp_before_close_open']`,
    `direct_rescue_stages=[]`, `home_high_realize_grasp_fails=YES`,
    `broader_command_realization_blocker=NO`,
    `local_grasp_pose_only_blocker=YES`, and `latch_seen=NO`.
- B200 stderr lines 1-4 contain only the known cpufreq/NVML/Fabric warnings and
  no Python traceback.

Implication:

- D037 is refined: target delivery and realization work earlier in the approach
  sequence. Do not call the current blocker a broad articulation
  target-realization failure.
- Do not call the current blocker post-latch-only. The failure persists before
  CLOSE at the grasp pose with gripper open and `_grasped=NO`.
- Direct set+sim-step does not rescue the grasp-before-CLOSE pose, so env-step
  overwrite/null-action remains unlikely for this local failure.
- The next valid diagnostic should inspect local grasp-pose causes: drive/limit
  behavior, contact/proximity effects, controller state around the low grasp pose,
  or whether a slightly different pre-grasp/grasp staging pose avoids the local
  command-realization dead zone. This still must remain pre-integration unless
  the user explicitly approves constraint insertion.
- This is not P7 success, not attach physics, not release physics, not attached
  transport, and not constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
- `claudedocs/session_20260518_p7_branch_b_approach_target_delivery.md`
- B200 `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.{out,err}`
