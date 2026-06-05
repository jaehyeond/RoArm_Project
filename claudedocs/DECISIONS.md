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

## D039 — The grasp-pose shoulder-nudge dead zone is contact/proximity-shaped, not a pure low-pose drive failure

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`, a
  diagnostic-local pre-integration matrix. It compares the same +5deg shoulder
  nudge around the grasp pose with nominal sponge, sponge far/no-contact, higher
  local z offsets, and sub-threshold partial gripper close. It does not insert
  constraint prims, integrate fixed/dynamic constraints, attach SurfaceGripper,
  execute attached transport, go to the transport target, run release, run P7
  training, tune P7, or edit env/train/chain defaults.
- B200 default log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_b200.out`:
  - line 41 confirms the strict scope and explicitly reports
    `attach_physics_validated=NO`, `release_physics_validated=NO`, and
    `claim_attach_success=NO`;
  - line 43 confirms the run remains pre-transport:
    `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`;
  - line 72 reports `action_scale=0.100000`, `null_action_max_abs=0.000000`,
    soft limits, and arm/gripper drive fields;
  - line 85 reports nominal sponge/open-gripper failure despite target delivery:
    `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000004`,
    `max_realized_tcp_delta_m=0.001144`,
    `final_target_tcp_error_m=0.023952`,
    `final_shoulder_error_deg=5.035108`, `target_realized=NO`, and `_grasped=NO`;
  - the same line shows the target is a proximity/contact-risk target:
    `start_target_tcp_minus_sponge_top_m=-0.022771` and
    `start_target_xy_inside_sponge_aabb=YES`;
  - line 97 reports direct set+sim-step also fails for nominal sponge/open
    gripper, so env-step overwrite is not the cause;
  - line 110 reports the same robot q and target realizes when the sponge is far:
    `max_realized_tcp_delta_m=0.025811`,
    `final_target_tcp_error_m=0.000850`,
    `final_shoulder_error_deg=0.114004`, `target_realized=YES`,
    `start_d_tcp_sponge_m=0.676404`, and
    `start_target_xy_inside_sponge_aabb=NO`;
  - line 122 reports direct set also realizes with sponge far;
  - lines 135, 160, and 185 show +3mm, +6mm, and +12mm nominal-sponge variants
    still fail or remain insufficient by the diagnostic target-realized gate;
  - lines 210 and 222 show sub-threshold partial close at the nominal pose does
    not rescue;
  - lines 223-224 aggregate `env_realized_conditions=['far_sponge_open']`,
    nominal and +3/+6/+12mm/partial-close failures,
    `sponge_far_realizes_nominal_fails=YES`,
    `direct_set_also_fails_nominal=YES`, and no attach/release physics claim.
- B200 high-z cross-check
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zhi_b200.out`:
  - lines 85/97 repeat nominal failure and lines 110/122 repeat far-sponge
    realization;
  - lines 135/147 show +18mm nominal-sponge env/direct realization even though
    target TCP is still slightly below sponge top
    (`start_target_tcp_minus_sponge_top_m=-0.005662`);
  - lines 160/172 show +24mm env/direct realization with target just above
    sponge top (`start_target_tcp_minus_sponge_top_m=0.000859`);
  - lines 185/197 show +30mm env/direct realization with target clearly above
    sponge top (`start_target_tcp_minus_sponge_top_m=0.006679`);
  - lines 210/222 repeat partial-close failure;
  - lines 223-224 aggregate that far sponge and +18/+24/+30mm realize, nominal and
    partial close fail, `higher_z_realizes_nominal_fails=YES`, and no direct-rescue
    split appears.
- Both B200 stderr files have only the known cpufreq/NVML/Fabric warnings on
  lines 1-4 and no Python traceback.

Implication:

- D038 is refined again: the blocker is local to nominal sponge/grasp geometry
  and is contact/proximity-shaped. The same robot q and target realize when the
  sponge is moved far away, so this is not a broad articulation target-realization
  failure and not a pure low-pose drive/controller limit.
- Direct set+sim-step mirrors env-step in the decisive comparisons, so
  env-step/null-action overwrite is not the cause of the nominal failure.
- Do not treat the far-sponge or higher-z realization as attach, transport,
  release, SurfaceGripper, or constraint evidence. No attach physics, transport
  target, release, or constraint insertion was executed.
- Next valid work should remain pre-integration and isolate the contact/proximity
  boundary between +12mm fail/insufficient and +18mm pass, or sweep horizontal
  proximity, before any approved chain integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
- `claudedocs/session_20260518_p7_branch_b_grasp_pose_deadzone.md`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zhi_b200.{out,err}`

## D040 — The grasp-pose boundary is near +13mm under the current gate; horizontal offset evidence is posture-confounded

Evidence:

- Extended `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
  to support decimal z-offset labels, diagnostic-local horizontal sponge offsets,
  and explicit start sponge xyz/top/dx/dy metrics. Latest md5:
  `fa3c1445ac692cef85ab6a32cc8d6838`. This remains pre-integration only: no
  constraints, no SurfaceGripper, no transport target, no release, no P7
  training/tuning, and no env/train/chain default edits.
- B200 fine z log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zfine_b200.out`:
  - lines 85/97 repeat nominal sponge/open failure;
  - lines 135/147 show +13mm env/direct pass the diagnostic realization gate;
  - lines 160/172, 185/197, 210/222, and 235/247 show +14/+15/+16/+17mm also
    pass the same gate;
  - lines 273-274 aggregate far sponge and +13 through +17mm realization, with
    nominal and partial-close failures and no direct-rescue split.
- B200 micro z log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zmicro_b200.out`:
  - line 43 records z offsets `[12.0, 12.25, 12.5, 12.75, 13.0]`;
  - lines 136/148, 161/173, 186/198, and 211/223 show +12.0/+12.25/+12.5/+12.75mm
    env/direct failures;
  - lines 236/248 show +13.0mm env/direct pass the current reduction gate;
  - lines 274-275 aggregate that only far sponge and +13mm realize among the
    micro-z conditions.
- The +13mm pass is not exact target convergence. In the micro-z log, line 236
  still has `final_target_tcp_error_m=0.011054` and
  `final_shoulder_error_deg=2.224901`; it passes because the diagnostic gate is
  based on meaningful error reduction, not final 3mm convergence.
- B200 wide xy log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_xy_b200.out`:
  - line 43 records x offsets `[-35,-30,-25,-20,+20,+25,+30,+35]` mm and y
    offsets `[-25,-20,-15,+15,+20,+25]` mm;
  - line 499 reports realization for far sponge, y -25mm, and y +15mm.
- B200 targeted y-check log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_ycheck_b200.out`:
  - line 43 retests y offsets `[-25,-20,+15,+20]` with start sponge xyz/top/dx/dy
    metrics;
  - line 136 contradicts the earlier y -25mm pass: y -25mm fails with
    `target_realized=NO`, `final_target_tcp_error_m=0.014990`,
    `final_shoulder_error_deg=3.120213`, `start_target_xy_inside_sponge_aabb=NO`,
    and `start_sponge_top_z_m=0.047000`;
  - line 186 repeats y +15mm realization but reveals the confound:
    `start_sponge_xyz=([+0.265269, -0.031637, +0.011000])`,
    `start_sponge_top_z_m=0.034500`, and
    `start_target_tcp_minus_sponge_top_m=-0.010271`, versus nominal top
    `0.047000` on lines 86/98;
  - line 211 shows y +20mm fails with the usual top height
    `start_sponge_top_z_m=0.047000` and
    `start_target_tcp_minus_sponge_top_m=-0.022771`;
  - lines 249-250 aggregate only far sponge and y +15mm as realized.
- All follow-up stderr files contain only the known cpufreq/NVML/Fabric warnings
  on lines 1-4 and no Python traceback.

Implication:

- The z boundary is tighter than D039 stated: under the current reduction-based
  diagnostic gate and nominal settled sponge posture, the transition is between
  +12.75mm and +13.0mm.
- Do not treat +13mm as a robust solved grasp target. It is a marginal
  command-realization improvement with centimeter-scale final TCP error.
- Do not treat horizontal offset as an independently validated fix. The only
  reproducible y-offset pass is confounded by lower settled sponge top/posture,
  and the earlier y -25mm pass did not reproduce.
- The strongest current hypothesis remains local contact/clearance/posture
  interaction around the nominal grasp geometry. Before any chain integration,
  a valid next diagnostic should explicitly control or log sponge pose/orientation
  while testing z/clearance. This is still not P7 success, attach physics,
  transport/release validation, SurfaceGripper validation, or constraint
  integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
- `claudedocs/session_20260518_p7_branch_b_grasp_pose_deadzone.md`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zfine_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zmicro_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_xy_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_ycheck_b200.{out,err}`

## D041 — Pose/top control preserves the +12.8mm boundary, but the pass remains a thin reduction-gate artifact

Evidence:

- Extended `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py` to
  log sponge root quaternion, `up_z`, tilt, upright top, oriented top, and to
  optionally reassert sponge pose before delivery. Latest md5:
  `bee46b8203e9dfdd5d86b69301551af0`. This remains pre-integration only: no
  constraints, no SurfaceGripper, no transport target, no release, no P7
  training/tuning, and no env/train/chain default edits.
- B200 uncontrolled pose/top log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_log_zboundary_b200.out`:
  - line 43 tests +12.75/+12.875/+13.0mm;
  - lines 86/98 show nominal sponge is effectively upright with
    `start_sponge_oriented_top_z_m=0.047000`;
  - lines 136/148 show +12.75mm env/direct fail with
    `start_target_tcp_minus_sponge_oriented_top_m=-0.010666`;
  - lines 173 and 186/198 show +12.875mm and +13.0mm env/direct pass the
    reduction gate;
  - lines 224-225 aggregate +12.875/+13.0mm realized and +12.75mm failed.
- B200 controlled pose/top log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zboundary_b200.out`:
  - line 42 confirms `reassert_sponge_before_delivery=YES` and
    `reassert_sponge_z_m=0.0235`;
  - reassert lines 76, 130, 157, and 184 show the requested/actual sponge pose is
    `(+0.250000,-0.040000,+0.023500)` with identity quaternion and
    upright/oriented top `0.047000m` before delivery;
  - lines 141/154 show +12.75mm env/direct still fail;
  - lines 168/181 show +12.875mm env/direct pass;
  - lines 195/208 show +13.0mm env/direct pass;
  - lines 236-237 aggregate the same split and no direct-rescue condition.
- B200 controlled micro2 log
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zmicro2_b200.out`:
  - line 42 again confirms controlled pose reassert at `z=0.0235`;
  - lines 130/157/184 show identity/upright top `0.047000m` before each tested
    z condition;
  - lines 141/154 show +12.8125mm env/direct fail:
    env `final_target_tcp_error_m=0.011191`,
    `final_shoulder_error_deg=2.266477`, `target_realized=NO`; direct
    `final_target_tcp_error_m=0.011147`,
    `final_shoulder_error_deg=2.290611`, `target_realized=NO`;
  - lines 168/181 show +12.84375mm env/direct pass:
    env `final_target_tcp_error_m=0.011184`,
    `final_shoulder_error_deg=2.244741`, `target_realized=YES`; direct
    `final_target_tcp_error_m=0.011148`,
    `final_shoulder_error_deg=2.262267`, `target_realized=YES`;
  - lines 195/208 show +12.875mm env/direct pass;
  - lines 236-237 aggregate +12.84375/+12.875mm realized and +12.8125mm failed.
- All three stderr files contain only the known cpufreq/NVML/Fabric warnings on
  lines 1-4 and no Python traceback.

Implication:

- The +12.75 to +13.0mm split is not explained by sponge tilt or top-height
  drift. It survives identity-quaternion/top-controlled reassertion.
- Under the current diagnostic gate, the controlled transition is between
  +12.8125mm and +12.84375mm.
- This does not mean the grasp pose is solved. Passing cases still retain about
  11mm final TCP error, and the pass/fail flip is a thin reduction-gate boundary
  driven by shoulder-error reduction, not exact target convergence.
- Do not tune the diagnostic gate or P7 thresholds to make this look like a
  success. The actionable lesson is that the nominal contact/clearance posture is
  marginal and needs a mechanically valid grasp/clearance strategy before any
  RoArm chain constraint integration.
- Still not P7 success, attach physics, transport/release validation,
  SurfaceGripper validation, or constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
- `claudedocs/session_20260518_p7_branch_b_grasp_pose_deadzone.md`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_log_zboundary_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zboundary_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zmicro2_b200.{out,err}`

## D042 — The grasp-pose dead zone is a sponge-top contact clamp; reduction-gate passes are not useful below-top command realization

Evidence:

- Extended `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
  with diagnostic-only per-step tracing, per-joint error/velocity fields,
  Articulation `joint_pos_target` snapshots, final TCP-vs-oriented-top metrics,
  and optional shoulder-nudge magnitude sweep. Latest md5:
  `e0e84e481c3be8be7777a85ef2465c57`. This remains pre-integration only: no
  constraints, no SurfaceGripper, no transport target, no release, no P7
  training/tuning, and no env/train/chain default edits.
- B200 stall trace
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_stall_trace_zmicro_b200.out`:
  - line 42 confirms the run used the unchanged `target_error_gate_m=0.003000`,
    `joint_nudge_degs=[5.0]`, `trace_every_step=YES`, and sponge pose reassert at
    `z=0.0235`;
  - line 421 shows +12.8125mm fails, but not from target-buffer loss:
    `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000008`, final TCP
    is clamped at the top (`final_tcp=...+0.047017`,
    `final_tcp_minus_sponge_oriented_top_m=-0.000043`) while the target remains
    below top (`final_target_tcp_minus_sponge_oriented_top_m=-0.010667`);
  - line 552 shows +12.84375mm passes only the reduction gate with essentially the
    same clamp: `final_target_tcp_error_m=0.011184`,
    `final_shoulder_error_deg=2.244741`,
    `final_tcp_minus_sponge_oriented_top_m=-0.000036`, and target still below top
    by `-0.010620m`;
  - line 683 shows +13.0mm also passes with the same pattern:
    `final_target_tcp_error_m=0.011042`,
    `final_tcp_minus_sponge_oriented_top_m=-0.000018`, target below top by
    `-0.010475m`;
  - lines 860-862 aggregate the unchanged split: +12.8125mm fails while
    +12.84375mm and +13.0mm pass the current gate; no attach/release physics is
    claimed.
- B200 nudge-direction run
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_nudge_direction_b200.out`:
  - lines 42-43 confirm a controlled pose-reassert run with no z-offset variants
    and shoulder nudges `[-5.0, 2.5, 5.0]`;
  - line 87 shows a nominal-sponge `-5deg` shoulder nudge realizes because the
    target is above the sponge top (`start_target_tcp_minus_sponge_oriented_top_m=0.023941`,
    `final_target_tcp_error_m=0.000986`);
  - line 114 shows a smaller `+2.5deg` downward nudge still fails when its target is
    below top (`start_target_tcp_minus_sponge_oriented_top_m=-0.011317`) and final
    TCP remains near top (`final_tcp_minus_sponge_oriented_top_m=-0.000135`);
  - line 141 repeats the `+5deg` downward failure with target below top by
    `-0.022771m`;
  - lines 195 and 222 show the same +2.5/+5deg targets realize with the sponge far,
    preserving the contact/proximity attribution.
- Both new stderr files contain only the known cpufreq/NVML/Fabric warnings on
  lines 1-4 and no Python traceback.

Implication:

- D041 is refined: the +12.84375/+13.0mm pass is not useful command realization.
  It is a shoulder-error reduction gate crossing while the TCP remains clamped at
  the sponge top and the target TCP is still about 10-11mm below the oriented top.
- The immediate blocker should be treated as local sponge-top contact equilibrium
  around the nominal pre-close posture. Upward/above-top commands realize; below-top
  downward commands stall near the top even when target buffers are correct.
- Do not tune the diagnostic gate or describe these passes as a grasp solution.
  The next valid work remains diagnostic-only: design/test a mechanically valid
  pre-close clearance/grasp posture before any RoArm chain constraint integration.
- Still not P7 success, attach physics, transport/release validation,
  SurfaceGripper validation, or constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
- `claudedocs/session_20260518_p7_branch_b_grasp_pose_deadzone.md`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_stall_trace_zmicro_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_grasp_pose_nudge_direction_b200.{out,err}`

## D043 — A valid pre-close strategy must avoid below-top targets inside the nominal sponge footprint

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
  md5 `5be8cfb8c1a58f6de43f431db0befff4`. This is diagnostic-only and
  pre-integration: no constraint prim insertion, no fixed/dynamic integration,
  no SurfaceGripper, no attached transport, no transport target, no release or
  scripted release variant, no P7 training/tuning, no diagnostic gate tuning,
  and no env/train/chain default edits.
- B200
  `/tmp/p7_branch_b_roarm_chain_preclose_clearance_strategy_b200.out`:
  - line 41 confirms the strict scope and no attach/release physics claim;
  - line 42 records `target_error_gate_m=0.003000` and
    `reduction_gate_reference_only=YES`;
  - line 43 confirms no MOVE commands were executed;
  - line 172 shows the nominal below-top baseline fails exact convergence and
    top-clamps: target is inside the sponge AABB, target is `-0.022821m` below
    oriented top, final TCP is near the top
    (`final_tcp_minus_sponge_oriented_top_m=0.000099`),
    `final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
    `top_clamped=YES`, and clean realization `NO`;
  - line 272 shows the same below-top q target realizes when the sponge is far:
    `final_target_tcp_error_m=0.000854`, target outside AABB,
    `exact_converged=YES`, and `top_clamped=NO`;
  - lines 372/466 show upward-first then above-top exact-converges;
  - line 660 shows upward-first then top-tangent exact-converges with
    `final_target_tcp_error_m=0.000495` and `top_clamped=NO`;
  - line 854 shows upward-first followed by a final below-top target still
    reclamps: `final_target_tcp_error_m=0.011778`,
    `reduction_gate_would_pass=YES`, `exact_converged=NO`,
    `top_clamped=YES`, and clean realization `NO`;
  - line 1048 shows side-edge tangent outside the AABB exact-converges with
    `final_target_tcp_error_m=0.000910`;
  - lines 1050-1052 aggregate the clean candidates, clamped segments,
    `below_top_nominal_invalid=YES`, `attach_calls=0`, no NaN/done, and no
    attach/release physics claim.
- stderr lines 1-4 contain only the known cpufreq/NVML/Fabric warnings and no
  Python traceback.

Implication:

- D042 is not just a measurement artifact; it is now a pre-close strategy rule.
  Do not command a final pre-close TCP target below the sponge top inside/near
  the nominal footprint.
- Upward-first clearance is useful only when the final commanded target remains
  above or tangent to the top. Upward-first followed by a below-top target still
  reclamps and can misleadingly pass reduction-style metrics.
- The next valid pre-integration candidates are above/top-tangent pre-close and
  side/edge tangent approach. They are diagnostic candidates only, not attach
  physics, not transport/release, not SurfaceGripper, and not constraint
  integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`
- B200 `/tmp/p7_branch_b_roarm_chain_preclose_clearance_strategy_b200.{out,err}`

## D044 — For pre-close, final above/tangent geometry dominates tested clearance height; below-top inside-footprint remains banned

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
  md5 `95b4a8a317a9fb176c7ed258229925e5`. This is diagnostic-only and
  pre-integration: no constraint prim insertion, no fixed/dynamic integration,
  no SurfaceGripper, no attached transport, no transport target, no release or
  scripted release variant, no P7 training/tuning, no diagnostic gate tuning,
  and no env/train/chain default edits.
- B200 v2
  `/tmp/p7_branch_b_roarm_chain_preclose_geometry_sweep_v2_b200.out`:
  - line 41 confirms strict pre-integration scope and no attach/release physics
    claim;
  - line 42 records the unchanged exact gate (`target_error_gate_m=0.003000`),
    reference-only reduction gate, final top margins
    `[0.000200, 0.000500, 0.001000, 0.002000]`, clearance heights
    `[0.012000, 0.024000, 0.036000]`, and side margins
    `[0.002000, 0.006000, 0.012000, 0.018000]`;
  - line 43 confirms no MOVE commands were executed;
  - line 44 shows all IK targets converged before simulation;
  - line 172 preserves the nominal below-top inside-footprint baseline failure:
    `final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
    `top_clamped=YES`, and clean realization `NO`;
  - line 272 preserves the far-sponge no-contact control:
    `final_target_tcp_error_m=0.000854`, target outside AABB, exact convergence
    `YES`, and top clamp `NO`;
  - lines 466/660/854/1048 show final top margins +0.2/+0.5/+1.0/+2.0mm all
    exact-converge cleanly with no top clamp;
  - lines 1242/1436/1630 show upward clearance heights 12/24/36mm all
    exact-converge cleanly when the final target remains at +0.5mm top margin;
  - lines 1824/2018/2212/2406 show side-edge final targets at +2/+6/+12/+18mm
    outside the sponge AABB all exact-converge cleanly;
  - lines 2408-2409 separate the far-sponge control from contact candidates:
    `below_inside_segments_clean=[]`,
    `below_top_inside_targets_realize_cleanly=NO`,
    `contact_candidate_strategies=[...]`, `far_control_is_no_contact_control=YES`,
    `attach_calls=0`, no NaN/done, and no attach/release physics claim.
- stderr lines 1-4 contain only the known cpufreq/NVML/Fabric warnings and no
  Python traceback.

Implication:

- D043 is refined, not relaxed: final pre-close targets inside/near the nominal
  footprint must stay above/tangent to the oriented sponge top. Below-top
  inside-footprint targets remain mechanically invalid even if a reduction-style
  metric would look improved.
- Within the tested range, upward clearance height is secondary once the final
  target is above/tangent: 12/24/36mm clearance heights all realized cleanly.
- Side-edge tangent remains a diagnostic candidate even at a small +2mm outside
  width margin, but this is still not object attachment, transport, release,
  SurfaceGripper validation, or constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`
- B200 `/tmp/p7_branch_b_roarm_chain_preclose_geometry_sweep_v2_b200.{out,err}`

## D045 — Exact 3mm pre-close convergence is not enough; mechanical validity must gate below-top inside-footprint candidates

Evidence:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_selector_guard_probe.py`
  md5 `e50f7dfcb5651507b0c200af1299f171`. This is a diagnostic-only wrapper
  around the unchanged selector md5 `aa24ef00acbb9d8cd0aeee061b08f85f`; it does
  not train, tune gates, edit env/train/chain defaults, integrate constraints,
  attach SurfaceGripper, execute transport, or execute release.
- B200
  `/tmp/p7_branch_b_roarm_chain_preclose_selector_guard_b200.out`:
  - lines 2-4 confirm the guard wrapper scope and the adversarial case:
    `invalid_top_margin_m=-0.001500`, expected selector behavior `REJECT`, and
    interpretation `below_top_inside_invalid_even_if_exact_gate_passes`;
  - lines 43-45 confirm strict pre-integration scope, unchanged exact gate
    `target_error_gate_m=0.003000`, reduction gate reference-only, and
    `top_margin_m=-0.001500`;
  - line 46 confirms no MOVE commands were executed;
  - line 52 rejects `candidate_top_tangent_margin_neg1p5mm` before
    interpretation with reason `below_top_inside_footprint_invalid`;
  - line 476 shows the trap: the final target is below top and inside the AABB,
    yet `final_target_tcp_error_m=0.001268` and `exact_converged=YES`; it remains
    `top_clamped=YES`, `mechanically_valid_target=NO`, and
    `clean_realized_without_reduction_artifact=NO`;
  - line 477 keeps the strategy rejected;
  - lines 1060-1062 keep accepted clean candidates limited to above-top and
    side-edge, with `below_inside_segments_clean=[]`, `attach_calls=0`, no
    NaN/done, and no attach/release physics claim.
- stderr lines 1-4 contain only the known cpufreq/NVML/Fabric warnings and no
  Python traceback.

Implication:

- D043/D044 are tightened: exact target convergence under the unchanged 3mm gate
  is not sufficient for pre-close interpretation. A near-top below/inside target
  can satisfy the exact numeric gate while the TCP is effectively clamped at the
  sponge top.
- Candidate selection must apply mechanical validity first: below-top targets
  inside/near the nominal sponge footprint remain invalid, even if exact
  convergence and reduction-style metrics pass.
- Continue to use the selector only as a diagnostic gate. This is still not P7
  success, attach physics, transport/release validation, SurfaceGripper
  validation, or constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_preclose_selector_guard_probe.py`
- `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`
- B200 `/tmp/p7_branch_b_roarm_chain_preclose_selector_guard_b200.{out,err}`

## D046 — The below-top ban is footprint-specific; shallow below-top side-edge can be clean only when outside AABB

Evidence:

- Reused the unchanged selector
  `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  `--side_top_margin_m -0.0015`. No code, gate, env/train/chain default,
  constraint, SurfaceGripper, transport, release, or training change was made.
- B200
  `/tmp/p7_branch_b_roarm_chain_preclose_selector_side_below_guard_b200.out`:
  - line 41 confirms strict pre-integration scope and no attach/release physics
    claim;
  - line 42 confirms unchanged exact gate `target_error_gate_m=0.003000`,
    reduction gate reference-only, and `side_top_margin_m=-0.001500`;
  - line 43 confirms no MOVE commands were executed;
  - line 44 shows all IK targets converged;
  - line 52 accepts `candidate_side_edge_margin_2mm_top_margin_neg1p5mm` with
    reason `side_edge_target_outside_aabb`, while the final target is below top
    by `-0.001162m` and outside AABB;
  - line 1055 shows that side-edge below-top final segment exact-converged cleanly:
    `final_target_tcp_error_m=0.001074`, `exact_converged=YES`,
    `top_clamped=NO`, `mechanically_valid_target=YES`, and
    `clean_realized_without_reduction_artifact=YES`;
  - lines 1057-1059 still keep the nominal below-top inside-footprint baseline
    clamped, list the clean below segment only under the outside-AABB side-edge
    case, keep `below_inside_segments_clean=[]`, and report `attach_calls=0`, no
    NaN/done, and no attach/release physics claim.
- stderr lines 1-4 contain only the known cpufreq/NVML/Fabric warnings and no
  Python traceback.

Implication:

- D045 is not weakened: below-top targets inside/near the nominal footprint
  remain invalid even if exact convergence passes.
- The current selector's outside-AABB exception has diagnostic support for a
  shallow below-top side-edge target. The key distinction is footprint class:
  below-top inside-footprint can exact-converge by top clamp; below-top
  outside-AABB side-edge did not top-clamp in this run.
- This remains a pre-close diagnostic candidate only. It is not a chain-ready
  strategy, not attach physics, not transport/release validation, not
  SurfaceGripper validation, and not constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`
- B200 `/tmp/p7_branch_b_roarm_chain_preclose_selector_side_below_guard_b200.{out,err}`

## D047 — Side-edge below-top needs positive outside-AABB margin; the AABB boundary is not robust

Evidence:

- Reused the unchanged selector
  `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  `--side_top_margin_m -0.0015` and side margins
  `0.0/0.5/1.0/2.0/4.0/6.0mm`. No code, gate, env/train/chain default,
  constraint, SurfaceGripper, transport, release, or training change was made.
- B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_{0p0,0p5,1p0,2p0,4p0,6p0}_b200.{out,err}`.
- All runs kept the unchanged exact gate and scope on stdout line 42:
  `target_error_gate_m=0.003000`, reduction gate reference-only,
  `side_top_margin_m=-0.001500`.
- All runs kept the nominal below-top inside-footprint clamp baseline on line 179
  (`final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
  `top_clamped=YES`, `mechanically_valid_target=NO`) and the far-sponge
  no-contact control on line 279 (`0.000854`, exact-converged but
  `mechanically_valid_target=NO`).
- Top/tangent and above controls remained clean in every run: line 473
  (`0.000920`, exact-converged, no top clamp) and line 667 (`0.000921`, same).
- The zero-margin side-edge case is the boundary trap:
  - line 52 accepted `candidate_side_edge_margin_0mm_top_margin_neg1p5mm`
    because the planned final target was just outside the nominal AABB
    (`target_dy_sponge_m=0.011033`, `final_target_xy_inside_sponge_aabb=NO`);
  - line 1055 exact-converged numerically (`0.001251`) with no top clamp, but
    the final target was inside the realized AABB
    (`final_target_xy_inside_sponge_aabb=YES`), so
    `mechanically_valid_target=NO` and clean realization `NO`;
  - lines 1058-1059 therefore report accepted contact candidates clean `NO` and
    diagnostic success `NO`.
- Positive margins were clean:
  - 0.5mm line 1055: `0.001203`, exact `YES`, no top clamp, mechanically valid
    `YES`, clean `YES`;
  - 1.0mm line 1055: `0.001156`, exact `YES`, no top clamp, mechanically valid
    `YES`, clean `YES`;
  - 2.0mm line 1055: `0.001074`, exact `YES`, no top clamp, mechanically valid
    `YES`, clean `YES`;
  - 4.0mm line 1055: `0.000899`, exact `YES`, no top clamp, mechanically valid
    `YES`, clean `YES`;
  - 6.0mm line 1055: `0.000529`, exact `YES`, no top clamp, mechanically valid
    `YES`, clean `YES`.
- Lines 1057-1058 in the positive-margin runs keep
  `below_inside_segments_clean=[]`, `attach_calls=0`, no NaN/done, and no
  attach/release physics claim. stderr lines 1-4 in all six runs are only the
  known cpufreq/NVML/Fabric warnings.

Implication:

- D046 is refined: shallow below-top side-edge is a diagnostic candidate only
  with positive outside-AABB clearance. A nominal zero-margin boundary should be
  treated as invalid/not robust because tiny realized sponge pose differences can
  move the target back inside the footprint.
- Candidate interpretation must check realized/final AABB class, not just the
  planned selector class. Below-top inside-footprint remains banned even when the
  exact 3mm gate passes and top clamp is not triggered.
- This is still pre-integration diagnostic evidence only: not P7 success, not
  chain-ready, not attach physics, not transport/release validation, not
  SurfaceGripper validation, and not constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`
- B200 `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_{0p0,0p5,1p0,2p0,4p0,6p0}_b200.{out,err}`

## D048 — The tested side-edge boundary is between 0.0mm and 0.1mm, but 0.1mm is not a deployment margin

Evidence:

- Reused the unchanged selector
  `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  `--side_top_margin_m -0.0015` and fine side margins
  `0.1/0.2/0.3/0.4/0.5mm`. No code, gate, env/train/chain default,
  constraint, SurfaceGripper, transport, release, or training change was made.
- B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_{0p1,0p2,0p3,0p4,0p5}_b200.{out,err}`.
- All five runs kept the unchanged exact gate and scope on stdout line 42 and
  the unchanged selector rule on line 46.
- All five runs preserved controls:
  - line 179 kept the nominal below-top inside-footprint baseline clamped
    (`0.023923m`, `exact_converged=NO`, `top_clamped=YES`,
    `mechanically_valid_target=NO`);
  - line 279 kept the far-sponge below-top no-contact control exact-converged
    (`0.000854m`) but mechanically invalid as a contact candidate;
  - lines 473 and 667 kept the top-tangent and above-top controls clean
    (`0.000920/0.000921m`, exact `YES`, no top clamp).
- Fine side margins all stayed outside AABB in the realized/final segment and
  cleanly exact-converged on line 1055:
  - 0.1mm: planned `target_dy_sponge_m=0.011133` on line 52; line 1055
    `final_target_tcp_error_m=0.001241`, exact `YES`, no top clamp,
    mechanically valid `YES`, clean `YES`, final target outside AABB;
  - 0.2mm: `0.011232`; line 1055 `0.001232`, mechanically valid/clean `YES`;
  - 0.3mm: `0.011332`; line 1055 `0.001222`, mechanically valid/clean `YES`;
  - 0.4mm: `0.011432`; line 1055 `0.001213`, mechanically valid/clean `YES`;
  - 0.5mm: `0.011532`; line 1055 `0.001203`, mechanically valid/clean `YES`.
- Lines 1057-1059 in all five fine runs keep
  `below_inside_segments_clean=[]`, `attach_calls=0`, accepted candidates clean
  `YES`, no NaN/done, and diagnostic success `YES`. stderr lines 1-4 in all
  five runs are only the known cpufreq/NVML/Fabric warnings.

Implication:

- D047 is refined: in this deterministic B200 diagnostic, the observed
  realized/final AABB boundary lies between 0.0mm and 0.1mm side margin for
  `side_top_margin_m=-0.0015`.
- Do not treat 0.1mm as a robust deployment or chain margin. It is only the
  minimum tested positive pass in this local diagnostic. Future diagnostic-local
  selection should still prefer a nonzero safety margin and must check
  realized/final AABB class before interpreting below-top side-edge candidates.
- This remains pre-integration diagnostic evidence only: not P7 success, not
  chain-ready, not attach physics, not transport/release validation, not
  SurfaceGripper validation, and not constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`
- B200 `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_{0p1,0p2,0p3,0p4,0p5}_b200.{out,err}`

## D049 — At 2mm outside-AABB, side-edge below-top depth is exact-clean through about -3mm and loses exact convergence by -4mm

Evidence:

- Reused the unchanged selector
  `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with fixed
  `--side_margin_m 0.0020` and side top margins
  `-0.5/-1.0/-1.5/-2.0/-3.0/-4.0/-6.0mm`. No code, gate, env/train/chain
  default, constraint, SurfaceGripper, transport, release, or training change
  was made.
- B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_{neg0p5,neg1p0,neg1p5,neg2p0,neg3p0,neg4p0,neg6p0}_b200.{out,err}`.
- All seven runs kept the unchanged exact gate/scope and selector rule. The
  side candidate stayed outside AABB in all final segments, and
  `below_inside_segments_clean=[]`, `attach_calls=0`, no NaN/done, and no
  attach/release physics claim were preserved.
- The side-edge final segment on line 1055 remained clean through -3mm:
  - -0.5mm: final error `0.000548`, exact `YES`, no top clamp, mechanically
    valid `YES`, clean `YES`, final target class `tangent`;
  - -1.0mm: `0.000702`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target class `tangent`;
  - -1.5mm: `0.001074`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target class `below`;
  - -2.0mm: `0.001504`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`;
  - -3.0mm: `0.002409`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`.
- The deeper side-edge targets were still outside AABB and not top-clamped, but
  they lost the unchanged 3mm exact gate:
  - -4.0mm line 1055: final error `0.003346`, exact `NO`, top clamp `NO`,
    mechanically valid `YES`, clean `NO`; lines 1058-1059 report accepted
    contact candidates clean `NO` and diagnostic success `NO`;
  - -6.0mm line 1055: final error `0.005177`, exact `NO`, top clamp `NO`,
    mechanically valid `YES`, clean `NO`; lines 1058-1059 likewise report
    diagnostic success `NO`.
- stderr lines 1-4 in all seven runs are only the known cpufreq/NVML/Fabric
  warnings; no Python traceback was found.

Implication:

- The side-edge outside-AABB exception has a depth limit under the unchanged
  3mm exact gate. At 2mm outside-AABB, this deterministic B200 diagnostic is
  clean through about -3mm below top and loses exact convergence by -4mm.
- The -4/-6mm failures should not be interpreted as inside-footprint clamp
  failures: they are outside AABB and not top-clamped, but residual target error
  exceeds the exact gate. Mechanical validity alone is not enough; exact
  convergence still gates diagnostic cleanliness.
- This remains pre-integration diagnostic evidence only: not P7 success, not
  chain-ready, not attach physics, not transport/release validation, not
  SurfaceGripper validation, and not constraint integration.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`
- B200 `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_{neg0p5,neg1p0,neg1p5,neg2p0,neg3p0,neg4p0,neg6p0}_b200.{out,err}`

## D050 — Real RoArm can produce a CLOSE-near top-tangent 4mm local signal, but only as virtual-carrier evidence

Evidence:

- Added and B200-ran
  `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
  md5 `2b63df20972ad1e923f24e05c2810957` after static review fixes. The fixes
  were limited to this new diagnostic: `--reassert_sponge_z_m` now sets the
  actual sponge root z, and `post_close_marker` now gates success before local
  signal execution.
- The probe is virtual-carrier/signal-only. B200
  `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_default_b200.out`:
  - line 41 confirms no constraint prim insertion, no fixed/dynamic constraint
    integration, no SurfaceGripper, no attached transport, no transport target,
    no release marker, no scripted release variant, no P7 training/tuning, no
    diagnostic gate tuning, no env/train/chain default edits, and no attach
    success claim;
  - line 42 confirms the default top-tangent geometry, `signal_stage=just_before_close`,
    `micro_delta_m=0.004000`, unchanged `0.003000m` target gate, and `0.010000m`
    max TCP step gate;
  - line 43 confirms `move_cmds_executed=0`, raw planner gap still
    `0.211271`, and `raw_gap_ok=NO`;
  - lines 44-46 show IK convergence for the clearance, final top-tangent signal
    pose, `micro_plus_x`, and `micro_return_x` targets.
- B200 local execution:
  - line 279 reached safe clearance with final target error `0.002505m`;
  - line 285 reached the top-tangent signal pose with final target error
    `0.002050m`;
  - line 291 passed a 5-step stationary hold with final target error
    `0.000922m`;
  - line 297 reached the 4mm `micro_plus_x` target in 5 steps with final target
    error `0.002267m`;
  - line 300 reached `micro_return_x` in 2 steps with final target error
    `0.001351m`.
- Line 301 reports `prep_events_done=38/38`, `max_final_target_error_m=0.002505`,
  `max_tcp_step_m=0.003353`, `max_tcp_anchor_offset_error_m=0.00000000`,
  `max_sponge_drift_m=0.000000`, `max_sponge_speed_mps=0.000540`,
  `max_quat_angle_deg=0.000`, `min_upright_z=1.000000`, `attach_calls=0`,
  `posewrite_calls=0`, `virtual_carrier_only=YES`, `transport_target=NO`, and
  `release_marker=NO`.
- Lines 302-303 report all intended gates YES and
  `ROARM_CLOSE_NEAR_LOCAL_SIGNAL_SUCCESS=YES`.
- B200 stderr lines 1-4 contain the known cpufreq/NVML/Fabric messages and the
  stdout/stderr scan found no Python traceback or exception. A post-run process
  check found no matching P7/Isaac/training process.

Implication:

- D038/D043-D049 are refined: the real RoArm can produce a small CLOSE-near local
  TCP signal when the final geometry is admissible top-tangent, so the current
  blocker is narrower than "no local signal near CLOSE".
- This does not validate dynamic-anchor constraint integration. The carrier is
  virtual; no USD joint/constraint was inserted, no object was attached by
  physics, no attached transport target was visited, and no release occurred.
- Do not convert this into a P7 success claim, attach claim, transport claim,
  release claim, SurfaceGripper claim, or chain-ready constraint claim.
- Any follow-up must remain signal-only unless explicitly approved. Reasonable
  follow-ups are limited checks such as the same script's `post_close_marker`
  mode or conservative side-edge geometry; they are not a license to go to
  transport/release or integrate constraints.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
- `claudedocs/session_20260519_p7_branch_b_close_near_local_signal.md`
- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_default_b200.{out,err}`

## D051 — Top-tangent CLOSE-near local signal also survives a close-marker-only step, but still is not attach evidence

Evidence:

- Reused
  `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
  md5 `2b63df20972ad1e923f24e05c2810957` with
  `--signal_stage post_close_marker`. This was an approved follow-up in the same
  virtual-carrier/signal-only envelope: no constraint prim insertion, no
  fixed/dynamic constraint integration, no SurfaceGripper, no attached transport,
  no transport target, no release marker, no P7 training/tuning, no diagnostic
  gate tuning, and no env/train/chain default edits.
- B200
  `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_post_close_marker_b200.out`:
  - line 41 confirms the strict no-overclaim scope and
    `claim_attach_success=NO`;
  - line 42 confirms `geometry=top_tangent`,
    `signal_stage=post_close_marker`, `micro_delta_m=0.004000`, unchanged
    `0.003000m` target gate, and `0.010000m` max TCP step gate;
  - line 43 confirms `move_cmds_executed=0`, raw planner gap still
    `0.211271`, and `raw_gap_ok=NO`;
  - lines 274-276 show the close-marker-only/no-posewrite step reached with
    final target error `0.001131`, `attach_calls=0`, `posewrite_calls=0`, and
    `claim_attach_success=NO`;
  - lines 282, 288, 294, 299, and 302 show safe clearance, top-tangent signal
    pose, stationary hold, `micro_plus_x`, and `micro_return_x` all reached;
  - line 303 reports `prep_events_done=38/38`,
    `max_final_target_error_m=0.002576`, `max_tcp_step_m=0.003432`,
    `max_tcp_anchor_offset_error_m=0.00000000`,
    `max_sponge_drift_m=0.000000`, `max_sponge_speed_mps=0.000341`,
    `max_quat_angle_deg=0.000`, `min_upright_z=1.000000`,
    `attach_calls=0`, `posewrite_calls=0`, `virtual_carrier_only=YES`,
    `transport_target=NO`, and `release_marker=NO`;
  - lines 304-305 report all intended gates YES and
    `ROARM_CLOSE_NEAR_LOCAL_SIGNAL_SUCCESS=YES`.
- B200 stderr lines 1-4 contain only the known cpufreq/NVML/Fabric messages
  seen in other Isaac diagnostics. A post-run process check found no matching
  P7/Isaac/training process.

Implication:

- D050 is strengthened but not broadened into attach/transport evidence: the
  top-tangent 4mm local TCP signal exists both just before CLOSE and after a
  close-marker-only/no-posewrite step.
- The result still does not validate `_grasped` attach physics. The close marker
  did not use env pose-write attach, did not insert a USD fixed/dynamic
  constraint, did not attach SurfaceGripper, did not visit a transport target,
  and did not execute release.
- Do not treat this as P7 success, dynamic-anchor chain integration, object
  attachment, SurfaceGripper validation, attached transport, transport target, or
  release validation.
- The remaining unvalidated boundary before any constraint integration is still
  the actual attach/constraint handoff surface under a new explicit approval and
  falsifiable gate. This D051 result only says the real RoArm can supply the
  local top-tangent signal that such a future handoff design would need.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
- `claudedocs/session_20260519_p7_branch_b_close_near_local_signal.md`
- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_post_close_marker_b200.{out,err}`

## D052 — Conservative side-edge can hold the CLOSE-near pose, but does not realize the 4mm local signal

Evidence:

- Reused
  `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
  md5 `2b63df20972ad1e923f24e05c2810957` with only `--geometry side_edge`.
  This was an approved follow-up in the same virtual-carrier/signal-only
  envelope: no constraint prim insertion, no fixed/dynamic constraint
  integration, no SurfaceGripper, no attached transport, no transport target, no
  release marker, no P7 training/tuning, no diagnostic gate tuning, and no
  env/train/chain default edits.
- The source guard keeps this side-edge diagnostic conservative:
  `side_margin_m >= 0.0020` and `side_top_margin_m >= -0.0030`.
- B200
  `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_side_edge_b200.out`:
  - lines 41-43 confirm strict no-overclaim scope, `geometry=side_edge`,
    `signal_stage=just_before_close`, `micro_delta_m=0.004000`,
    `move_cmds_executed=0`, raw planner gap `0.211271`, and `raw_gap_ok=NO`;
  - lines 44-46 show IK convergence for clearance, side-edge signal pose,
    `micro_plus_x`, and `micro_return_x`;
  - line 279 shows side-edge clearance reached with final target error
    `0.002567m`;
  - line 285 shows the conservative side-edge signal pose reached with final
    target error `0.002879m`, target below top by about `-0.003005m`, and
    outside the sponge AABB;
  - line 291 shows the 5-step stationary hold reached with final target error
    `0.002875m`;
  - lines 292-295 show the 4mm `micro_plus_x` target remained around
    `0.005342-0.005379m` error through 60 steps;
  - line 296 reports `micro_plus_x` `reached=NO`, steps `60`,
    `final_target_error_m=0.005342`, `set_target_seen=YES`, and
    `early_kill=YES`;
  - line 297 reports `prep_events_done=38/38`,
    `max_final_target_error_m=0.005342`, `max_tcp_step_m=0.003899`,
    `max_tcp_anchor_offset_error_m=0.00000000`,
    `max_sponge_drift_m=0.000040`, `max_sponge_speed_mps=0.013705`,
    `attach_calls=0`, `posewrite_calls=0`, `virtual_carrier_only=YES`,
    `transport_target=NO`, and `release_marker=NO`;
  - lines 298-299 report `micro_motion_realized_ok=NO`, `target_error_ok=NO`,
    and `ROARM_CLOSE_NEAR_LOCAL_SIGNAL_SUCCESS=NO`.
- B200 stderr lines 1-4 contain only the known cpufreq/NVML/Fabric messages.
  A stdout/stderr scan found no Python traceback or exception, and a post-run
  process check found no matching P7/Isaac/training process.

Implication:

- D050/D051 are geometry-specific positive evidence for top-tangent local
  signal. They must not be generalized to conservative side-edge 4mm local
  micro-motion.
- Conservative side-edge geometry remains admissible as pre-close geometry
  evidence through about -3mm depth, but that does not imply a realized 4mm local
  CLOSE-near signal.
- This result does not validate `_grasped` attach physics, constraint insertion,
  SurfaceGripper, attached transport, transport target, or release.
- Do not proceed to post-close-marker+side-edge, constraint integration,
  transport, SurfaceGripper, or release without a separate explicit approval and
  a new narrow falsifiable gate.
- No new pre-close matrix is justified by this result. It is a single-point
  side-edge signal failure inside the already-conservative diagnostic envelope.

Sources:

- `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
- `claudedocs/session_20260519_p7_branch_b_close_near_local_signal.md`
- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_side_edge_b200.{out,err}`

## D053 — Normalize the grasp problem around a 2cm cube object-frame primitive before scaling demonstrations

Evidence:

- The user corrected the current professor feedback on 2026-05-20: the important
  direction is to normalize the task around the physically cut
  `2cm x 2cm x 2cm` sponge cube, not to keep chasing the old long-sponge grasp
  geometry.
- The professor's suggested structure is object-frame and gripper-geometry aware:
  account for the effectively fixed jaw, do not drive the TCP blindly into the
  cube center, open before descent, descend to a lateral/contact-height offset,
  close, hold, and lift.
- The same normalized primitive should later vary `x/y` translation, `z` layer
  height, and yaw/rotation so B200 can generate large sim demonstration corpora
  for VLA imitation/co-training instead of relying on months of real-only data.
- Current local static evidence supports the need for a more principled
  object-frame model. v3's gripper-mounted counter was statically plausible but
  imbalanced at close_26 (`moving_y=0.004011m`, `counter_y=0.000261m`) and B200
  close/latch still failed. v4 only balances AABB overlap at close_26
  (`0.002011m / 0.002011m`) and loses moving contact at close_30; it is a
  diagnostic latch-stop26 candidate, not a solved grasp primitive.

Implication:

- Do not treat v2, v3, or v4 opposing-jaw geometry as solved grasp.
- Do not treat `_grasped_marker=YES` as success; require reached, stable hold,
  lift follow, low drift/speed/tilt, and `posewrite_calls=0`.
- Before B200 conversion or physics runs, perform a static object-frame geometry
  audit: fixed-jaw frame, moving-jaw frame, gripper opening/closing angle sweep,
  cube contact height, and z/yaw variation plan.
- If a physics test is later approved, keep it close/lift-only and diagnostic:
  no training, no SurfaceGripper, no constraints/default integration, no
  transport target, and no release.
- Only after a canonical cube grasp primitive passes falsifiable physics gates
  should the project scale to B200 procedural sim demonstration generation.

Sources:

- `claudedocs/session_20260520_p7_branch_b_normalized_cube_grasp_feedback.md`
- `sim_scripts/p7_branch_b_cube2cm_close_equilibrium_static_analysis.py`
- `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v4_urdf.py`
- Local logs:
  `/tmp/p7_branch_b_cube2cm_close_equilibrium_static_analysis_v3_v4_sweep_local.out`
  and `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_urdf_prep_local.out`

## D054 — Treat URDF-to-USD conversion, static contact, and runtime grasp physics as separate gates

Evidence:

- The earlier 2026-05-20 rolling docs were stale: `START_HERE.md`,
  `claudedocs/EXPERIMENT_LEDGER.md` row 73, and
  `claudedocs/session_20260520_p7_branch_b_normalized_cube_grasp_feedback.md`
  still described v4 as not converted / physics-unvalidated.
- Later B200 logs prove v4 conversion did run:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_convert_b200.out:82` reports
  `cube2cm_counter_jaw_v4_link` merged into `gripper_link`, and
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_usd/roarm_m3.usd`
  has md5 `4497024d25abab11de5c50e144124553`.
- v4 still failed physics/telemetry:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_close26_hold_lift_b200.out:390-391`
  reports `reached=NO` and `verdict=LATCH_FAIL`; runtime telemetry
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v4_b200.out:422-423`
  reports `moving_contact=YES`, `counter_contact=NO`, `one_sided_push=YES`,
  and `success_claim=NO`.
- v5 repeated the pattern: prep/conversion succeeded
  (`/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_urdf_prep_b200.out:25,28`;
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_convert_b200.out:82,84,86`),
  but runtime telemetry failed with one-sided push
  (`/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v5_b200.out:422-423`).
- v6 static prep and conversion succeeded
  (`/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_urdf_prep_b200.out:23-26`;
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_convert_b200.out:82`), but valid
  LD_PRELOAD runtime telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v6_ldpreload_b200.out:417-418`
  reports `target_error_m=0.024584`, `counter_contact=NO`,
  `one_sided_push=YES`, `reached=NO`, and `success_claim=NO`.
- v7 static/prep succeeded
  (`/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.out:23-28`),
  but conversion is blocked by B200 NVIDIA/NVML/GLX driver-library mismatch:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_b200.err:1-7,64-66,87-90`
  and
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_ldpreload_b200.err:1-7,64-66,87-90`.

Implication:

- Do not claim grasp physics success from URDF-to-USD conversion, USD md5s, or
  static AABB contact alone.
- Report each candidate in three separate states: static/prep, conversion/USD,
  and runtime physics. A candidate can pass one gate and fail the next.
- Current v4/v5/v6 evidence points to a simulation contact-proxy failure mode:
  the rigid cube is pushed by the moving jaw before counter contact closes.
  Correct framing is that the current Isaac rigid-cube/jaw collision/contact
  proxy is not reproducing real foam grasp; do not say the real robot cannot
  grasp the cube.
- The initial v7 conversion attempts were blocked by the B200 driver/library
  mismatch until the D024 conversion-only retry recovered USD export. v7 remains
  physics-unvalidated; stop before runtime telemetry unless separately approved.

Sources:

- `claudedocs/session_20260520_p7_branch_b_cube_contact_state_repair.md`
- `START_HERE.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- B200 logs under `/tmp/p7_branch_b_cube2cm_*_b200.{out,err}` cited above.

## D055 — Use the D024 B200 override path for v7 conversion-only recovery; conversion still is not physics validation

Evidence:

- B200 still has the known userspace mismatch: `libnvidia-ml.so.1` points to
  `libnvidia-ml.so.580.159.03`, while `libnvidia-ml.so.580.95.05` is also
  present; plain `nvidia-smi` reports driver/library mismatch.
- The first v7 conversion and the wrong-library LD_PRELOAD retry both crashed:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_b200.err:1-7,64-66,87-90`
  and
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_ldpreload_b200.err:1-7,64-66,87-90`.
- The D024 conversion-only retry used
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`. It exited 0.
- B200 `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.out:84-89`
  reports `cube2cm_fixed_counter_jaw_v7_link` merged into `link5`, `hand_tcp`
  merged into `link5`, and `base_link` merged into `world`.
- B200 `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.err:1-6`
  contains cpufreq and `NVML_ERROR_UNINITIALIZED` messages only; grep found no
  traceback, exception, fatal, segfault, or driver/library mismatch.
- v7 D024 USD md5s:
  - `roarm_m3.usd` `4497024d25abab11de5c50e144124553`
  - `config.yaml` `f2777880ff2c90182484d82b7f49e5a6`
  - `configuration/roarm_m3_base.usd` `d7aae34ddca6a4d4f1ce092bda28d1a2`
  - `configuration/roarm_m3_physics.usd` `75f7b1e6da1f5f14019a53f091ec2076`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`
- Post-run process check found no matching Isaac/conversion/training process.

Implication:

- v7 is now static/prep-valid and USD-converted under D024, but still
  physics-unvalidated.
- Do not rerun v7 conversion with the plain B200 environment or the wrong
  `580.159.03` preload path.
- Do not interpret v7 conversion as grasp success. Runtime jaw telemetry or
  hold-lift remains a separate gate requiring separate explicit approval.

Sources:

- `claudedocs/session_20260520_p7_branch_b_cube_contact_state_repair.md`
- B200 `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.{out,err}`
- B200 `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/`

## D056 — v7 link5 fixed-counter runtime telemetry still fails; do not escalate to hold-lift

Evidence:

- After user approval, `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  was patched to support v7 diagnostic telemetry. The patch uses the D024 v7 USD
  path, tracks the fixed counter jaw with `counter_parent=link5`, uses the actual
  runtime link5 transform, and logs strict contact separately from the 1mm slop
  contact used by the static v7 audit. The patched script md5 is
  `0b4d3f579d3bb56f994983a876198d65`; local and B200 `py_compile` passed, and
  remote md5 matched.
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:38`
  confirms strict scope: diagnostic-only, `variant=v7`, D024 USD path, no
  training, no constraint insertion, no SurfaceGripper, no transport/release, no
  gate tuning, close_26-only, and `claim_p7_success=NO`.
- B200 line 39 selected the 3cm cube and reported IK OK
  (`ik_err_mm=(0.477,0.316)`, `max_fk_error_m=0.000518`).
- B200 line 68 confirmed the intended authored geometry:
  `counter_parent=link5`, `design_moving_center_ref=([+0.000000,+0.014250,+0.002000])`,
  `design_counter_center_ref=([+0.000000,-0.019600,+0.002000])`, and
  `counter_contact_slop_m=0.001000`.
- B200 final close line 419 reports `target_error_m=0.023422`,
  `moving_contact=YES`, `counter_contact=NO`, `moving_slop_contact=YES`,
  `counter_slop_contact=NO`, `one_sided_push=YES`, and `reached=NO`.
- B200 aggregate line 420 reports `approach_ok=YES`, `descend_ok=YES`,
  `close_reached=NO`, `grasped_seen=NO`, `attach_calls=0`,
  `posewrite_calls=0`, `telemetry_only=YES`, and `success_claim=NO`.
- B200 stderr lines 1-4 contain cpufreq/NVML-uninitialized/Fabric messages; grep
  found no traceback, exception, fatal, segfault, or driver/library mismatch.
  Post-run process check was empty. Log md5s:
  stdout `3939f08ea684c34f76669293b96610ba`, stderr
  `a0cb0d2eb0dca684599e693fcd1e7af7`.

Implication:

- v7 is not a physics success. It is static/prep-valid and USD-converted, but
  close_26 runtime telemetry still fails.
- The v7 fixed-counter hypothesis did not resolve the current rigid-cube
  one-sided-push failure. Even the diagnostic 1mm slop contact did not reach the
  counter side at final close.
- Do not run hold-lift from this state; close-time contact/dynamics did not pass.
- Correct framing remains: the current Isaac rigid-cube/jaw collision/contact
  proxy is not reproducing real foam grasp. This does not mean the real robot
  cannot grasp the cube.
- Next technical work should be analytical/modeling: decide whether more rigid
  proxy probing is still informative or whether the project should explicitly
  model foam/contact compliance before any dataset generation.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `claudedocs/session_20260520_p7_branch_b_cube_contact_state_repair.md`
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.{out,err}`

## D057 — Stop spending primary effort on rigid offset variants; next cube branch must model compliance explicitly

Evidence:

- v7 static prep was already a tolerance/slop candidate, not a strict rigid
  two-sided pinch. B200
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.out:23-28` reports
  moving jaw strict contact YES, fixed counter strict contact NO, and fixed
  counter 1mm slop contact YES.
- The approved v7 runtime telemetry briefly entered the intended neighborhood:
  close step 2 had moving strict contact and counter 1mm slop contact, but
  strict counter contact remained NO. At close step 3, object speed rose to about
  `0.061935m/s` and `one_sided_push=YES`. By close step 4, counter slop contact
  was also NO.
- Final v7 close step 45
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:419` reports
  `target_error_m=0.023422`, moving contact YES, counter contact NO, counter
  slop contact NO, one-sided push YES, and reached NO.
- Aggregate line 420 reports approach/descent success but close failure, with
  `attach_calls=0`, `posewrite_calls=0`, telemetry-only YES, and success claim
  NO.
- Local static v7 analysis found slop-based candidates but no strict two-sided
  rigid contact candidates: 9/168 close-both-slop hits and 0/168 close-both-strict
  hits.
- Rerunning
  `sim_scripts/p7_branch_b_cube2cm_v6_static_runtime_contact_audit.py` locally
  reproduced the prior v4/v5 pattern: authored static designs can show two-sided
  contact, but logged runtime endpoints remain moving-only even with simple
  contact-patch margins up to 5mm.
- Existing project analyses independently identify the same sim-real gap:
  `sim_gap_analysis.py:190-210` marks deformable sponge contact vs rigid Isaac
  approximation as a critical contact-dynamics gap; `data_v5_crossvalidation_v2.py`
  treats 18-20deg as realistic sponge-held gripper state due to compliance.

Implication:

- The current blocker is close-time contact/dynamics, not TCP-only IK.
- Another small rigid offset can still pass static/prep while failing the same
  runtime one-sided-push mode. It should not be the default next branch unless it
  explicitly tests a new, falsifiable mechanism.
- The next primary branch should explicitly model foam/contact compliance:
  bounded contact-patch/slop abstraction, softer/contact-parameter diagnostic, or
  a true deformable/foam proxy if Isaac Lab support is practical.
- Keep gates separate: static/prep, USD conversion, runtime close contact,
  hold/lift, then only later dataset/training. Do not run hold-lift or dataset
  generation until close-time contact/dynamics passes.

Sources:

- `claudedocs/session_20260521_p7_branch_b_compliance_direction_analysis.md`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
- `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out`

## D058 — Compliance contact labels alone are not enough; future close_26 must change dynamics

Evidence:

- Added static-only analysis script
  `sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
  md5 `bd1f26da1d371e27b559528a6210a941`. It does not launch Isaac, train,
  generate datasets, edit defaults, insert constraints, attach SurfaceGripper,
  transport, release, tune gates, or claim success.
- The script encodes the rechecked v7 close_26 B200 telemetry samples:
  step 2 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:376`,
  step 3 `:377`, step 4 `:378`, step 5 `:379`, and final step 45 `:419`.
- Local run result:
  `required_budget_close_steps_2_to_4_m=0.001813`,
  `required_budget_close_steps_2_to_5_m=0.002911`, and
  `required_budget_final_step_45_m=0.014319`.
- Under the existing runtime push gate
  (`push_speed_gate_mps=0.005`, `push_drift_gate_m=0.00020`), step 3 and step 4
  still fail dynamically even if a 2mm contact/compression envelope relabels
  counter support through step 4. Step 3 speed is the previously verified
  `0.061935m/s`.
- A 15mm envelope would be required to relabel final step 45 counter support,
  which is outside the declared 5mm plausible diagnostic budget and would be
  contact-label overclaim rather than a foam grasp mechanism.

Implication:

- The next compliance branch must not only expand slop/contact labels. It must
  reduce the early asymmetric impulse/speed and keep the object inside the
  counter-support basin through at least close step 4.
- Future close_26-only runtime pass criteria should require both:
  two-sided support under the declared compliance model and no push-gate
  violation at close steps 2-4.
- Do not proceed to hold-lift, dataset generation, training, constraints,
  SurfaceGripper, transport/release, or gate tuning from a label-only pass.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:376-379,419`

## D059 — Mass-only inertia is not the next compliance proxy; use soft-contact dynamics first

Evidence:

- Added static-only design calculator
  `sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
  md5 `d43c93d2810dd56468e5d8b885013146`. It imports the verified v7 close_26
  samples from the static compliance audit and does not launch Isaac, run
  telemetry, train, generate datasets, edit defaults, insert constraints, attach
  SurfaceGripper, transport, release, tune gates, or claim success.
- Local `py_compile` passed.
- Local run reported:
  - step 3 speed `0.061935m/s`, allowed residual ratio `0.080730`;
  - step 4 speed `0.043783m/s`, allowed residual ratio `0.114200`;
  - step 5 speed `0.054294m/s`, allowed residual ratio `0.092091`;
  - required speed suppression across steps 3-5 `0.919270` (`91.9%`).
- Mass-only constant-impulse estimate rejected:
  with current diagnostic object mass `0.020kg`, holding steps 3-5 below the
  `0.005m/s` push-speed gate would require worst-case mass `0.247740kg`, above
  the declared `0.050kg` plausible diagnostic cap.
- With a 2mm support budget, step 4 counter support is possible
  (`step4_counter_gap_m=0.001813`), but step 5 and final support are not
  (`0.002911m` and `0.014319m`), and step 4 target error is already slightly
  outside the 3mm gate (`0.003151m`).

Implication:

- Do not spend the next runtime approval on mass-only object changes or
  contact-label-only expansion.
- The minimal future runtime mechanism, if separately approved, should be a
  soft-contact/material diagnostic that attempts to absorb the early asymmetric
  impulse while keeping counter support through step 4.
- Future pass criteria must include: step-3 speed below the existing push gate,
  no one-sided push through steps 2-4, counter support at step 4, close reached,
  `attach_calls=0`, `posewrite_calls=0`, and `success_claim=NO`.
- If soft-contact/material tuning cannot meet those telemetry changes, reserve
  the more artificial virtual-compression-plus-damping proxy; do not jump to
  hold-lift, constraints, SurfaceGripper, transport/release, dataset generation,
  or training.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
- `sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377-379,419`

## D060 — Soft-contact material diagnostic may be prepared only as default-off, falsifiable close_26 candidate

Evidence:

- Patched `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `7a261b72386ee549cb0ce162916597f7` to add a single default-off
  `--soft_contact_material_diagnostic` runtime candidate switch. The flag is not
  used unless a future runtime is separately approved.
- Baseline behavior is preserved when the flag is absent: object material and
  rigid-body constants remain the prior values (`static_friction=1.5`,
  `dynamic_friction=1.2`, `restitution=0.0`, solver iterations `8/1`,
  max depenetration velocity `5.0`).
- The soft-contact candidate changes only diagnostic object contact/material
  response: higher friction, more solver iterations, lower max velocities, and
  lower max depenetration velocity. It does not add constraints, SurfaceGripper,
  transport/release, attach posewrite, env default edits, chain default edits,
  dataset generation, training, or success claims.
- Added `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `a28c2fa8d8d58c617720f96417707677`, a local posthoc stdout-log audit. It
  does not launch Isaac.
- The audit now rejects wrong-mode logs by requiring metadata:
  `soft_contact_material_diagnostic=YES`,
  `object_physics mode=soft_contact_material_diagnostic`, and
  `runtime_candidate_requires_separate_approval=YES`.
- `python -m py_compile` passed for both scripts.
- Running the audit on `--use_v7_reference` intentionally returns FAIL, using
  the previously verified B200 log lines:
  close step 3 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377`,
  close step 4 `:378`, aggregate `:420`. The rejected criteria are:
  baseline/wrong-mode metadata, `close_reached=NO`, step-3 speed
  `0.061935m/s > 0.005m/s`, one-sided push at steps 3-4, and step-4 target error
  `0.003151m > 0.003m`.
- Running the audit on `--use_synthetic_pass_reference` returns PASS. This
  proves the criteria implementation is not hardwired to reject all inputs; it
  accepts a close_26 sample only when the fixed telemetry requirements are met.

Implication:

- The next separately approved runtime, if any, must be killable by telemetry
  rather than judged by qualitative contact labels.
- Required future pass criteria are fixed before the run: `approach_ok=YES`,
  `descend_ok=YES`, `close_reached=YES`, step-3 speed <= `0.005m/s`, no
  one-sided push through steps 2-4, step-4 counter gap <= `0.002m`, step-4 target
  error <= `0.003m`, `attach_calls=0`, `posewrite_calls=0`, and
  `success_claim=NO`.
- Do not treat the default-off code path as runtime approval. Do not proceed to
  hold-lift, dataset generation, training, constraints, SurfaceGripper,
  transport/release, or diagnostic gate tuning unless separately approved.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377-378,420`

## D061 — Record v7 as a dynamics failure pattern, not a generic grasp failure

Evidence:

- Added `claudedocs/p7_branch_b_cube2cm_failure_mode_register.md` as the
  reusable failure-mode register for Track A P7/Branch B cube grasp work.
- The register cites the verified v7 B200 failure chain:
  - line 38: strict diagnostic-only scope;
  - line 39: 3cm cube, `ik_ok=YES`, `max_fk_error_m=0.000518`;
  - line 68: intended link5 counter geometry with 1mm slop;
  - line 376: close step 2 has moving contact and counter slop support, but
    strict counter contact is still `NO`;
  - line 377: close step 3 starts the dynamic failure with
    `object_speed_mps=0.061935` and `one_sided_push=YES`;
  - line 378: close step 4 loses counter slop support and has
    `target_error_m=0.003151`;
  - line 419: final step remains moving-only with counter y-gap `0.014319m`;
  - line 420: aggregate `close_reached=NO`, `attach_calls=0`,
    `posewrite_calls=0`, `telemetry_only=YES`, `success_claim=NO`.
- The register explicitly separates narrow successes from physics success:
  v7 asset/prep/conversion is usable as a diagnostic platform; static compliance
  and synthetic audit checks are useful constraints; none are a real grasp pass.

Implication:

- The current failure should be recorded as a close-time contact/dynamics
  failure of the Isaac rigid-cube/jaw proxy, not as evidence that the real robot
  cannot grasp the foam cube.
- Future work should not repeat conversion-overclaim, rigid-offset-only probing,
  slop-label-only passes, mass-only inertia, or one-sided validators.
- A future success must show the specific telemetry transition: step-3 speed
  below `0.005m/s`, no one-sided push through steps 2-4, step-4 counter gap <=
  `0.002m`, step-4 target error <= `0.003m`, close reached, zero attach/posewrite,
  and no success claim.

Sources:

- `claudedocs/p7_branch_b_cube2cm_failure_mode_register.md`
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:38-39,68,376-379,419-420`

## D062 — Soft-contact candidate is statically ready for separate runtime approval, not pre-approved to run

Evidence:

- Added `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `1d022dbbcd57481d1fbf6763663c5041`.
- The readiness script is local/static only. It explicitly prints
  `isaac_run=NO`, `runtime_probe_executed=NO`, `training=NO`,
  `dataset_generation=NO`, `constraints=NO`, `surface_gripper=NO`,
  `transport_release=NO`, `gate_tuning=NO`, and `success_claim=NO`.
- Local `py_compile` passed.
- Local run reported `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` after verifying:
  - runtime probe soft-contact wiring exists and remains default-off;
  - the posthoc criteria audit has metadata guards;
  - the audit rejects the encoded v7 reference with return code 1;
  - the audit accepts the synthetic pass reference with return code 0;
  - the future candidate command includes `--variant v7`, `--close_deg 26.0`,
    and `--soft_contact_material_diagnostic`.
- The future command emitted by readiness is a proposal only and requires
  separate runtime approval. After B200 environment repair, the correct command
  uses the `isaacsim_5_1` micromamba env, not system Python or
  `./IsaacLab/isaaclab.sh -p`.
- The first posthoc analysis after any future approved run must be:
  `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out`.

Implication:

- The next executable runtime candidate is now precisely specified, but it is not
  approved by this static readiness pass.
- If approved later, success requires both correct metadata and fixed telemetry:
  step-3 speed <= `0.005m/s`, no one-sided push through close steps 2-4, step-4
  counter gap <= `0.002m`, step-4 target error <= `0.003m`, close reached, zero
  attach/posewrite, telemetry only, and no success claim.
- If the future approved run fails those criteria, record the failure before
  trying another mechanism; the next fallback should be explicit virtual
  compression plus damping, not rigid offsets, slop labels, mass-only inertia,
  hold-lift, dataset/training, constraints, SurfaceGripper, transport/release, or
  gate tuning.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`

## D063 — Approved soft-contact/material close_26 runtime failed; pivot to explicit compression plus damping

Evidence:

- User approved the next close_26-only soft-contact/material runtime.
- Two execution-command failures occurred before the valid run and are preserved:
  - direct system Python failed with `ModuleNotFoundError: No module named
    'isaaclab'` in
    `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_python_direct_fail_b200.err`
    md5 `4261bcab144070602917ac4e1ab228e1`;
  - `./IsaacLab/isaaclab.sh -p` failed because
    `IsaacLab/_isaac_sim/python.sh` was missing in
    `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_isaaclab_launcher_fail_b200.err`
    md5 `88e033670a9853c9b4c045a1e6d048d1`.
- The valid B200 run used
  `OMNI_KIT_ACCEPT_EULA=YES`, D024 NVML/Vulkan overrides, and the
  `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1`
  micromamba env.
- Valid stdout:
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out`,
  423 lines, md5 `c3c81c1e6d481f23fdbb35411987ea8a`.
- Valid stderr:
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.err`,
  4 lines, md5 `c0d91f52cb47b553b3d7746ac08995f8`; grep found no traceback,
  exception, fatal, segfault, driver mismatch, missing Isaac module, or missing
  Python executable. Stderr is limited to cpufreq/NVML/Fabric messages.
- Valid stdout line 37 confirms strict scope and
  `soft_contact_material_diagnostic=YES`.
- Line 39 confirms `mode=soft_contact_material_diagnostic`,
  `runtime_candidate_requires_separate_approval=YES`, friction `2.5/2.0`,
  solver iterations `16/4`, max linear velocity `2.0`, max angular velocity
  `5.0`, and max depenetration velocity `0.25`.
- Line 377 fails the decisive speed criterion:
  `object_speed_mps=0.049059` vs required `<=0.005`, with
  `one_sided_push=YES`.
- Line 378 still has `one_sided_push=YES`, counter y-gap `0.001989m`, and
  `target_error_m=0.003492` vs required `<=0.003`.
- Line 420 reports `future_close26_posthoc_pass=NO`.
- Line 421 reports aggregate failure:
  `approach_ok=YES`, `descend_ok=YES`, `close_reached=NO`, `attach_calls=0`,
  `posewrite_calls=0`, `telemetry_only=YES`, `success_claim=NO`.
- Updated posthoc audit md5 is `a28c2fa8d8d58c617720f96417707677`; it correctly
  rejects the valid soft-contact runtime log while confirming the metadata
  criteria pass.

Implication:

- The minimal material-only explanation is falsified for this proxy. It improved
  the step-3 speed from rigid-v7 `0.061935m/s` to `0.049059m/s`, about 20.8%
  suppression, but the required criterion needs `<=0.005m/s`, about 91.9%
  suppression from the original baseline.
- The candidate kept step-4 counter support barely inside the 2mm budget, but it
  did not prevent one-sided push and did not keep target error within the 3mm
  criterion.
- Do not spend more runtime on material-only friction/solver/depenetration
  changes unless a new falsifiable mechanism is added.
- Next branch should be static-first explicit virtual compression plus damping:
  bounded support/compression budget plus a mechanism that directly suppresses
  asymmetric close impulse before step 3.
- Hold-lift, dataset generation, training, constraints, SurfaceGripper,
  transport/release, and gate tuning remain blocked.

Sources:

- B200 `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out:37-39,67-68,376-379,419-421`
- B200 `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.err:1-4`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `claudedocs/p7_branch_b_cube2cm_failure_mode_register.md`

## D064 — Next mechanism must be explicit compression plus damping, with step-3 damping active

Evidence:

- Added static-only design script
  `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  md5 `aab11fb5ecaec645e49f4a9e34d9c185`.
- The script encodes the verified rigid-v7 and approved soft-contact B200
  close_26 telemetry samples and does not launch Isaac, run runtime, train,
  generate datasets, insert constraints, attach SurfaceGripper, transport/release,
  tune gates, or claim success.
- Local `py_compile` passed.
- Local run reported:
  - step 3 material-only suppression vs rigid v7: `0.207895` (20.8%);
  - step 3 still requires extra suppression from soft-contact result:
    `0.898082` (89.8%);
  - step 4 extra suppression required: `0.867381` (86.7%);
  - step 5 extra suppression required: `0.903574` (90.4%);
  - worst required extra suppression from the soft-contact result: `90.4%`;
  - step 4 counter gap remains barely supportable (`0.001989m <= 0.002m`), but
    step 4 target error fails (`0.003492m > 0.003m`);
  - step 5 is already outside the 2mm support budget (`0.003205m`).

Implication:

- More material-only friction/solver/depenetration changes are not the right next
  mechanism unless they introduce an explicit falsifiable damping/compression
  behavior.
- The next static/code design should model bounded compression plus damping:
  damping must be active by close step 3, support must remain bounded through
  step 4, and the posthoc runtime falsifier remains step-3 speed above gate,
  one-sided push at steps 2-4, or step-4 target error above gate.
- Do not proceed to hold-lift, dataset/training, constraints, SurfaceGripper,
  transport/release, or gate tuning.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
- B200 `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377-379`
- B200 `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out:377-379`

## D065 — Virtual compression plus damping is now the default-off next candidate; runtime remains separately unapproved

Evidence:

- Re-verified the approved soft-contact/material B200 runtime against the actual
  log. Stdout `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out`
  has md5 `c3c81c1e6d481f23fdbb35411987ea8a`; stderr md5 is
  `c0d91f52cb47b553b3d7746ac08995f8`.
- Soft-contact stdout line 37 confirms strict diagnostic scope and
  `soft_contact_material_diagnostic=YES`; line 39 confirms
  `mode=soft_contact_material_diagnostic` and
  `runtime_candidate_requires_separate_approval=YES`.
- Soft-contact line 377 still fails step-3 speed:
  `object_speed_mps=0.049059` and `one_sided_push=YES`; line 378 has step-4
  `target_error_m=0.003492`, counter y-gap `0.001989m`, and
  `one_sided_push=YES`; line 420 reports `future_close26_posthoc_pass=NO`;
  line 421 reports `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.
- Local static script
  `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  md5 `c45fb69a4cef556deaa87cb5247b4c73` now prints the proposed proxy:
  compression budget `0.002m`, max plausible compression `0.003m`, damping start
  close step `3`, residual velocity ratio `0.08`, and no attach/posewrite,
  constraints, SurfaceGripper, transport/release, env default edits, gate tuning,
  or success claim.
- The same local run projects damped speeds from the soft-contact result:
  step 3 `0.003925m/s`, step 4 `0.003016m/s`, and step 5 `0.004148m/s`, all
  below the `0.005m/s` speed gate; however step 4 target error remains a required
  runtime falsifier and step 5 is outside the 2mm support budget.
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py` md5
  `9e5292f176d9b90df30cfd23bdb36028` now has default-off
  `--virtual_compression_damping_diagnostic`, mutually exclusive with
  `--soft_contact_material_diagnostic`. The runtime candidate logs
  `virtual_compression_damping_diagnostic`, metadata mode
  `virtual_compression_damping_diagnostic`, bounded support eligibility,
  velocity-damping writes, and still keeps attach/posewrite counters separate.
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py` md5
  `fba03491e25bdd637c73dc90ca6a0836` now accepts
  `--expected_mechanism virtual_compression_damping_diagnostic`, rejects wrong
  metadata, rejects the encoded v7 reference, and accepts a synthetic pass.
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py` md5
  `dcec12b0b0063fb34115e3467d435a51` is still local/static only and printed
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` for the future command shape using
  `--virtual_compression_damping_diagnostic`. This is readiness, not runtime
  approval or physics success.
- Local `py_compile` passed for the runtime probe, audit, readiness, and static
  design scripts. No Isaac runtime, training, dataset generation, constraint,
  SurfaceGripper, transport/release, hold-lift, gate tuning, or success claim was
  run in this static/code pass.

Implication:

- The next mechanism is no longer material-only soft-contact. It is explicit
  virtual compression plus damping, and it remains a separately approved
  close_26-only runtime candidate.
- Any future runtime must be killed by wrong metadata, step-3 speed above
  `0.005m/s`, one-sided push in close steps 2-4, step-4 counter gap above
  `0.002m`, step-4 target error above `0.003m`, `close_reached=NO`,
  nonzero attach/posewrite, or `success_claim=YES`.
- Passing the static readiness script is not evidence of grasp success. It only
  means the future diagnostic command and posthoc falsifier are now specified.

Sources:

- B200 `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out:37-39,377-378,420-421`
- B200 `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.err:1-4`
- `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`

## D066 — Virtual compression+damping audit must require actual damping activation, not only metadata

Evidence:

- Code review found a falsifiability gap after D065: runtime probe logs
  `virtual_damping_active`, per-step `virtual_velocity_damping_writes_total`,
  and aggregate `virtual_velocity_damping_writes`, but the posthoc audit only
  required virtual metadata plus outcome gates. That could let a future log with
  correct virtual metadata but zero damping writes pass if other numbers happened
  to look good.
- Updated
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py` md5
  `065110aa514e49c62747fe4ab6ceecf4` to parse `virtual_support`,
  `virtual_damping_active`, `virtual_velocity_damping_writes_total`, and
  aggregate `virtual_velocity_damping_writes`.
- For `--expected_mechanism virtual_compression_damping_diagnostic`, the audit
  now requires positive aggregate damping writes, `virtual_support_step3=YES`,
  `virtual_damping_active_step3=YES`, and at least one damping write by close
  step 3. It still requires the D065 gates: correct metadata, close reached,
  no early kill, attach/posewrite zero, telemetry-only, no success claim,
  step-3 speed <= `0.005m/s`, no one-sided push in steps 2-4, step-4 counter
  gap <= `0.002m`, and step-4 target error <= `0.003m`.
- Added an embedded synthetic negative control:
  `--use_synthetic_virtual_no_damping_reference`. It has correct virtual
  metadata and passing numeric contact gates, but zero damping writes and
  `virtual_damping_active=NO`; the audit rejects it.
- Updated
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py` md5
  `04934025ecf5a4793002c2d9fed20b36` so local readiness requires the new audit
  checks and verifies the no-damping synthetic rejection.
- Local verification passed:
  `python -m py_compile` for runtime probe, audit, readiness, and static design;
  `python sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`;
  v7 reference audit with expected virtual mechanism returned FAIL;
  synthetic no-damping virtual reference returned FAIL;
  synthetic virtual pass returned PASS;
  readiness returned `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`;
  `git diff --check` passed.
- No Isaac runtime, training, dataset generation, hold-lift, constraints,
  SurfaceGripper, transport/release, gate tuning, or success claim was run.

Implication:

- A future virtual compression+damping runtime cannot pass merely by naming the
  mechanism. It must show the damping mechanism was active by close step 3 and
  wrote velocity damping at least once, while still satisfying the outcome
  falsifiers.
- Readiness remains only a command-shape/posthoc-contract check. It is not
  runtime approval and not physics success.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`

## D067 — Approved virtual compression+damping runtime partially fixed speed/one-sided push but still failed close_26

Evidence:

- User approved one close_26-only B200 runtime for Track A P7/Branch B
  `--virtual_compression_damping_diagnostic`. No training, cube sim dataset
  generation, hold-lift, constraints/default integration, SurfaceGripper,
  transport/release, gate tuning, or success claim was run.
- B200 stdout
  `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out`
  has md5 `7097b2c2eb70ba77d363dcfade601952`; stderr md5 is
  `35dc65de1f7982e1a7b1115784cff075`.
- Stdout line 37 confirms strict diagnostic scope, close_26 only,
  `soft_contact_material_diagnostic=NO`,
  `virtual_compression_damping_diagnostic=YES`, no constraints,
  SurfaceGripper, transport/release, training, gate tuning, hidden posewrite, or
  success claim.
- Lines 39-40 confirm `mode=virtual_compression_damping_diagnostic`,
  `runtime_candidate_requires_separate_approval=YES`, compression budget
  `0.002m`, max plausible compression `0.003m`, residual velocity ratio `0.08`,
  damping start close step `3`, `damping_writes_pose=NO`, and
  `damping_writes_velocity=YES`.
- Step 3, line 378, is the key partial improvement: pre-damping speed was
  `0.061935m/s`, logged speed after damping was `0.004955m/s`, support was YES,
  `virtual_damping_active=YES`, `virtual_velocity_damping_writes_total=1`, and
  `one_sided_push=NO`. This passes the step-3 speed gate and proves the virtual
  damping path actually activated.
- Step 4, line 379, remains a fail despite support and damping: speed
  `0.003203m/s` and counter y-gap `0.001794m` are within criteria, but
  `target_error_m=0.003130 > 0.003`.
- Step 5, line 380, shows why this is not a stable close-time mechanism:
  counter y-gap grows to `0.002738m`, `virtual_support=NO`,
  `virtual_damping_active=NO`, speed rebounds to `0.050912m/s`, and
  `one_sided_push=YES`.
- Final lines 421-422 report `future_close26_posthoc_pass=NO`,
  `close_reached=NO`, `virtual_velocity_damping_writes=2`, attach/posewrite
  zero, telemetry-only, and `success_claim=NO`.
- B200 posthoc audit returned FAIL. Passing checks included metadata, positive
  damping writes, step-3 speed below gate, step-3 support/damping activation, no
  one-sided push in steps 2-4, and step-4 counter support. Failing checks were
  `close_reached` and `target_step4_within_gate` (`0.003130 > 0.003`).
- Stderr lines 1-4 contained the known cpufreq/NVML/Fabric messages only; grep
  for traceback, exception, fatal, segfault, driver mismatch, missing module, and
  missing python returned no matches.

Implication:

- The virtual compression+damping mechanism is not a pass, but it is informative:
  explicit damping fixed the step-3 speed problem and removed one-sided push for
  the required steps 2-4.
- The remaining blocker is no longer simply "damping absent"; it is target-error
  control plus support/damping horizon. The mechanism drops out of support at
  step 5 and the close never reaches.
- Do not tune the 3mm target-error gate to rescue this. Do not jump to hold-lift,
  transport/release, constraints, SurfaceGripper, or training. The next work
  should be static/code-first failure attribution for target error and support
  retention before any further runtime approval.

Sources:

- B200 `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:37-40,68-69,378-382,421-422`
- B200 `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.err:1-4`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`

## D068 — Next static requirement is target-error control plus support/damping horizon, not stronger speed damping alone

Evidence:

- Added
  `sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
  md5 `0cccd8d9f3e5aaf7dc27fc3eb034967c`. It encodes the approved virtual
  runtime B200 lines 378-380 and 420, and is local/static only: no Isaac runtime,
  training, dataset generation, hold-lift, constraints, SurfaceGripper,
  transport/release, gate tuning, or success claim.
- The static analysis reports step-3 damping suppression `0.919997` with final
  step-3 speed `0.004955m/s`, support YES, damping active YES, and one-sided push
  NO.
- It reports step-4 damping suppression `0.919989` and speed `0.003203m/s`, but
  target error remains `0.003130m`, which exceeds the fixed 3mm gate by
  `0.000130m` (`0.130mm`).
- It reports step-5 support excess `0.000738m` (`0.738mm`) beyond the 2mm support
  budget. Step 5 is still within the declared 3mm max plausible compression by
  only `0.000262m`, but the current runtime turns damping off at that point, so
  speed rebounds to `0.050912m/s` and one-sided push returns.
- Final line 420 remains far outside any plausible support proxy:
  final counter y-gap `0.013828m`, which is `0.010828m` beyond the 3mm max
  plausible compression, with final target error `0.022778m`.

Implication:

- The next mechanism should not be "increase damping" in isolation. Speed damping
  already passed the early speed gate when active.
- The next static/code-first design must explain both target-error control below
  3mm and support/damping retention beyond step 4. It must still keep attach and
  posewrite zero and cannot rely on gate tuning.
- Any future runtime before that failure attribution would likely retest the same
  partial mechanism and is not justified.

Sources:

- `sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
- B200 `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:378-380,420`

## D069 — B200 endgame requires Track A preservation before Track B heavy training; next Track A mechanism is target-guarded micro-close plus support horizon

Evidence:

- User stated the Track B schedule for the remaining B200 window:
  backup pipeline test; B200 `openvla-oft` environment setup with
  `flash-attn==2.5.5` and HARD RULE #15 nightly cu128 recovery; 1K smoke with
  `action_dim=6` and `image=top`; OpenVLA-OFT 30K-50K finetune selected from
  smoke time/step; offline eval and final backup; and pi0 RunPod handoff after
  B200 release around 2026-05-22 23:59.
- Track B remains separate from Track A P7/Branch B. Track B training results
  must not overwrite Track A runtime/contact verdicts.
- Local backup state includes untracked `b200_backup_20260521/`. During
  inspection it had `env.sh` and a growing rsync-style temporary file
  `._speedtest_model.safetensors.MIJ5aq`; final check later showed only
  `env.sh` remaining. Do not treat the transient temp file as a completed backup
  artifact.
- Added
  `claudedocs/b200_endgame_track_a_preservation_track_b_plan_20260521.md` to
  record the Track B phases, Track A `/tmp` logs to preserve, backup guardrails,
  and the separation rule.
- Added
  `sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
  md5 `dca5322e654f3b0d415822f0972d383e`; `py_compile` passed and local static
  run completed.
- The static design rejects stronger damping alone:
  step 4 still exceeds the fixed target gate by `0.130mm`, and step 5 target
  excess is `1.843mm`.
- It rejects support-label-only:
  final counter y-gap is `0.013828m`, which is `0.010828m` beyond the 3mm max
  plausible compression.
- It selects the next mechanism shape as default-off target-guarded micro-close
  plus support-horizon damping, with unchanged fixed audit gates:
  3mm target gate, 2mm step-4 support budget, no attach/posewrite, no
  constraints, no SurfaceGripper, no transport/release, no env default edits, no
  gate tuning, and no success claim.

Implication:

- Before Track B consumes B200 with long OpenVLA-OFT runs, preserve Track A logs,
  docs, code, and USD artifacts with md5 manifests.
- Do not rerun the same Track A virtual compression+damping parameters.
- The next Track A code work, if approved, should implement a default-off
  target-guarded micro-close/support-horizon diagnostic. It must be falsifiable
  by step-4 target error, step-5 support/horizon loss, close_reached=NO, or
  attach/posewrite nonzero.
- `MEMORY.md` remains an index/hard-rule source; project truth stays in
  `START_HERE.md`, `DECISIONS.md`, ledger, session docs, and B200 logs.

Sources:

- `claudedocs/b200_endgame_track_a_preservation_track_b_plan_20260521.md`
- `sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
- B200 `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:378-380,420`

## D070 — Target-guarded v2 convergence fixed backlog blowout but starved close progression

Evidence:

- User approved one close_26-only B200 runtime for Track A P7/Branch B
  `--target_guarded_micro_close_v2_convergence_diagnostic`. No training,
  dataset generation, hold-lift, constraints/default integration,
  SurfaceGripper, transport/release, gate tuning, or success claim was run.
- Local v2 code md5s synced to B200 before runtime:
  runtime probe `5446716a908d0869c0c308d22af0eb75`, criteria audit
  `baf1cbec4f8a837458e3695a158a129c`, readiness
  `ca34226d94db9ff09231a84fee8ab1bf`, and static attribution
  `7114699126c3f24f5ba4523ba0439e7f`.
- B200 stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out`
  has md5 `52fa5cf2cc0cc5dbdc2f55f0d099611f`; stderr md5 is
  `9061693c9914e735b53a19417cdebb9c`.
- Stdout line 37 confirms strict diagnostic scope, close_26 only,
  `target_guarded_micro_close_v2_convergence_diagnostic=YES`, no training,
  constraints, SurfaceGripper, transport/release, gate tuning, hidden posewrite,
  or success claim.
- Stdout line 39 confirms
  `mode=target_guarded_micro_close_v2_convergence_diagnostic` and
  `runtime_candidate_requires_separate_approval=YES`.
- Stdout line 41 confirms the v2 contract: zero-backlog hold, close command
  writes only, posewrite/constraints/SurfaceGripper NO, and advance requires
  command convergence, support margin, and non-worsening target error.
- Step 3, stdout line 379, shows the intended convergence behavior:
  `object_speed_mps=0.000128`, `target_guarded_command_backlog_deg=0.000`,
  command converged YES, support margin YES, target non-worsening YES,
  support horizon YES, damping active YES, and one-sided push NO.
- The posthoc audit returned FAIL. Audit stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_audit_b200.out`
  has md5 `563a9194dfc1cbe611aa38b9bee45dd3`; audit stderr md5 is empty
  `d41d8cd98f00b204e9800998ecf8427e`.
- Audit line 14 fails `close_reached`; audit line 24 also fails
  `virtual_support_step3` because stdout line 379 has `virtual_support=NO` and
  `virtual_compression_gap_max_m=0.002262`.
- Runtime summary line 422 reports `future_close26_posthoc_pass=NO`,
  `virtual_velocity_damping_writes=43`, `target_guarded_close_advances=17`,
  `target_guarded_close_holds=28`, and `target_guarded_zero_backlog_holds=28`.
- Aggregate line 423 reports `approach_ok=YES`, `descend_ok=YES`,
  `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.
- Final close line 421 shows the main failure shape: target error remains within
  the fixed gate at `0.001921m`, object speed is low at `0.000527m/s`,
  one-sided push is NO, support horizon is YES, but gripper actual is only
  `6.087deg` with command `6.089deg`, leaving `19.913deg` to the 26deg target.
- The first support-margin block appears at stdout line 410, step 34:
  `target_guarded_support_margin_ok=NO`, counter y-gap `0.001583m` just above
  the v2 advance margin `0.0015m`, while the broader fixed support budget is
  still `0.002m` and max plausible horizon remains `0.003m`.

Implication:

- v2 is a useful negative result: zero-backlog holds prevented the v1 backlog
  blowout and avoided one-sided push, but the mechanism over-constrained close
  progression and starved the gripper before reaching close_26.
- Do not treat "stable, no one-sided push" as a pass. `close_reached=NO` remains
  a hard fail, and the audit also exposed an early `virtual_support_step3` gap.
- Do not rerun the same v2 parameters, do not loosen fixed audit gates, and do
  not jump to hold-lift, dataset generation, training, constraints, or
  SurfaceGripper.
- The next Track A work should be static/code-first: preserve v2's stability
  while adding a bounded progress mechanism or true compliant counter behavior
  that can reach close_26 under the unchanged audit.

Sources:

- B200 `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out:37,39,41,379-381,410,421-423`
- B200 `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.err:1-4`
- B200 `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_audit_b200.out:1-41`
- `claudedocs/session_20260522_track_a_v2_convergence_runtime_fail.md`

## D071 — OpenVLA-OFT 7B on stock transformers ≥4.50 requires `_supports_sdpa` class-attr patch

Evidence:

- HF Hub `openvla/openvla-7b/modeling_prismatic.py` (and openvla-oft's local
  fork `prismatic/extern/hf/modeling_prismatic.py`) define
  `PrismaticPreTrainedModel._supports_sdpa` as `@property` that reads
  `self.language_model._supports_sdpa`. This was written for transformers 4.40.
- transformers ≥4.50 (we tested 4.57.6) calls `self._sdpa_can_dispatch(...)`
  inside `PreTrainedModel.__init__` (`modeling_utils.py:2076`), which happens
  **before** the openvla subclass assigns `self.language_model = ...`.
- Verified on 2026-05-22 B200 smoke: error `AttributeError:
  'OpenVLAForActionPrediction' object has no attribute '_supports_sdpa'`.

Implication:

- For openvla-oft on any modern transformers, replace the @property with a
  class-level `_supports_sdpa: bool = True` attribute. Llama2-7B supports sdpa,
  so the static True is correct.
- The patch must be applied **both** to the HF Hub snapshot file
  (`.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/<sha>/modeling_prismatic.py`)
  and to the active `transformers_modules/_<sha>/modeling_prismatic.py` cache.
  `update_auto_map()` re-copies from the hub snapshot into transformers_modules
  on every `from_pretrained`, so patching only one breaks on next run.

Sources:

- `claudedocs/session_20260522_openvla_oft_7b_30k_lora_complete.md`
- B200 `/tmp/openvla_oft_v6_smoke.out` initial fail trace
- patched file `models--openvla--openvla-7b/snapshots/<sha>/modeling_prismatic.py:207`

## D072 — OpenVLA-OFT requires the fork modeling files over HF Hub originals

Evidence:

- HF Hub `openvla/openvla-7b/modeling_prismatic.py` is the original OpenVLA-1
  modeling code: no `set_num_images_in_input`, `get_num_images_in_input`,
  `get_num_patches`, `FiLMedPrismaticVisionBackbone` integration, or
  action-chunking hooks.
- openvla-oft's fork at `prismatic/extern/hf/modeling_prismatic.py` adds these
  OFT-specific methods.
- Original `finetune.py:846` calls
  `vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)`
  immediately after `from_pretrained(..., trust_remote_code=True)`. With the
  hub modeling, this raises `AttributeError`.

Implication:

- After the first model download, **copy the fork's `modeling_prismatic.py`,
  `configuration_prismatic.py`, and `processing_prismatic.py` over the HF Hub
  snapshot files**, keeping `*.hub_orig.bak` backups.
- `update_auto_map()` will re-propagate them to `transformers_modules` cache
  automatically on next `from_pretrained` call.
- Combine with D071 — patch `_supports_sdpa` after the fork-file overwrite,
  not before (the fork file has the same @property defect).

Sources:

- `claudedocs/session_20260522_openvla_oft_7b_30k_lora_complete.md`
- `code/openvla-oft/prismatic/extern/hf/modeling_prismatic.py` (has
  `def set_num_images_in_input` at line 177); HF Hub snapshot original does not.

## D073 — `merge_lora_during_training=True` hangs with PEFT 0.18 on openvla-oft; default to False

Evidence:

- 2026-05-22 B200 smoke (PEFT 0.18.0): set `merge_lora_during_training=True`
  (openvla-oft default). After the step-5 save, the merge step
  (`PeftModel.from_pretrained` → `merge_and_unload` → `save_pretrained` of a
  ~14 GB merged 7B base) hung at 99% CPU with `0% GPU` utilization and no new
  files written for 22+ minutes; only the LoRA adapter and action head from
  the pre-merge save were on disk.
- Killed and re-ran with `--merge_lora_during_training False`: full 30K steps
  + 12 checkpoint saves completed in 2h 23min on B200 with no hang.

Implication:

- For openvla-oft training on PEFT ≥0.18, always set
  `--merge_lora_during_training False`. Merge offline after training using a
  dedicated script (`vla-scripts/merge_lora_weights_and_save.py`).
- Per-checkpoint merged save also adds ~14 GB NFS write each save_freq, which
  is expensive even when not hanging.

Sources:

- `claudedocs/session_20260522_openvla_oft_7b_30k_lora_complete.md`
- B200 `/tmp/openvla_oft_v6_smoke.out` smoke #4 hang vs smoke #5 success
- `outputs/openvla_oft_v6_b200/...--*_chkpt/` size (679 MB; lora_adapter +
  action_head only) under `merge_lora_during_training=False`.

## D074 — LeRobot v3 → openvla-oft path: thin `LeRobotV3RLDSCompatDataset` instead of RLDS conversion

Evidence:

- openvla-oft `finetune.py` imports `RLDSDataset` (TFDS-backed IterableDataset)
  and `RLDSBatchTransform`. Default data path expects RLDS TFRecord trees built
  via moojink's `rlds_dataset_builder` repo.
- v6 LeRobot is parquet + AV1 video; converting to RLDS would require a
  custom builder + TFDS rebuild step (estimated 3-6 h debug).
- A map-style Dataset that yields the exact RLDS batch dict schema —
  `{"dataset_name", "action": (chunk, dim), "observation": {"image_primary":
  (1, H, W, 3) uint8, "proprio"?: (dim,)}, "task": {"language_instruction":
  bytes}}` — can be passed through the existing `RLDSBatchTransform` callable
  without changing anything else in the original pipeline.
- 2026-05-22 implementation: `openvla_oft_roarm/lerobot_rlds_compat.py`
  (~160 lines). 6942 frames index loads in < 1 s; per-sample fetch ~0.7 s
  (AV1 decode dominated). With `num_workers=4 persistent_workers=True
  pin_memory=True drop_last=True`, throughput reached 3.84 it/s @ batch=8 on
  B200 and data was never the bottleneck.

Implication:

- For openvla-oft on any LeRobot v3 dataset: skip RLDS conversion entirely.
- Apply BOUNDS_Q99 normalization using `stats.json` q01/q99 directly:
  `clip(2*(x - q01)/(q99 - q01 + 1e-8) - 1, -1, 1)`. Matches openvla-oft's
  `prismatic/vla/datasets/rlds/utils/data_utils.py:67-81`.
- Use LeRobot 0.4.x API: `ds.meta.episodes["dataset_from_index"]` /
  `["dataset_to_index"]` instead of `ds.episode_data_index` (deprecated).
- Wrap the original `for batch_idx, batch in enumerate(dataloader)` with an
  infinite generator so finite-dataset epoch boundaries don't terminate
  `max_steps`. ROARM_M3 constants path requires `roarm` in `sys.argv` for
  `prismatic/vla/constants.py:detect_robot_platform()`; ensure `--dataset_name`
  contains `roarm`.

Sources:

- `openvla_oft_roarm/lerobot_rlds_compat.py`
- `openvla_oft_roarm/train_roarm_v6.py`
- `claudedocs/session_20260522_openvla_oft_7b_30k_lora_complete.md`
- openvla-oft `prismatic/vla/datasets/datasets.py:36-91` (RLDSBatchTransform
  input schema)

## D075 — Track A target-guarded v2 fails primarily by zero-backlog pulse starvation

Evidence:

- Track A B200 target-guarded v2 close_26 stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out`,
  md5 `52fa5cf2cc0cc5dbdc2f55f0d099611f`.
- Runtime line 41 confirms the v2 mechanism: 2deg micro close, 0.75deg command
  convergence gate, 0.0015m advance support margin, non-worsening target-error
  gate, and zero-backlog holds.
- Runtime lines 377-379 show the core pulse pattern: line 377 advances, line 378
  has actual gripper `0.361deg` against command `2.000deg` with backlog
  `1.639deg`, and line 379 has the command reset to actual with backlog
  `0.000deg`.
- Runtime line 409 is the last advance (`target_guarded_close_advances_total=17`).
  Runtime line 421 ends at gripper actual `6.087deg`, command `6.089deg`,
  target error `0.001921m`, support horizon YES, virtual support YES, but support
  margin NO.
- Runtime lines 422-423 report posthoc FAIL, 17 advances, 28 holds,
  28 zero-backlog holds, close_reached NO, attach/posewrite zero, telemetry-only,
  and no success claim.
- Static script
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v2_progress_static_analysis.py`
  md5 `7269b126b9aa1b6ce2da75e67f78702c` verifies the B200 md5 and computes:
  average actual motion after each advance `0.360deg`, average next-step backlog
  before zeroing `1.641deg`, and discarded fraction `0.820` of each 2deg micro
  command.
- The same static run projects that even the maximum alternating advance count
  in 45 close steps (`23`) would reach only `8.279deg`, leaving `17.721deg` to
  the 26deg target.
- First support-margin block appears at runtime line 410: counter gap
  `0.001583m`, only `0.000083m` above the v2 margin `0.0015m`, while still
  inside the fixed support budget `0.002m`, horizon `0.003m`, target gate
  `0.003m`, and one-sided push is NO.

Implication:

- Do not treat v2 as grasp success or close_26 success.
- Do not rerun the same v2 parameters.
- Do not "fix" this by relaxing fixed close_26 audit gates.
- The 0.0015m support margin is too strict as an advance blocker, but support
  margin relaxation alone is insufficient: the current pulse/reset schedule is
  already mathematically unable to reach 26deg in 45 steps.
- The next Track A mechanism must provide structural actual-progress behavior:
  do not discard the micro-close command backlog after one physics step; instead
  advance or settle until bounded actual gripper progress occurs, and rollback
  only on real safety degradation while preserving fixed close_26 audit gates.

Sources:

- `claudedocs/session_20260522_track_a_v2_progress_starvation_static_analysis.md`
- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v2_progress_static_analysis.py`
- B200 stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out:37,39,41,377-379,402,409-411,421-423`

## D076 — Track A v3 must preserve backlog and prove actual progress before runtime

Evidence:

- D075 established that v2 failed primarily by zero-backlog pulse starvation:
  B200 lines 377-379 show a 2deg command becoming only `0.361deg` actual motion
  before `1.639deg` backlog was discarded; line 421 ended at only `6.087deg`.
- Support margin `0.0015m` was strict but secondary: B200 line 410 had counter
  gap `0.001583m`, only `0.000083m` above that margin, while still inside the
  fixed `0.002m` support budget and `0.003m` support horizon.
- Added default-off runtime candidate
  `--target_guarded_micro_close_v3_progress_diagnostic` in
  `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `9cdfd04876078110186435bb15ba34ab`.
- v3 keeps the 0.0015m margin as warning telemetry, but hard support for advance
  uses the fixed close_26 support budget (`0.002m`) plus the existing support
  horizon (`0.003m`).
- v3 no longer zeroes backlog during normal holds. It logs backlog-preserved
  holds, actual progress, projected backlog, safety state, and safety rollbacks.
  Command rollback to actual is reserved for safety degradation.
- Updated audit
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `8b75c9d7d678419d0d1c96bf61115aed` with expected mechanism
  `target_guarded_micro_close_v3_progress_diagnostic`.
- v3 audit keeps fixed close_26 gates unchanged and additionally requires
  positive backlog-preserved holds, zero zero-backlog holds, zero safety
  rollbacks, and step3 actual progress >= `0.25deg`.
- Updated readiness
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `b853e0f2198d1d005ac72bc1e83dafcd`.
- Local/static verification passed: `py_compile`, v3 synthetic pass PASS, v3
  no-damping FAIL, v3 zero-backlog FAIL, old v2 B200 stdout audited as v3 FAIL,
  old v2 synthetic PASS, old target-guarded v1 synthetic PASS, readiness
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`, and `git diff --check`.

Implication:

- v3 is not a success claim. It is the next close_26-only runtime candidate,
  requiring separate approval.
- Do not rerun v2 or tune the 0.0015m support margin as the next action.
- Do not relax fixed close_26 audit gates.
- If approved, the next Track A runtime is exactly one close_26-only v3 run,
  followed immediately by the posthoc audit with expected mechanism
  `target_guarded_micro_close_v3_progress_diagnostic`.
- Hold-lift, dataset generation, training, transport/release, constraints, and
  SurfaceGripper remain blocked until close_26 audit PASS.

Sources:

- `claudedocs/session_20260522_track_a_v3_progress_candidate_static_readiness.md`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- B200 stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out:377-379,410,421-423`

## D077 — Track A v3 fixed zero-backlog starvation but failed on target-pose safety rollback

Evidence:

- A first B200 command using
  `--target_guarded_micro_close_v3_progress_diagnostic` failed before Isaac
  because the remote code did not yet include the v3 flag. Preserved preflight
  stderr md5: `acbbdfe97f41fe0a2130816a4c281d63`; preflight stdout was empty
  md5 `d41d8cd98f00b204e9800998ecf8427e`.
- After rsync to B200, runtime/audit/readiness md5s were
  `9cdfd04876078110186435bb15ba34ab`,
  `22db78e81d25804cc6ed26ccbe608579`, and
  `b853e0f2198d1d005ac72bc1e83dafcd`.
- After the v3 runtime FAIL, readiness was updated and synced to md5
  `5675db108ac15de6f333caf2d2e9ce9d` so it blocks v3 reruns with
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=NO`.
- The actual close_26-only v3 runtime exited 0 but failed audit. Runtime stdout
  md5: `5f2d1a626edcdccce8086fafd321c9af`; stderr md5:
  `13671d0ae55c7faee9ae90a4e8c242c6`. Final audit stdout md5:
  `ca60c09b03a156c85197e34ec7b28bb5`; audit stderr empty md5:
  `d41d8cd98f00b204e9800998ecf8427e`.
- Runtime line 37 confirms diagnostic-only, close_26-only, no training,
  constraints, SurfaceGripper, transport/release, gate tuning, posewrite, or
  success claim. Line 39 confirms mode
  `target_guarded_micro_close_v3_progress_diagnostic` with separate approval.
  Line 41 confirms v3 zero-backlog hold NO, backlog preserve YES, fixed hard
  support budget/horizon, and rollback only on safety degradation.
- v3 fixed the v2 zero-backlog failure mode: runtime line 422 reports
  `target_guarded_zero_backlog_holds=0` and
  `target_guarded_backlog_preserved_holds=5`; audit lines 23-24 and 42-43 pass
  those checks.
- v3 did not pass close_26: audit line 15 is `close_reached pass=NO`, line 25 is
  `target_guarded_v3_safety_rollbacks_zero pass=NO value=34`, and line 46 is
  `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- First safety rollback is runtime line 387 step 11: gripper `6.890deg`,
  command `10.000deg`, backlog `3.111deg`, target error `0.002769m` exceeding
  the v3 design limit `0.0027m`; speed is `0.001647m/s`, support budget YES,
  support horizon YES, one-sided push NO.
- Last advance is line 391 step 15: gripper `6.878deg`, target error
  `0.002698m`, safety rollback count already 4.
- Peak target/support excursion is line 392 step 16: gripper `7.235deg`, target
  error `0.003070m` exceeding the fixed `0.003m` target gate, and counter gap
  `0.002074m` exceeding the fixed `0.002m` support budget.
- Final line 421 remains far from close_26: gripper `7.144deg`, command
  `7.147deg`, remaining close `18.856deg`, target error `0.002872m`, safety
  rollback YES. Aggregate line 423 reports `close_reached=NO`, attach/posewrite
  zero, telemetry-only YES, success_claim NO, 6 advances, 39 holds, 34 safety
  rollbacks.
- Added static attribution script
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v3_progress_runtime_static_analysis.py`
  md5 `b3c446c1872127b19b49af929ded95ce`. It verifies the runtime/audit md5s and
  reports primary attribution `target_pose_error_safety_rollback_after_progress`.

Implication:

- v3 is not grasp success and not close_26 success.
- Do not rerun v2 or v3 as the next experiment.
- Do not respond by relaxing fixed target/support gates: line 392 already
  exceeds both the fixed target gate and fixed support budget.
- Do not reintroduce zero-backlog holds; v3 fixed that specific failure mode.
- Next work must be local/static/code-first: design a contact-compatible close or
  target-error recovery mechanism that preserves fixed close_26 gates and proves
  no safety rollback before any hold-lift, dataset generation, or training.

Sources:

- `claudedocs/session_20260522_track_a_v3_progress_runtime_fail.md`
- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v3_progress_runtime_static_analysis.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- B200 stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out:37,39,41,377-392,421-423`
- B200 audit stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_audit_b200.out:1-46`

## D078 — OpenVLA-OFT inference requires patching ALL transformers_modules cache locations, not just hub snapshot

Evidence:

- 2026-05-22 12:00 KST offline-eval dry-run 1 crashed with
  `AttributeError("'OpenVLAForActionPrediction' object has no attribute '_supports_sdpa'")`
  on stock transformers 4.57.6, despite D071's hub-snapshot patch being
  verified at
  `$HF_HOME/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/modeling_prismatic.py:311`
  (`_supports_sdpa: bool = True`, md5 `8c2223ab`).
- Investigation found a second `modules/transformers_modules/openvla/openvla_hyphen_7b/47a0ec7fc4ec123775a391911046cf33cf9ed83f/modeling_prismatic.py`
  (md5 `0e1ea109`, line 208 `@property def _supports_sdpa`) — stock prismatic,
  NOT the openvla-oft fork. Transformers ≥4.57 uses the canonical
  `<owner>/<repo_with_hyphens>/<commit>/` path, so the patched copy at
  `_<sha>/` was never read at inference.
- The training run worked because training was invoked via a different code
  path that happened to hit the `_<sha>/` cache first, or executed early enough
  that the canonical cache hadn't been written. Inference triggers the
  canonical path consistently.

Implication:

- D071's "patch hub snapshot AND transformers_modules cache" must be done at
  all four locations:
  - `hub/.../snapshots/<commit>/`
  - `transformers/.../snapshots/<commit>/` (legacy `TRANSFORMERS_CACHE`)
  - `modules/transformers_modules/_<commit>/`
  - `modules/transformers_modules/<owner>/<repo_with_hyphens>/<commit>/`
- After overwriting, invalidate `__pycache__/*.pyc` files in each location;
  Python will recompile from the patched .py on next import.
- Preserve `.preserve_orig.bak` of each overwritten file so the patch is
  reversible.
- Add `local_files_only=True` and pin `revision=<commit_sha>` in
  `from_pretrained` calls to block transformers from re-fetching fresh fork
  files mid-load (dry-run 1 had also re-downloaded
  `processing_prismatic.py` from hub, overwriting our patched fork copy).

Sources:

- `claudedocs/session_20260522_openvla_oft_offline_eval_v6_in_progress.md`
- B200 `/tmp/openvla_oft_v6_eval_dryrun.{out,err}`,
  `/tmp/openvla_oft_v6_eval_dryrun2.err`
- `openvla_oft_roarm/eval_offline_v6.py:62-99` (apply_sdpa_class_attr_patch
  fallback)
- `openvla_oft_roarm/eval_offline_v6.py:148-175` (load_vla_with_lora pinned
  revision + local_files_only)

## D079 — openvla-oft `action_head--*_checkpoint.pt` is saved DDP-wrapped; strip `module.` prefix when loading for inference

Evidence:

- 2026-05-22 12:03 KST offline-eval dry-run 2 crashed loading ckpt 2500
  action_head: `RuntimeError: Missing key(s) "model.layer_norm1.weight" ...
  Unexpected key(s) "module.model.layer_norm1.weight" ...`.
- All 12 v6 checkpoints (2500..30000) saved via the project's
  `train_roarm_v6.py` show the same `module.` prefix; the training wrapper
  saves the DDP/accelerate-wrapped state_dict directly without unwrapping
  even though training used `nproc_per_node=1` (no actual DDP).
- L1RegressionActionHead expects keys like `model.layer_norm1.weight` (no
  `module.` prefix).

Implication:

- Any inference / eval / deploy script that loads our v6 OFT action_head must
  strip a leading `module.` prefix from every key before
  `load_state_dict`. Pattern:
  `state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}`
- This rule also applies to any new openvla-oft finetuning runs we do that
  reuse `train_roarm_v6.py` style saving.
- Do not patch the training save side to fix this; doing so would break the
  reproducibility of the already-saved 12 v6 checkpoints. Strip on load
  instead.

Sources:

- `claudedocs/session_20260522_openvla_oft_offline_eval_v6_in_progress.md`
- B200 `/tmp/openvla_oft_v6_eval_dryrun2.out` (RuntimeError)
- `openvla_oft_roarm/eval_offline_v6.py:248-257` (load_vla_with_lora
  action_head DDP-prefix strip)

## D080 — `norm_stats` must be assigned to `vla.base_model.model`, not the PeftModel wrapper, for openvla-oft `predict_action` to see the correct unnorm key

Evidence:

- 2026-05-22 12:05 KST offline-eval dry-run 3 crashed inside
  `predict_action`'s `_check_unnorm_key`:
  `AssertionError("The unnorm_key you chose is not in the set of available
  dataset statistics, please choose from: dict_keys([...25 OpenVLA pretraining
  datasets...])")`.
- The eval script had set `vla.norm_stats = norm_stats` (PeftModel
  attribute). Python's attribute setter writes to the PeftModel instance dict,
  but `predict_action` is a bound method on the underlying prismatic model
  (`vla.base_model.model`), and reads `self.norm_stats` where `self` is the
  prismatic model — which still held the OpenVLA pretraining defaults
  (austin_buds, bc_z, bridge_orig, etc.).
- Setting on `vla.base_model.model.norm_stats` directly resolved the error
  and inference proceeded normally.

Implication:

- For PEFT-wrapped openvla-oft inference, **always** set norm_stats on
  `vla.base_model.model` (the actual `OpenVLAForActionPrediction` instance).
  Setting it on the PeftModel or LoraModel layer is silently ignored by
  `predict_action`.
- Same caveat applies to any future override of attributes accessed inside
  prismatic methods (e.g., `pad_token_id`, `num_images_in_input`): walk down
  to the underlying model.

Sources:

- `claudedocs/session_20260522_openvla_oft_offline_eval_v6_in_progress.md`
- B200 `/tmp/openvla_oft_v6_eval_dryrun3.out` (AssertionError)
- `openvla_oft_roarm/eval_offline_v6.py:175-191` (load_vla_with_lora
  norm_stats wiring)
- `prismatic/extern/hf/modeling_prismatic.py:1068-1080`
  (`_check_unnorm_key` reads `self.norm_stats`)

## D081 — Track A v4 must recover target error before further close advance, not rollback or relax fixed gates

Evidence:

- v3 B200 stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out`
  reverified md5 `5f2d1a626edcdccce8086fafd321c9af`; final audit stdout
  reverified md5 `ca60c09b03a156c85197e34ec7b28bb5`.
- v3 fixed the v2 zero-backlog failure mode but failed close_26: stdout line
  422 reports `target_guarded_zero_backlog_holds=0`,
  `target_guarded_backlog_preserved_holds=5`, and
  `target_guarded_safety_rollbacks=34`; aggregate line 423 reports
  `close_reached=NO`, attach/posewrite zero, telemetry-only YES, and
  success_claim NO. Audit lines 15/25/46 fail `close_reached`, safety
  rollbacks zero, and final PASS.
- Static v4 design script
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v4_recovery_static_design.py`
  md5 `265391a9a421bb7535925a77ef3e5b37` verifies those md5s and identifies
  the first structural intervention at stdout line 385 step 9: v3 advanced
  while `target_guarded_target_nonworsening=NO`; target error grew by
  `0.000546m`, above the existing `0.000250m` growth tolerance.
- The same static check reclassifies stdout line 387 step 11
  (`target_error_m=0.002769`, fixed target gate `0.003m`, counter gap
  `0.001909m`, fixed support budget `0.002m`) as recoverable: v4 should
  preserve backlog and recovery-hold, not rollback the command to actual.
- Stdout line 392 step 16 already exceeds both fixed gates
  (`target_error_m=0.003070 > 0.003`, counter gap `0.002074m > 0.002`), so
  v4 must treat that as a hard audit fail if reached; it must not relax target
  or support gates to pass.
- Added default-off runtime candidate
  `--target_guarded_micro_close_v4_recovery_diagnostic` in runtime probe md5
  `2326b68cf5fc7098182b574b4f7a1eb1`, audit md5
  `7f3b368460d26acb3da549ace3e4b25f`, and readiness md5
  `db3a8a48ba17cea7570d8e9c45d028e7`. Readiness is local/static only and
  prints `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`; no Isaac runtime was run.

Implication:

- Do not rerun v2 or v3 as the next experiment.
- Do not reintroduce zero-backlog holds; v2 already proved that starves close
  progress.
- Do not use target/support gate relaxation to rescue v3/v4. Fixed gate breach
  remains failure evidence, not a pass condition.
- The next Track A runtime, if separately approved, is exactly one close_26-only
  v4 recovery run followed immediately by v4 posthoc audit. Hold-lift,
  dataset generation, training, transport/release, constraints, and
  SurfaceGripper remain blocked until close_26 audit PASS.

Sources:

- `claudedocs/session_20260522_track_a_v4_recovery_static_readiness.md`
- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v4_recovery_static_design.py`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out:385,387,392,421-423`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_audit_b200.out:1-46`

## D082 — Track A v4 fixed rollback starvation but failed by target/support hard-freeze

Evidence:

- User separately approved exactly one close_26-only B200 runtime with
  `--target_guarded_micro_close_v4_recovery_diagnostic`, followed immediately by
  the v4 posthoc audit. No dataset generation, training, hold-lift,
  transport/release, constraints, SurfaceGripper, gate tuning, or success claim
  was run.
- B200 stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out`
  has md5 `fe6a733727a6eeb288c6c6464c178af1`; stderr md5 is
  `4dc0d3c542e38524807f8fe75a82f841`.
- B200 audit stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_audit_b200.out`
  has md5 `47f4ec7b78298fde0a46ac57105a6e6c`; audit stderr is empty md5
  `d41d8cd98f00b204e9800998ecf8427e`.
- Runtime line 37 confirms strict diagnostic-only, close_26-only, v4 flag YES,
  no forbidden mechanisms, no posewrite, and no success claim. Line 39 confirms
  `mode=target_guarded_micro_close_v4_recovery_diagnostic` with separate
  approval. Line 41 confirms v4 recovery target error `0.002400m`, zero-backlog
  hold NO, recovery holds preserve backlog YES, rollback on safety degradation
  NO, and hard safety violation fails candidate YES.
- v4 corrected the v3 line-385 scheduler mistake: stdout line 385 step 9 is a
  recovery hold, not an advance, while target non-worsening is NO.
- v4 also eliminated v3 safety rollback: stdout line 422 reports
  `target_guarded_safety_rollbacks=0`, and audit line 26 passes safety rollbacks
  zero.
- The runtime still failed close_26. Runtime line 391 step 15 is the first hard
  safety freeze: `target_error_m=0.003035 > 0.003` and counter gap
  `0.002050m > 0.002`, while speed `0.001148m/s`, one-sided push NO, and support
  horizon YES were not the blockers.
- Final line 421 remains far from close_26: gripper `7.977deg`, command
  `8.000deg`, remaining close `18.023deg`, target error `0.003826m`, counter gap
  `0.002496m`, hard freeze YES.
- Runtime lines 422-423 report posthoc FAIL, 4 advances, 41 holds, zero
  zero-backlog holds, 41 backlog-preserved holds, zero safety rollbacks, 10 v4
  recovery holds, 31 hard safety freezes, `close_reached=NO`, attach/posewrite
  zero, telemetry-only YES, and success_claim NO.
- Audit line 16 fails `close_reached`, line 28 fails hard safety freezes zero,
  lines 50-52 fail hard freeze / fixed target / fixed support criteria, and line
  54 reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- Added static attribution script
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v4_recovery_runtime_static_analysis.py`
  md5 `e381cbbe65ff899c479e3aad3c399d4a`. It verifies the runtime/audit md5s and
  reports primary attribution
  `target_support_hard_gate_freeze_after_recovery_hold`.

Implication:

- v4 is not grasp success and not close_26 success.
- Do not rerun v4 as the next experiment; it already falsified recovery-hold-only
  scheduling under the fixed target/support gates.
- Do not relax fixed target/support gates. The first v4 hard freeze and the final
  plateau exceed both fixed gates while speed and one-sided push are acceptable.
- Keep the useful parts: zero-backlog holds must remain zero, safety rollbacks
  must remain zero, and attach/posewrite must remain zero.
- Next Track A work must be local/static/code-first: a structural target/support
  recovery or contact-compatible close mechanism that can recover target error
  and counter support before hard freeze, still under the unchanged close_26
  audit. Hold-lift, dataset generation, training, transport/release,
  constraints, and SurfaceGripper remain blocked until close_26 audit PASS.

Sources:

- `claudedocs/session_20260522_track_a_v4_recovery_runtime_fail.md`
- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v4_recovery_runtime_static_analysis.py`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out:37,39,41,385,390-392,421-423`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_audit_b200.out:1-54`

## D083 — The RL→expert→rollout→demo plan is valid only after a no-attach contact Stage 0 gate

Evidence:

- The user restated the intended pipeline: B200 Isaac Lab RL learns from random
  action + reward; the trained policy becomes the expert; expert rollouts record
  state/action/observation; rollouts are converted to LeRobot/RLDS demos.
- Added
  `sim_scripts/p7_branch_b_contact_rl_stage0_preflight_static_analysis.py`
  md5 `73fa3e8dc18fcc4a0e5a4cf702985eee`. It verifies v4 stdout md5
  `fe6a733727a6eeb288c6c6464c178af1` and audit md5
  `47f4ec7b78298fde0a46ac57105a6e6c`.
- The preflight reports direct B200 PPO now `NO`: latest no-attach v4 still has
  `close_reached=NO`, 31 hard freezes, final gripper `7.977deg`, and remaining
  close `18.023deg`; existing `roarm_rl/train_ppo.py` targets
  `RoArm-Pick-Direct-v0` / `RoArm-Stack-Direct-v0`, while both default envs use
  kinematic attach / `write_root_pose_to_sim`.
- Added
  `sim_scripts/p7_branch_b_cube2cm_contact_rl_v5_static_design.py`
  md5 `ab1b5c0b1b0655ebef4dc9c42d3e8de1`. It identifies v4 line 390 as the last
  safe pre-freeze step: target error `0.002891m`, target margin `0.000109m`,
  counter gap `0.001969m`, support margin `0.000031m`; v4 line 391 is already a
  fixed-gate breach.

Implication:

- The four-stage RL-to-expert-to-demo plan is not rejected; it is the correct
  high-level pipeline after Stage 0.
- Do not use existing attach-based Pick/Stack PPO envs as Track A no-attach
  contact expert evidence.
- Stage 0 must first create a no-attach contact RL env or v5 contact-close gate:
  robot joint target writes only, object attach/posewrite zero, fixed
  target/support gates unchanged, zero zero-backlog holds, zero safety rollbacks,
  and preemptive target/support recovery before a line-391-style hard freeze.
- Random-action sanity, PPO training, expert rollout, and demo dataset generation
  all still require separate approval after Stage 0 readiness.

Sources:

- `claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md`
- `sim_scripts/p7_branch_b_contact_rl_stage0_preflight_static_analysis.py`
- `sim_scripts/p7_branch_b_cube2cm_contact_rl_v5_static_design.py`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out:37,390-391,421-423`
  and audit `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_audit_b200.out:1-54`

## D084 — v5 recovery writes alone are insufficient; next advance must be projected against fixed target/support margins

Evidence:

- The approved B200 v5 close_26 runtime stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v5_preemptive_recovery_v7_close26_b200.out`
  has md5 `f93ddaa75920a560777f8f9c8fae26f0`; audit stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v5_preemptive_recovery_audit_b200.out`
  has md5 `7709c2bc37424bc7c3874e978b34d104`.
- Runtime line 393 had 8 v5 recovery writes and 0 IK failures, but support
  margin was already only `0.000335m`. Runtime line 394 then advanced toward
  the next close command with support margin `0.000243m`; runtime line 395
  immediately breached both fixed gates (`target_error_m=0.003008 > 0.003`,
  counter gap `0.002146m > 0.002`) and started hard freezes.
- Audit line 17 fails `close_reached`; line 30 fails hard-freezes-zero with
  value 32; lines 52-54 fail hard freeze / fixed target / fixed support; line
  59 reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- A static recomputation over B200 v5 lines 390-395 using the new v6 projection
  formula marks lines 390-392 advance-safe, but lines 393-394 unsafe because
  projected support margin is negative (`-0.000062m` and `-0.000003m`).
- Added default-off v6 projected-guard code and synced it to B200. Local/B200
  md5s match: runtime `e4d72390150a6660ce624d9ba1b4425d`, audit
  `d30c4583c2efd20a9449885e58a5dd80`, readiness
  `821f523cf99bec4eedfb11016d977aa1`. B200 readiness reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` for the v6 command on GPU0.

Implication:

- v5 is not grasp success and not close_26 success. It preserves useful
  constraints (attach/posewrite zero, zero zero-backlog holds, zero safety
  rollbacks, IK recovery writes), but it does not prevent a line-394-style
  unsafe advance.
- The next Track A runtime candidate is v6 projected guard, not PPO/training
  and not dataset generation. v6 is still only static/B200 readiness until
  separately approved and run once with immediate audit.
- Do not relax fixed target/support gates, tune gates, add constraints,
  SurfaceGripper, hold-lift, transport/release, dataset generation, or training
  as Track A evidence before close_26 audit PASS.

Sources:

- `claudedocs/session_20260522_track_a_v5_runtime_fail_v6_projected_guard_readiness.md`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v5_preemptive_recovery_v7_close26_b200.out:43,45,393-395,427-428`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v5_preemptive_recovery_audit_b200.out:17,28-30,52-57,59`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`

## D085 - v6 projection blocks unsafe advance, but projection alone is insufficient after recovery stalls

Evidence:

- The approved B200 v6 close_26 runtime stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out`
  has md5 `9a4f8825a88ee3c9d93d83e5b9a28b41`; audit stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out`
  has md5 `480a3355864937763eb665e086aadbb0`.
- Runtime lines 43 and 45 confirm strict v6 metadata:
  close_26-only, v6 flag YES, no training, constraints, SurfaceGripper,
  transport/release, gate tuning, posewrite, or success claim.
- Runtime lines 393-397 show v6 did the intended thing that v5 lacked: once
  projected support/target margins went unsafe, it blocked further advance and
  kept recovery active with IK OK.
- Runtime line 398 is the first hard freeze and first fixed-support failure:
  support gap `0.002075m > 0.002m` while target error `0.002914m` was still
  within the fixed `0.003m` target gate.
- Runtime line 399 then breached both fixed gates: target error `0.003052m`
  and support gap `0.002146m`.
- Runtime lines 427-428 report 4 advances, 41 holds, zero zero-backlog holds,
  zero safety rollbacks, 12 recovery writes, 0 IK failures, 29 hard freezes,
  `close_reached=NO`, attach/posewrite zero, telemetry-only YES, and
  success_claim NO.
- Audit line 18 fails `close_reached`; line 31 fails hard-freezes-zero with
  value 29; lines 51-53 fail hard freeze / fixed target / fixed support; line
  58 reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

Implication:

- v6 is not grasp success and not close_26 success. Runtime exit 0 is irrelevant
  unless the posthoc audit reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=YES`.
- The projected guard fixed the v5 line-394 unsafe-advance class, but the
  recovery action after projection blocked advance was too weak/passive to
  restore target/support margins before fixed-gate failure.
- Do not rerun v6 unchanged. Do not relax fixed target/support gates, tune gates,
  add constraints, SurfaceGripper, hold-lift, transport/release, dataset
  generation, PPO/training, or rollout as Track A evidence before close_26 audit
  PASS.
- The next Track A work should be local/static/code-first active target/support
  recovery after a projected block, while preserving no attach/posewrite, zero
  zero-backlog holds, zero safety rollbacks, and fixed gates.

Sources:

- `claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out:43,45,393-399,427-428`
- B200
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out:18,27-31,51-58`


## D086 - OpenVLA-OFT 7B local inference deps + inline L1 action head bypass dlimp chain

Evidence:

- 2026-05-22 Track B P4 attempted to load OpenVLA-OFT ckpt 7500 on Lenovo 4090 stack
  (Ubuntu 22.04, roarm conda env, torch 2.7.1+cu126, transformers 4.57.6, lerobot 0.4.4).
- `prismatic` editable install from `/home/cgxr/Documents/Robotics/openvla-oft/`
  fails to import without `peft`, `rich`, `timm 0.9.16`, and **`dlimp`** (a private
  GitHub package, not on PyPI, used only by RLDS data loading path).
- Import chain that pulls `dlimp` into inference: `prismatic.models.action_heads`
  → triggers `prismatic.models.__init__` → `from .load import ...` → load.py
  imports `prismatic.models.vlas` → `vlas/__init__` → `from .openvla import OpenVLA`
  → openvla.py imports `prismatic.vla.action_tokenizer` → triggers
  `prismatic.vla.__init__` → `from .materialize import get_vla_dataset_and_collator`
  → materialize.py imports `prismatic.vla.datasets` → `datasets.py` line 23
  imports `prismatic.vla.datasets.rlds` → rlds/dataset.py line 13: `import dlimp as dl`
  → `ModuleNotFoundError`.
- `prismatic.vla.constants` and `prismatic.extern.hf.modeling_prismatic` import
  independently without going through this chain (verified empirically).
- `L1RegressionActionHead` definition in `prismatic.models.action_heads.py:84-107`
  is a simple `MLPResNet`-based MLP with `num_blocks=2, input_dim=hidden_dim*ACTION_DIM,
  output_dim=action_dim`, no diffusion/RLDS/dlimp dependency. The B200-trained
  checkpoint `action_head--7500_checkpoint.pt` has 16 tensors (all `module.`-prefixed
  from DDP) matching this exact shape (`input_dim=4096, hidden_dim=4096, action_dim=6`).
- Inline copy of `_MLPResNetBlock`, `_MLPResNet`, `L1RegressionActionHead` in
  `deploy_openvla_oft.py:78-138` loads the trained ckpt with `strict=True`,
  zero missing / zero unexpected keys, 134,328,326 params, identical forward shape
  `(1, NUM_ACTIONS_CHUNK*ACTION_DIM=48, 4096) → (1, 8, 6)`.

Implication:

- For any OpenVLA-OFT inference-only deployment that doesn't need RLDS data
  loading, install only: `peft==0.18.0`, `rich`, `timm==0.9.16` (HARD RULE #15
  pin), plus editable prismatic — do NOT install `dlimp` and do NOT import
  through `prismatic.models.action_heads`.
- Instead inline-copy the relevant action head class. The `prismatic.vla.constants`
  + `prismatic.extern.hf.modeling_prismatic.OpenVLAForActionPrediction` imports
  remain safe (they bypass the `vla.__init__` materialize chain via direct module
  path).
- This pattern is what `openvla_oft_roarm/eval_offline_v6.py` should also follow if
  it were rewritten to support a smaller inference env (currently it relies on B200
  full prismatic install).
- This pattern is required for any future local inference scripts (e.g., closed-
  loop deploy, ablation runs, head-to-head SmolVLA comparison).

Sources:

- `deploy_openvla_oft.py:78-138` (inline classes)
- `deploy_openvla_oft.py:166-179` (apply_sdpa_class_attr_patch)
- `deploy_openvla_oft.py:228-292` (load_openvla_oft norm_stats inject + action head load)
- `claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md` Sanity 1
- `/home/cgxr/Documents/Robotics/openvla-oft/prismatic/models/action_heads.py:84-107`
- `/home/cgxr/Documents/Robotics/openvla-oft/prismatic/vla/__init__.py:1`
- `/home/cgxr/Documents/Robotics/openvla-oft/prismatic/vla/datasets/rlds/dataset.py:13`
- HARD RULE #15 (timm 0.9.16 pin; nightly cu128 is for B200, NOT Lenovo 4090 sm_89)


## D087 - B200 lease retirement means future research must be local/RunPod from verified backups

Evidence:

- The B200 lease ends on 2026-05-22 at 23:59 KST. After that point, new work
  must not depend on SSH access to `JHPark/roarm_b200` or B200 `/tmp`.
- Track A backup was reverified from local and B200 sources before retirement:
  `/tmp/p7_branch_b_*` has 494 files on both sides, path+size hash
  `c308d1a682560cf51136cdd1a018c50ce2e7b488f1a0d4620e31abf7de80cfd4`,
  and file-content aggregate hash
  `cca0586b77c36ee79532d0640f9a35b2f1056654ab2758f256ea2bc1f149a4ae`.
- Track A B200 `sim_scripts` snapshot was also reverified: 53 non-pycache files,
  path+size hash `98563bbc3d27426351abd13272a88537009372b2c709b46d2a5021560c5ea23a`,
  file-content aggregate hash
  `fefe4c873c1e45ec4cb95226a2c1a0d53860e4eca926c93d3da1b9887c9ca83f`.
- Track B outputs were verified locally across three locations:
  `b200_backup_20260522_final/outputs`, `b200_backup_20260521`, and
  `openvla_oft_b200_pulls`. OpenVLA full has zero missing remote files locally
  (`comm -23 remote local = 0`); local has only three extras (`_pull.log` and two
  eval JSON files).
- B200 env specs and wandb cache were also preserved:
  `env_specs` manifest hash
  `5e357fb4ebd4efc1a9b2918af30ecbec39128c8a54d93029557dd1f1fdb01151`;
  `wandb_cache` 35 files / 5.7M, manifest hash
  `d68c65cb1f08ed76a02634952e62b1d4c24b3300f39ec3c7dee13649db8ce871`.

Implication:

- Do not plan future work around entering B200, rerunning B200 Isaac, or pulling
  additional B200 files. Treat B200 as retired/unavailable after 2026-05-22 23:59
  KST.
- Do not copy, request, or depend on `.ssh` private material as "research data".
  Research continuity comes from backed-up artifacts, not login secrets.
- Future compute should be local 4090 and/or RunPod. Start each RunPod/local
  continuation by rebuilding/verifying env from backups, then run a small smoke
  test before full training/eval.
- The local output layout is split: complete OpenVLA full checkpoints are in
  `openvla_oft_b200_pulls`, while some older SmolVLA outputs live under
  `b200_backup_20260521`. Do not assume every complete Track B artifact is under
  `b200_backup_20260522_final/outputs`.

Sources:

- `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
- `b200_backup_20260522_final/README_BACKUP.md`


## D088 - B200 disconnect is now confirmed; do not attempt recovery-by-SSH in future sessions

Evidence:

- On 2026-05-23 KST, after the 2026-05-22 23:59 KST lease expiry, the user
  reported that B200 now shows `disconnect`.
- D087 already verified the required Track A/B research artifacts locally before
  retirement:
  - Track A `/tmp/p7_branch_b_*`: 494 files, path+size hash
    `c308d1a682560cf51136cdd1a018c50ce2e7b488f1a0d4620e31abf7de80cfd4`,
    content aggregate
    `cca0586b77c36ee79532d0640f9a35b2f1056654ab2758f256ea2bc1f149a4ae`.
  - Track A B200 `sim_scripts`: 53 files, path+size hash
    `98563bbc3d27426351abd13272a88537009372b2c709b46d2a5021560c5ea23a`,
    content aggregate
    `fefe4c873c1e45ec4cb95226a2c1a0d53860e4eca926c93d3da1b9887c9ca83f`.
  - Track B outputs: preserved across `b200_backup_20260522_final/outputs`,
    `b200_backup_20260521`, and `openvla_oft_b200_pulls`.

Implication:

- Future sessions must not try to "just check B200", "pull one more file", or
  recover by SSH unless the user explicitly provides a new, valid compute
  allocation. Treat B200 as unavailable.
- The correct next step is local/RunPod continuation from verified backups.
- If a path depends on a B200-only file that is not in the backups, treat that
  path as blocked and redesign from local evidence rather than attempting B200
  access.

Sources:

- User report in chat on 2026-05-23 KST: B200 ended and shows disconnect.
- `START_HERE.md` B200 Retired / Backup Truth section.
- `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`


## D089 - Track A large datasets and learning must be gated by no-attach Stage 0

Evidence:

- v6 projected guard is still FAIL, not grasp success. Local backup runtime stdout
  md5 is `9a4f8825a88ee3c9d93d83e5b9a28b41`; audit stdout md5 is
  `480a3355864937763eb665e086aadbb0`.
- v6 runtime lines 393-397 show projected unsafe advance was blocked and recovery
  writes continued with IK OK, but runtime line 398 is the first support-gate hard
  freeze (`counter gap 0.002075m > 0.002m`) and line 399 breaches both fixed
  target/support gates.
- v6 audit line 18 fails close_reached, line 31 fails hard-freezes-zero with value
  29, lines 51-53 fail hard freeze / fixed target / fixed support, and line 58
  reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- The professor/user pipeline - RL learns, trained policy becomes expert, expert
  rollouts become demos, demos become LeRobot/RLDS-style datasets - is valid in
  principle, but Stage 0 preflight showed existing `RoArm-Pick-Direct-v0` and
  `RoArm-Stack-Direct-v0` are attach/posewrite envs and cannot produce Track A
  no-attach expert evidence.
- Added
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  md5 `14a462526945f3c5bca1c5e8c3e13525`. It reverified local v6 md5s and showed
  that pre-freeze recovery rows increased target error by `0.000626m` and support
  gap by `0.000319m`; v6 had recovery writes, but not active target/support
  recovery.

Implication:

- Do not start PPO/training, rollout collection, dataset generation, or large-scale
  learning from current Track A evidence.
- The required order is:
  1. local/static active target/support recovery design after v6 projected block;
  2. default-off runtime candidate plus audit/readiness support;
  3. close_26-only runtime plus immediate posthoc audit, only after explicit
     approval;
  4. hold-lift gate after close_26 PASS;
  5. no-attach RL env / random sanity / small PPO smoke;
  6. expert rollout to a small pilot dataset;
  7. replay/audit PASS;
  8. large dataset scaling;
  9. BC/VLA/IL learning.
- A dataset is not "proper" just because it is large. For Track A it must be
  generated from a no-attach, no-posewrite contact primitive that has passed
  close_26, hold-lift, and pilot replay gates.

Sources:

- `claudedocs/session_20260526_track_a_stage0_to_dataset_step_plan.md`
- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
- `claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md`
- `claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md`
- B200 local backup
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out:43,45,393-399,427-428`
- B200 local backup
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out:18,27-31,51-58`


## D090 - Claude and Codex MCP configurations are separate; verify loaded RunPod tools before using them

Evidence:

- The user reported that Claude could use RunPod MCP while this Codex session
  could not. Local verification showed this was a configuration/loading
  mismatch, not proof that Codex cannot use MCP.
- Claude config `/home/cgxr/.claude.json` has an MCP server named `runpod` with
  command `npx`, args `["-y", "@runpod/mcp-server@latest"]`, and env key
  `RUNPOD_API_KEY`. The key value was not printed.
- Before the update, Codex config `/home/cgxr/.codex/config.toml` did not have
  RunPod; it listed other MCP servers such as context7, filesystem,
  sequential-thinking, memory, fetch, github, and arxiv.
- Codex config was updated with `[mcp_servers.runpod]`, using the same RunPod MCP
  package and a copied `RUNPOD_API_KEY` value from Claude config without printing
  the secret. A backup was created at
  `/home/cgxr/.codex/config.toml.bak_runpod_20260526` md5
  `1ef4acf6f1c92a64b9bbd79a2e35b7e7`.
- Redacted post-edit verification showed `/home/cgxr/.codex/config.toml:71`
  contains `[mcp_servers.runpod]`, line 73 contains
  `@runpod/mcp-server@latest`, and line 74 contains `RUNPOD_API_KEY`.
- Same-session `tool_search` after the config edit still did not expose
  `mcp__runpod__...`, so the current session likely needs restart/new-session
  tool loading before RunPod MCP becomes callable.

Implication:

- Do not say "Codex cannot use RunPod MCP" unless current Codex docs/tools prove
  it; the verified issue here was missing Codex MCP registration in this session.
- Do not assume RunPod MCP is usable merely because `/home/cgxr/.codex/config.toml`
  has a RunPod block. Each new session must verify that `mcp__runpod__...` tools
  are actually loaded before trying RunPod actions.
- If RunPod MCP is not loaded after config registration, restart/new-session/tool
  reload is the next operational step, not B200 SSH and not stale pod reuse.
- Do not use the old RunPod pod `az53n8t8alp8pz` from 2026-05-06 unless the user
  explicitly confirms it is current and active.
- Continue to obey D087-D088: B200 is unavailable; no `ssh JHPark`, no B200 pull,
  and no `.ssh` copying.

Sources:

- `claudedocs/session_20260526_runpod_mcp_codex_registration_and_next_prompt.md`
- `/home/cgxr/.codex/config.toml:71-75`
- `/home/cgxr/.claude.json` RunPod MCP server keys, redacted verification only


## D091 - Local CUDA can be healthy while Codex default sandbox hides GPU device nodes

Evidence:

- The local CUDA block was first caused by an NVIDIA driver/userspace mismatch:
  loaded kernel module `580.126.09` vs userspace/NVML `580.159.03`, after the
  2026-05-21 06:04 KST apt upgrade and before reboot.
- After the user rebooted the local PC on 2026-05-26, the host NVIDIA stack
  matched at `580.159.03`: `/proc/driver/nvidia/version`,
  `/sys/module/nvidia/version`, and `libnvidia-ml.so.1`.
- In the default Codex sandbox, `nvidia-smi` still failed and `/dev/nvidiactl`,
  `/dev/nvidia0`, and `/dev/nvidia-uvm` were not visible.
- The same `nvidia-smi` run with `sandbox_permissions=require_escalated` succeeded
  and reported Driver `580.159.03`, CUDA `13.0`, and RTX 4090 Laptop GPU.
- `conda run -n isaaclab` run with `sandbox_permissions=require_escalated`
  reported `torch_cuda_available True`, `device_count 1`, and IsaacLab/roarm_rl
  imports OK.
- v7 static readiness after reboot still reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

Implication:

- Do not confuse a Codex default sandbox GPU-device visibility failure with a
  local host CUDA failure.
- GPU/IsaacLab commands that require `/dev/nvidia*` should be run with
  `sandbox_permissions=require_escalated` in Codex, while still obeying all Track
  A runtime gates.
- The next Track A v7 close_26 runtime can be run locally, but it is still a
  physics gate, not a success claim. It must be followed immediately by the v7
  posthoc audit.
- If the audit fails, analyze the first failing runtime/audit lines before any
  rerun. If it passes, do not start dataset/training; next gate is hold-lift.

Sources:

- `claudedocs/session_20260526_track_a_cuda_reboot_codex_sandbox_ready.md`
- `claudedocs/session_20260526_track_a_v7_local_runtime_cuda_blocked.md`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`


## D092 - v7 active recovery telemetry is real, but the mechanism still fails close_26

Evidence:

- After local reboot/CUDA recovery, one escalated local close_26-only v7 active
  recovery runtime was run and immediately audited. Runtime stdout md5
  `621d00b9d157b4e70178c28f94ca4c7f`; audit stdout md5
  `406b96557d94418f16273e517ec4d69b`.
- Runtime line 6 confirms the mechanism
  `target_guarded_micro_close_v7_active_recovery_diagnostic` with separate
  approval marker YES.
- Runtime line 8 confirms the v7 contract: finite-difference TCP sweep, current
  object pose, object posewrite NO, robot joint target writes only, constraints
  NO, and SurfaceGripper NO.
- Runtime lines 389-391 show v7 active recovery did trigger: 3 active recovery
  writes, 0 IK failures, and counter-gap deltas `-0.000684m`, `-0.000703m`, and
  `-0.000716m`.
- Runtime line 392 is the first hard freeze and first fixed-support failure:
  target error `0.002962m` was still inside the fixed 3mm gate, but counter gap
  was `0.002048m > 0.002m`. Runtime line 393 then breached both fixed gates:
  target error `0.003059m` and counter gap `0.002104m`.
- Runtime line 424 aggregate reports close_reached NO, 4 advances, 41 holds, 31
  hard freezes, 3 v7 active recovery writes, 0 v7 IK failures, attach/posewrite
  zero, telemetry-only YES, and success_claim NO.
- Audit line 19 fails close_reached; line 32 fails hard-freezes-zero with value
  31; lines 54-56 fail hard freeze / fixed target / fixed support; line 66
  reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- Audit lines 60-64 pass the v7-specific checks: active recovery present,
  trigger seen, IK OK, counter-gap reduction, and valid selected margins.

Implication:

- v7 is not grasp success and not close_26 success. It is a real physics/audit
  failure, not an infrastructure block.
- The v7 telemetry/audit path is validated enough to be diagnostic, but the
  mechanism is insufficient: selected active recovery candidates can look valid
  at the candidate level while the subsequent observed close rows still lose
  support/target margin.
- Do not rerun v7 unchanged. Do not proceed to hold-lift, PPO/training, rollout,
  dataset generation, constraints, SurfaceGripper, transport/release, or gate
  tuning from this result.
- The next Track A work should be static failure analysis/redesign before any
  new runtime approval. The redesign must preserve fixed target/support gates,
  no attach/posewrite, zero zero-backlog holds, zero safety rollbacks, and
  robot-joint-target-only writes.

Sources:

- `claudedocs/session_20260526_track_a_v7_active_recovery_runtime_fail.md`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out:6,8,389-393,423-424`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/audit.out:19,32,54-56,60-66`


## D093 - v7 failed because candidate recovery did not become observed recovery

Evidence:

- Added static analyzer
  `sim_scripts/p7_branch_b_cube2cm_v7_failure_analyzer.py` md5
  `e13605f058cd1908ff3d863e8239fbc4`; `py_compile` PASS.
- Analyzer output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v7_failure_static_analysis.out`
  md5 `0fbf57f32473fa253ee1082b888bdcb1`, final line 23
  `ANALYZER_RESULT=PASS`.
- Analyzer line 2 parsed 45 close rows from runtime lines 378-422. Analyzer
  line 3 found 3 v7 active rows.
- Analyzer line 5 identifies first v7 active recovery at runtime line 389 /
  step 12, with target margin `+0.000599m` and support margin `+0.000312m`.
- Analyzer line 4 identifies first low-margin row at runtime line 391 / step 14,
  with target margin `+0.000173m` and support margin only `+0.000037m`.
- Analyzer lines 7 and 9 identify runtime line 392 / step 15 as both first
  support breach and first hard freeze: target `0.002962m` is still inside the
  fixed 3mm gate, but counter gap `0.002048m > 0.002m`.
- Analyzer line 8 identifies runtime line 393 / step 16 as first fixed target
  breach: target `0.003059m`, counter gap `0.002104m`.
- Analyzer lines 11-13 compare selected v7 candidates against observed next
  rows. All 3 active followups predicted `-0.001500m` target-error improvement
  plus negative counter-gap deltas, but the observed next rows worsened target
  error and counter gap. TCP follow ratios were negative: `-0.164`, `-0.117`,
  and `-0.089`.
- Analyzer lines 15-20 classify the domains: audit contract mismatch NO,
  trigger timing late YES, candidate prediction mismatch YES, weak TCP follow
  YES, contact geometry suspect YES, and hard-safety lockout after active YES.

Implication:

- v7 should not be rerun unchanged. The failure is now narrowed: selected
  candidate-level margins and counter-gap deltas do not imply observed runtime
  recovery.
- The next valid Track A work is static v8 design only. It must account for
  earlier trigger timing, multi-step observed dynamics, actual TCP follow, and
  counter-contact geometry while preserving fixed target/support gates, no
  attach/posewrite, zero zero-backlog holds, zero safety rollbacks, and
  robot-joint-target-only writes.
- Do not proceed to hold-lift, PPO/training, rollout, dataset generation,
  constraints, SurfaceGripper, transport/release, or gate tuning from v7 or this
  static analysis.

Sources:

- `claudedocs/session_20260526_track_a_v7_failure_static_analysis.md`
- `sim_scripts/p7_branch_b_cube2cm_v7_failure_analyzer.py`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v7_failure_static_analysis.out:1-23`


## D094 - v8 must be observed-response driven, not candidate-margin driven

Evidence:

- Added static design script
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py`
  md5 `56a382377b7fb0f0c6391bf59163af0d`; `py_compile` PASS.
- Saved output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_observed_recovery_static_design.out`
  md5 `c14e80ec5fc69c6e6e17925d61f81d0b`.
- Output lines 1-2 verify the preserved post-reboot v7 runtime md5
  `621d00b9d157b4e70178c28f94ca4c7f` and v7 static analysis md5
  `0fbf57f32473fa253ee1082b888bdcb1`.
- Output line 3 finds the first projected reserve trigger at runtime line 386 /
  close step 9, before v7 first active recovery.
- Output line 4 shows v7 first active recovery at runtime line 389 / step 12.
- Output lines 5-7 show the first support breach/hard freeze at runtime line 392
  / step 15, and cross-check that the projected reserve trigger is 3 steps before
  v7 first active and 6 steps before first support breach.
- Output lines 8-11 reject unchanged v7 by observed response: every active
  followup worsened target and support gap, with TCP follow ratios `-0.164`,
  `-0.117`, and `-0.089`.
- Output lines 12-20 confirm the static v8 checks: earlier projected reserve
  trigger, reserve horizon before support breach, unchanged-v7 rejection by
  observed response/TCP follow, counter-contact geometry requirement, fixed gates
  preserved, and forbidden mechanisms forbidden.
- Output lines 21-26 define the v8 design contract; line 27 reports
  `RUNTIME_READY=NO`; line 28 reports `STATIC_V8_DESIGN_DONE=YES`.

Implication:

- v8 must trigger from projected reserve depletion before the v7 late-active
  window, and it must evaluate multi-step observed response after selected
  recovery actions.
- Candidate-level selected margins, negative counter-gap deltas, and IK success
  are necessary diagnostics but not success evidence.
- The next valid Track A step is default-off v8 runtime-candidate implementation
  plus matching audit/readiness static checks, not a runtime, not hold-lift, and
  not dataset/training.
- Any v8 runtime candidate must preserve fixed target/support gates, no
  attach/posewrite, no constraints, no SurfaceGripper, no fixed-gate tuning, zero
  zero-backlog holds, zero safety rollbacks, and robot-joint-target-only writes.

Sources:

- `claudedocs/session_20260526_track_a_v8_observed_recovery_static_design.md`
- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_observed_recovery_static_design.out:1-28`


## D095 - v8 is static-ready for separate runtime approval, not physics-validated

Evidence:

- Implemented default-off v8 runtime candidate and matching posthoc audit/readiness
  support. Runtime probe md5 `7e6dfc35bbfeacb5d1689f2f175e5120`; audit md5
  `8dbf621c983ec03f46e5d52843781fda`; readiness md5
  `a31ced20b754a4a42058349525d1a435`.
- The future v8 command uses the preserved local backup USD
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd`
  md5 `4497024d25abab11de5c50e144124553`, not volatile `/tmp`.
- Readiness output
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_runtime_candidate_readiness.out`
  md5 `6a2a62808451175b65e5d522b695b8b6`. Lines 1-2 confirm local/static only,
  no Isaac runtime, no training/dataset, no constraints, no SurfaceGripper, no
  attach/object posewrite, no transport/release, and no gate tuning.
- Readiness lines 3-4 confirm runtime wiring and audit metadata guard.
- Readiness lines 5-13 confirm negative controls reject archived v6-as-v8, v7
  reference, no damping, v3 zero-backlog, v4 hard-freeze, v7 no-active-recovery,
  v8 worsening response, v8 no TCP follow, and v8 no counter contact.
- Readiness line 14 accepts synthetic v8 PASS, line 16 prints the future local
  backup USD command, and line 19 reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- Additional v8 audit of the preserved post-reboot v7 runtime
  `v8_rejects_post_reboot_v7_audit.out` md5
  `cb082918d92a0f95b585ade432c34730` rejects unchanged v7 as expected: lines
  5/14/15 fail v8 metadata; lines 20/30 fail close/hard-freeze success; lines
  53/55/57/58 fail v8 reserve-trigger, observed-response, TCP-follow, and
  counter-contact modeling checks; line 60 reports PASS=NO.

Implication:

- v8 is ready only for a separately approved close_26 runtime attempt. It is not
  close_26 success, not hold-lift readiness, and not dataset/training readiness.
- If the v8 runtime is approved, run exactly one close_26-only local runtime with
  escalated Codex GPU/Isaac execution, capture stdout/stderr under
  `claudedocs/runtime_logs/`, and immediately audit with expected mechanism
  `target_guarded_micro_close_v8_observed_recovery_diagnostic`.
- If v8 audit fails, stop and analyze the first failing runtime/audit lines before
  any rerun. If it passes, the next gate is hold-lift, not dataset/training.

Sources:

- `claudedocs/session_20260526_track_a_v8_runtime_candidate_static_readiness.md`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_runtime_candidate_readiness.out:1-19`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_rejects_post_reboot_v7_audit.out:1-60`


## D096 - first v8 runtime failed before recovery; virtual damping inheritance is mandatory

Evidence:

- User approved exactly one local close_26-only v8 observed-recovery runtime after
  local CUDA was healthy. It ran under escalated Codex GPU/Isaac execution and was
  immediately audited. Runtime stdout
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/runtime.out`
  md5 `74095570c2d6a60abdf522c2413735db`; audit stdout
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/audit.out`
  md5 `7cd38eddb1dc9c925b01948cbc5cb416`.
- Runtime lines 4/6/8 confirm the intended v8 metadata, close_26-only scope, no
  attach/object posewrite, no constraints, no SurfaceGripper, and
  robot-joint-target-only recovery writes.
- Audit line 20 fails `close_reached`; line 26 fails positive virtual damping
  writes with value `0`; lines 35-36 show step3 virtual damping inactive and no
  write seen; lines 45-49/53 show no v5/v7 recovery present/triggered and no v8
  projected-reserve trigger; line 60 reports
  `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- Runtime lines 423-424 confirm the aggregate failure: `close_reached=NO`,
  `virtual_velocity_damping_writes=0`, 4 close advances, 41 holds, 39 hard safety
  freezes, attach/posewrite zero, telemetry-only YES, and success claim NO.
- Static failure analysis
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/v8_runtime_failure_static_analysis.out`
  md5 `7e81773b91d39658a3ec5c6eaf878f0c` reports first hard freeze at runtime
  line 384 / close step 7 with virtual damping inactive and no v8 trigger; first
  target/support margin breach at runtime line 392 / close step 15; and
  `seen_trigger=NO seen_needed=NO seen_recovery=NO`.
- Code inspection found the wiring split: post-fix runtime lines 1174-1184 include
  v8 in `target_guarded_close_active`, and lines 1255-1265 now include v8 in
  `virtual_damping_active`. Lines 1426-1434 show the v8 reserve trigger is still
  gated by hard safety OK, so missing damping can block the recovery window before
  the trigger is ever seen.
- Post-fail static fix md5s: runtime probe
  `acae0ca2e85a522dd4ac8fb583cb8fb8`, audit unchanged
  `8dbf621c983ec03f46e5d52843781fda`, readiness
  `dc2bdaa8d882f12b5cc901a677caccc0`. Post-fix readiness output
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/readiness_after_v8_damping_fix.out`
  md5 `b652520a81792bf12373ff742cdba6b5`: line 5 confirms the new
  `runtime_probe_v8_inherits_virtual_damping_active` check, and line 20 reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

Implication:

- The first v8 runtime is a useful negative physics result, not success. Runtime
  exit 0 and v8 metadata are insufficient.
- Pre-fix v8 must not be rerun unchanged. Any post-fail v8 runtime must use the
  fixed code where v8 inherits virtual damping and must be separately approved.
- The next valid Track A action is exactly one post-fix close_26-only v8 runtime
  plus immediate v8 audit, or further static review. Do not start hold-lift,
  dataset generation, PPO/training, rollout, constraints, SurfaceGripper, gate
  tuning, or transport/release from this result.

Sources:

- `claudedocs/session_20260526_track_a_v8_runtime_fail_and_damping_wiring_fix.md`
- `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/runtime.out:4,6,8,423-424`
- `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/audit.out:20,26,35-36,45-49,53,60`
- `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/v8_runtime_failure_static_analysis.out:1-12`
- `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/readiness_after_v8_damping_fix.out:1-20`
- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`


## D097 - cube push/tap rollout probe is separate from Track A grasp and from training

Evidence:

- Added `sim_scripts/cube3cm_push_rollout_probe.py` md5
  `8d329b79106e7ca2c03fa91b7ac87170` for the professor's endpoint-known 3cm
  cube push/tap question.
- The 20,480-trial local IsaacLab run is preserved under
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/`.
  Runtime stdout md5 `2aad344f08f95c880e43bc0d7f655998`, stderr md5
  `30b990c1766da0c11a257fc0bec68526`, summary md5
  `5c9278450b5531afb7b0ca2a1fed46ee`, per-env CSV md5
  `4c2864301bea8e2ae798a8f77adf23ab`, audit md5
  `3e0096ba54e7cc0ec0e55b1b26a50b8e`.
- Runtime stdout line 20 explicitly marks the run as local Isaac, 3cm cube,
  `grasp=NO`, `attach_posewrite=NO`, `rollout_object_posewrite=NO`,
  `training=NO`, and `dataset_generation=NO`.
- Runtime stdout line 21 defines the robot action semantics as normalized 6D
  joint-delta actions:
  `robot_dof_targets += action_scale(0.100) * action`, clip `[-1,1]`, gripper
  target open 0 rad.
- Runtime stdout line 42 reports `total_trials=20480`, `ik_ok_rate=1.0000`,
  `disp_xy_mean_m=0.031809`, `disp_xy_p95_m=0.089702`,
  `moved_5mm_rate=0.8774`, `push_positive_1mm_rate=0.9086`,
  `action_abs_mean=0.086382`, zero action saturation, zero grasp marker,
  zero attach calls, and zero rollout posewrite calls.
- `rollout_stats_audit.out` lines 1-4 cross-check row count, rates, mechanism
  separation, and action-scale conversion. Lines 5-11 show outlier risk:
  displacement max `0.521036748m`, cube speed max `4.549609073m/s`, and tip
  angle max `179.981780282deg`.

Implication:

- This run answers the professor's immediate "go near the 3cm cube and hit/push
  it many times" question as scripted physics rollout statistics.
- It must not be cited as Track A close_26 grasp success, hold-lift readiness,
  dataset readiness, PPO success, or VLA success.
- If a learned result is required, the next step is a separate no-attach
  cube-push RL task/env with explicit rewards and outlier filtering, not reuse of
  existing attach-based Pick/Stack PPO evidence.

Sources:

- `claudedocs/session_20260526_cube3cm_push_rollout_probe_professor_request.md`
- `sim_scripts/cube3cm_push_rollout_probe.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runtime.out:20-42`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/per_env.csv`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/rollout_stats_audit.out:1-21`


## D098 - cube-push PPO must pass frozen 1k clean-impact audit before 10k/100k scaling

Evidence:

- A separate no-attach cube-push RL env was added for the professor branch, not for
  Track A grasp success. Current `roarm_rl/roarm_cube_push_env.py` md5
  `b44996c396c099847e5196949ed86742` keeps a 3cm/20g cube, forces no attach,
  and defines IK endpoint reset plus speed-guard reward terms.
- The IK curriculum 50-iteration run completed with exit code 0 and model_49,
  but `ppo_ik_curriculum_50iter_audit.out` lines 4-8 show direction/impact
  regression: final push-aligned displacement `-0.105634235m`, XY displacement
  `0.604205787m`, target distance `0.610159278m`, and impact rate
  `0.351481140`. Line 18 verdict:
  `IK_CURRICULUM_RAN_IMPACT_HEAVY_REGRESSED_DIRECTION_NO_SUCCESS_CLAIM`.
- The clean-reward 50-iteration run improved training-log safety:
  `ppo_clean_reward_50iter_audit.out` line 18 shows prior/new XY displacement
  `0.604205787 -> 0.025637908`, target distance `0.610159278 -> 0.040735956`,
  and impact `0.351481140 -> 0.003580729`. But its frozen 1k eval still had
  high impact: `ppo_clean_reward_model49_eval1024_audit.out` lines 3-6 report
  controlled `0.631610942`, impact `0.283282675`, clean success marker
  `0.326443769`, and speed p95 `8.694638634m/s`. Line 18 verdict:
  `CLEAN_MODEL49_EVAL_SAFER_BUT_IMPACT_TOO_HIGH_NO_10K`.
- The speed-guard v3 changed action scale to `0.05` and added speed penalty/gate.
  `ppo_speed_guard_50iter_stdout.out` line 5 prints action scale `0.050`; line 7
  prints speed penalty and success speed max. Training-log audit line 22 says
  `SPEED_GUARD_SIGNAL_PRESENT_NEEDS_MODEL49_EVAL`.
- The speed-guard frozen 1k eval did not improve impact. Summary lines 2-20 and
  `ppo_speed_guard_model49_eval1024_audit.out` lines 3-6 show action scale
  `0.05`, controlled `0.619682540`, impact `0.286984127`, clean success marker
  `0.323809524`, and p95 speed `5.578794289m/s`; line 17 verdict:
  `SPEED_MODEL49_EVAL_NO_IMPACT_IMPROVEMENT_NO_10K`.

Implication:

- Do not scale cube-push learned-policy evaluation to 10k/100k trials just because
  the training loop ran or TensorBoard reward increased.
- The minimum scaling gate for this professor branch is: frozen-policy 1k eval
  exit 0, no attach/posewrite, grasp marker 0, impact rate below about 5%,
  controlled push above about 60%, and clean success marker above about 30%.
- Current best learned-policy result is informative but not a success claim:
  IK pre-contact works, no-attach PPO runs, clean reward reduces far-fling in
  training logs, but frozen eval still has too many high-speed impact cases.
- Next valid research step is not 10k/100k. It is a better speed/contact
  curriculum or action smoothing/velocity-limited controller, then another
  50-100 iteration run and frozen 1k audit.

Sources:

- `claudedocs/session_20260526_cube3cm_push_rl_reward_curriculum.md`
- `roarm_rl/roarm_cube_push_env.py:42-113,299-391`
- `roarm_rl/train_cube_push_ppo.py`
- `roarm_rl/eval_cube_push_policy.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_ik_curriculum_50iter_audit.out:1-18`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_ik_curriculum_model49_eval1024_audit.out:1-18`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_clean_reward_50iter_audit.out:1-21`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_clean_reward_model49_eval1024_audit.out:1-18`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_speed_guard_50iter_audit.out:1-22`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_speed_guard_model49_eval1024_audit.out:1-17`

## D099 - contact-speed PPO improved impact but still failed the 1k scale gate

Evidence:

- V4 action smoothing/velocity limit improved frozen-eval impact versus the
  speed-guard run but still left impact `0.245576787`; audit verdict remained
  `SMOOTH_MODEL49_EVAL_IMPROVED_BUT_NO_10K`.
- V5 scripted-teacher warm-start improved training logs, but teacher-off frozen
  eval did not transfer: impact `0.257686676`, clean success `0.095168375`, and
  verdict `TEACHER_MODEL49_EVAL_TEACHER_OFF_NO_TRANSFER_NO_10K`.
- V6 policy-only contact-speed curriculum used action scale `0.025`, joint delta
  cap `0.004`, contact scale `0.15`, fast-cube scale `0.05`, lead cap `0.030`,
  precontact clearance `0.020`, and speed threshold `0.200`. Training audit
  showed speed/impact improvement (`impact=0.000813802`, speed-over-0.5
  `0.011067709`) but high low-motion `0.377115905`.
- V6 frozen 1k eval improved impact to `0.153782895`, but failed the 5% gate and
  clean success stayed only `0.110197368`; verdict
  `CONTACT_SPEED_MODEL49_EVAL_IMPROVED_BUT_NO_10K`.
- A teacher-on scripted diagnostic was not a rescue: impact `0.162448980`, clean
  success `0.067755102`, low motion `0.341224490`, verdict
  `TEACHER_ON_DIAGNOSTIC_UNSAFE_OR_WEAK_NOT_LEARNED_NO_10K`.

Implication:

- Do not scale the professor cube-push learned-policy branch to 10k/100k yet.
  The right next step is not bigger evaluation volume; it is redesigning the
  teacher trajectory/contact-speed curriculum or adding a true imitation/resume
  fine-tuning path, followed by another 50-100 iteration PPO and frozen 1k audit.
- Teacher blending inside the action loop is not enough evidence of policy
  learning. Teacher-on diagnostics must be labeled scripted, not learned.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_smooth_limit_model49_eval1024_audit.out:3-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_teacher_warmstart_model49_eval1024_audit.out:3-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_contact_speed_50iter_audit.out:3-27`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_contact_speed_model49_eval1024_audit.out:3-18`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_contact_speed_teacher_on_eval1024_audit.out:3-18`

## D100 - professor cube-push IK branch must distinguish RoArm local IK from IsaacLab built-in Differential IK

Evidence:

- The professor's 2026-05-26 instruction is best interpreted as: if the cube
  endpoint/end-effector target is known, place the robot TCP near the cube using
  IK, then push/tap in IsaacLab and inspect the resulting actions and physics.
  This is not an FK-first instruction.
- The current cube-push env does use IK, but it imports RoArm-local
  `ik_dls`/`fk_tcp` from `sim_scripts.roarm_kinematics` and computes joint poses
  before writing them into IsaacLab (`roarm_rl/roarm_cube_push_env.py:22`,
  `:151-179`, `:181-210`, `:308-331`). FK appears only inside IK and reach
  verification; the user-facing command target remains an endpoint/TCP target.
- IsaacLab itself has a built-in `DifferentialIKController` and
  `DifferentialIKControllerCfg` supporting `command_type` `"position"`/`"pose"`
  and `ik_method="dls"` (`isaaclab/controllers/differential_ik_cfg.py:21-35`).
  The controller computes desired joint positions from current end-effector pose,
  Jacobian, and joint position (`isaaclab/controllers/differential_ik.py:148-174`);
  the task-space action path applies the resulting joint target to the articulation
  (`isaaclab/envs/mdp/actions/task_space_actions.py:200-211`).
- Therefore the next professor-branch experiment should not be described as
  merely "do more PPO" or "scale 10k/100k." It should first add a small
  IsaacLab built-in Differential IK cube-push probe that sends TCP targets near
  the cube, lets IsaacLab's live Jacobian IK compute joint targets, and audits
  no-attach physics push/tap outcomes.

Implication:

- Current RoArm-local IK rollout/PPO results remain useful evidence, but do not
  overclaim them as IsaacLab built-in Differential IK.
- The next valid action for the professor branch is a scoped
  DifferentialIKController-based probe with smoke runtime, CSV/summary/audit,
  and optional GUI demo. It remains separate from Track A grasp and must not be
  used as Track A close_26, hold-lift, dataset, or VLA/PPO success evidence.

Sources:

- `roarm_rl/roarm_cube_push_env.py:22,151-179,181-210,308-331`
- `sim_scripts/roarm_kinematics.py:99-141`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/controllers/differential_ik_cfg.py:21-35`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/controllers/differential_ik.py:148-174`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py:200-211`

## D101 - IsaacLab built-in Differential IK can push the 3cm cube, but the current path still has direction/position failure pockets

Date: 2026-05-27 KST

Decision:

- For the professor's 2026-05-26 cube push/tap branch, keep the next evidence
  track as an IsaacLab built-in `DifferentialIKController` scripted physics
  probe. Do not call it learned policy, dataset readiness, or Track A grasp
  success.
- The first short 16-env smoke was a useful negative: the mechanism ran, but
  approach time / joint-step budget was too small, producing low-motion only.
- A longer reach/horizon setting made the same IsaacLab Differential IK path
  produce real cube motion, and the frozen 1024 headless eval is now the current
  professor-branch evidence point.
- Do not jump from this to learned-policy 10k/100k. The next valid work is to
  reduce the weak `(1, 0)` direction, low-motion pockets, and speed/impact
  outliers, then rerun 1024 and only then consider 10k scripted-stat scaling or
  a proper imitation/RL version.

Evidence:

- Added `sim_scripts/cube3cm_push_diffik_probe.py` md5
  `cbb2176a80ed2a2c55552d0d98bc9ab9`, audit md5
  `5ed85775e31f805f4d43885a1de80246`, and posthoc md5
  `6bfc8ea3eac942d0af4c8fc852738f0e`.
- The probe prints the mechanism contract at runtime: `controller=IsaacLab_DifferentialIKController`,
  `ik_method=dls`, `command_type=position`, `local_roarm_ik_dls_control_loop=NO`,
  `training=NO`, `dataset_generation=NO`, `grasp=NO`,
  `attach_posewrite=NO`, and `rollout_object_posewrite=NO`
  (`diffik_probe_eval1024_seed779_stdout.out:20-21`).
- Short smoke audit: row count matched and mechanism PASS, but
  `low_motion_rate=1.000000000`, `disp_xy_mean_m=0.000007746`, and
  `final_tcp_target_err_mean_m=0.161282191`
  (`diffik_probe_smoke16_seed777_audit.out:1-6`).
- Reach smoke audit: mechanism PASS, controlled `0.937500000`, impact `0`,
  low-motion `0.062500000`, `disp_xy_mean_m=0.048690485`, and final TCP error
  `0.020573265` (`diffik_probe_reach16_seed778_audit.out:1-6`).
- Frozen 1024 audit: mechanism PASS, CSV rows `1024`, controlled
  `0.892578125`, impact `0.023437500`, low-motion `0.136718750`,
  success marker `0.520507812`, `disp_along_push_mean_m=0.033575789`,
  `disp_xy_mean_m=0.034856980`, max speed `1.931515932m/s`, final TCP error
  `0.028779610`, and `diffik_clip_rate_mean=0.658035710`
  (`diffik_probe_eval1024_seed779_audit.out:1-6`).
- Posthoc split: overall line 2 matches the 1024 audit; direction `(1, 0)` is
  the weak bucket with controlled `0.633333333`, impact `0.088888889`, and
  low-motion `0.274074074`; worst initial grid is `(1, 1)` by low+impact
  (`diffik_probe_eval1024_seed779_posthoc.out:1-17`).

Implication:

- The professor's "if endpoint is known, use IK to go near the cube and push"
  direction is now tested more literally in IsaacLab with built-in Differential
  IK and live Jacobians.
- The result is scientifically useful, but not clean enough to scale blindly:
  direction-dependent weakness and residual impact/low-motion must be addressed
  before claiming a robust scripted teacher or before starting learned-policy
  scale-up.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py`
- `sim_scripts/cube3cm_push_diffik_audit.py`
- `sim_scripts/cube3cm_push_diffik_posthoc.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_eval1024_seed779_stdout.out:20-21,400`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_smoke16_seed777_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_reach16_seed778_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_eval1024_seed779_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_eval1024_seed779_posthoc.out:1-17`

## D102 - Differential IK trajectory v2 reduces low-motion but worsens impact/tip risk

Date: 2026-05-27 KST

Decision:

- For the professor's cube3cm push/tap branch, keep the scripted IsaacLab
  Differential IK path, but do not promote trajectory v2 to a teacher or scale-up
  source.
- v2 is a mixed result: it fixes much of the low-motion pocket, including the
  `(1,1)` initial grid, but it increases tip/impact risk, especially in the weak
  `(1,0)` direction.
- The next valid step is trajectory v3 impact/tip control, not 10k/100k scripted
  scaling and not PPO/VLA learning.

Evidence:

- `sim_scripts/cube3cm_push_diffik_probe.py` now has a default-preserving
  `--trajectory_variant v2` path for `(1,0)`: closer precontact, lower TCP target
  height, shorter push-through, longer approach/push horizon, and smaller
  per-step DiffIK joint cap.
- v2 smoke16 seed780 exited 0 and audit lines 1-6 PASS, but summary
  `v2_posx_env_count=0`, so it was only a mechanism smoke and not evidence about
  the weak direction.
- v2 reach16 seed779 exited 0, included 6 `(1,0)` envs, and audit lines 1-6 PASS:
  controlled `1.000000000`, impact `0`, low-motion `0.062500000`.
- v2 frozen 1024 seed779 exited 0 and audit lines 1-6 PASS: controlled
  `0.932617188`, impact `0.038085938`, low-motion `0.051757812`, success marker
  `0.580078125`, final TCP error `0.024324538`, and clip rate `0.666682201`.
- Same-seed v1/v2 comparison lines 1-3 show rows `1024/1024`; overall controlled
  improved `0.892578125 -> 0.932617188`, low-motion improved
  `0.136718750 -> 0.051757812`, and final TCP error improved
  `0.028779610 -> 0.024324538`, but impact worsened
  `0.023437500 -> 0.038085938`.
- In direction `(1,0)`, controlled improved `0.633333333 -> 0.785185185` and
  low-motion improved `0.274074074 -> 0.085185185`, but impact worsened
  `0.088888889 -> 0.144444444`, success marker dropped
  `0.533333333 -> 0.440740741`, and tip p95/max increased.
- Grid `(1,1)` improved strongly on low-motion
  `0.304687500 -> 0.023437500` and controlled
  `0.796875000 -> 0.914062500`, but impact became nonzero
  `0 -> 0.031250000`.

Implication:

- v2 is useful physics evidence because it shows the pocket is trajectory
  sensitive, not a hard impossibility.
- It is still not robust enough to use as a scripted teacher. v3 should reduce
  `(1,0)` tip/impact while preserving v2's reach improvement, likely by testing
  lower/less edge-prone contact height, shorter or staged push-through in
  high-tip pockets, and a small lateral-offset sign sweep.
- This remains scripted Differential IK physics evidence only: not learned policy,
  not Track A grasp success, and not dataset readiness.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_smoke16_seed780_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_smoke16_seed780_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_reach16_seed779_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_reach16_seed779_posthoc.out:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_eval1024_seed779_stdout.out:20-21`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_eval1024_seed779_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_eval1024_seed779_posthoc.out:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_eval1024_seed779_compare_to_v1.out:1-8`

## D103 - Differential IK trajectory v3 fixes most v2 tip impact but is still scripted evidence

Date: 2026-05-27 KST

Decision:

- For the professor's cube3cm push/tap branch, trajectory v3 is now the preferred
  scripted IsaacLab Differential IK candidate for scale robustness testing.
- v3 should not be described as learned-policy success, Track A grasp success, or
  dataset readiness. It is still a scripted Differential IK physics result.
- A 10,240-env scripted v3 robustness audit is now scientifically defensible if
  the goal is the professor's "10,000 test"; using v3 as a teacher still needs a
  separate decision because `(1,0)` success/displacement remains weaker than the
  overall metrics suggest.

Evidence:

- Static/posthoc v2 analysis found that all 39 `(1,0)` v2 impact rows were caused
  by tip-angle exceeding the audit p99 threshold, not by final speed or total XY
  displacement. This justified v3's lower contact height, shorter push-through,
  longer/slower pos-x trajectory, and smaller pos-x joint-step cap.
- Current `sim_scripts/cube3cm_push_diffik_probe.py` md5 is
  `f4c8dfe7d9117d733ec38a0ac68e4019`; it adds default-preserving
  `--trajectory_variant v3`.
- v3 smoke16 seed780 exited 0 and audit lines 1-6 PASS: controlled `1.000000000`,
  impact `0`, low-motion `0`, but summary lines 37/49 show
  `v3_posx_env_count=0`, so this was mechanism-only.
- v3 reach16 seed779 exited 0, included 6 `(1,0)` envs, and audit lines 1-6 PASS:
  controlled `1.000000000`, impact `0`, low-motion `0.062500000`.
- v3 frozen 1024 seed779 exited 0 and audit lines 1-6 PASS: rows `1024`,
  controlled `0.969726562`, impact `0.004882812`, low-motion `0.035156250`,
  success marker `0.604492188`, final TCP error `0.023551417`, zero rollout
  posewrite, no training/dataset/grasp/attach.
- v3 posthoc line 6 shows weak direction `(1,0)` is still the weakest, but now
  controlled `0.929629630`, impact `0.014814815`, low-motion `0.088888889`.
- Same-seed v1/v2/v3 comparison lines 2-3 show overall impact
  `0.023437500 -> 0.038085938 -> 0.004882812` and `(1,0)` impact
  `0.088888889 -> 0.144444444 -> 0.014814815`; tip p95 in `(1,0)` improved
  `153.082306 -> 161.068298 -> 140.676743`.
- Critical caveat: comparison line 3 also shows `(1,0)` success marker
  `0.533333333 -> 0.440740741 -> 0.314814815`, final TCP error
  `0.043867240 -> 0.033135812 -> 0.039558476`, and clip
  `0.900232803 -> 0.977681103 -> 1.000000000`.

Implication:

- v3 is a better answer to "can IsaacLab Differential IK go near the cube and
  push/tap it physically?" than v1/v2 because it reduces the direction-specific
  tip-impact pocket below the earlier 5% gate.
- The next professor-branch scale test can be 10,240 env trials with v3, provided
  the result is framed as scripted physics robustness, not PPO/VLA learning.
- If the objective changes from push/tap evidence to teacher/dataset generation,
  run a small v3.1 sweep to recover `(1,0)` displacement/success while preserving
  v3's low impact.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_smoke16_seed780_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_smoke16_seed780_summary.json:37,49`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_reach16_seed779_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_reach16_seed779_posthoc.out:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval1024_seed779_stdout.out:20-21`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval1024_seed779_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval1024_seed779_posthoc.out:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval1024_seed779_compare_to_v1_v2.out:1-10`

## D104 - v3 10,240-trial DiffIK audit passes professor push/tap scale, not teacher readiness

Date: 2026-05-27 KST

Decision:

- The professor-branch "10,000 test" is complete for scripted IsaacLab
  Differential IK physics evidence: use the v3 10,240-trial audit as the current
  scale result.
- Do not run more scale-up just to increase the trial count until a new question
  is defined. The 10,240 result already answers the immediate robustness-statistics
  question better than the prior 1024-only evidence.
- Do not treat the 10,240 result as learned-policy success, Track A grasp success,
  or dataset/teacher readiness. The `(1,0)` low-motion/success caveat remains.

Evidence:

- `sim_scripts/cube3cm_push_diffik_probe.py` md5
  `dc6ca5a222f0bd9437d5f83bf5449729` keeps v3 trajectory behavior and fixes
  multi-episode accounting so posewrite calls are accumulated across episodes and
  `v3_posx_trial_count` is reported.
- The v3 10,240 run used `num_envs=1024`, `episodes=10`, seed `779`, and stdout
  lines 20-21 confirm IsaacLab built-in `DifferentialIKController`, no RoArm-local
  IK loop, no training, no dataset generation, no grasp/attach/object posewrite,
  `trajectory_variant=v3`, and total trials `10240`.
- Audit lines 1-6 PASS: `csv_rows=10240`, row count match, mechanism OK, zero
  posewrite during rollout, controlled `0.943164062`, impact `0.007519531`,
  low-motion `0.042480469`, success marker `0.594824219`, final TCP target error
  `0.023529604`, learned policy `NO`, Track A grasp success `NO`, dataset ready
  `NO`.
- Posthoc line 6 shows `(1,0)` remains the weakest direction: n=2566, controlled
  `0.874512860`, impact `0.012860483`, low-motion `0.122759158`, success marker
  `0.296570538`.
- Compare-to-1024 lines 2-3 show scaling degradation but still good impact:
  overall impact `0.004882812 -> 0.007519531`, `(1,0)` impact
  `0.014814815 -> 0.012860483`, while `(1,0)` low-motion worsens
  `0.088888889 -> 0.122759158`.
- Compare lines 10-14 show impact causes remain mostly tip-angle outliers: overall
  77 impact rows, 65 tip-only, 9 displacement-only, 3 tip+displacement; no impact
  row was caused by final speed alone.
- Compare lines 15-24 show per-episode impact stays below about 1.2% across all 10
  episodes.

Implication:

- For a professor-facing push/tap result, the honest statement is: "10,240
  scripted IsaacLab Differential IK trials ran; the controller physically pushed
  the cube with overall controlled 94.3%, impact 0.75%, low-motion 4.25%, and no
  training/dataset/grasp/posewrite."
- For teacher/dataset use, v3 still needs a v3.1 sweep focused on `(1,0)`
  low-motion/success recovery while preserving impact below the current 1-2%
  range.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval10240_seed779_stdout.out:20-21`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval10240_seed779_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval10240_seed779_posthoc.out:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval10240_seed779_summary.json:18-53`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval10240_seed779_compare_to_1024.out:1-25`

## D105 - Professor v3 video is a replay visualization, not new learning/dataset evidence

Date: 2026-05-27

Decision:

- Use the generated v3 MP4s as professor-facing visualization of scripted
  IsaacLab Differential IK push/tap samples. Prefer the four-direction parallel
  replay for presentation because it shows all four push directions at once.
- Do not count the MP4 as a dataset, training result, learned-policy evidence, or
  additional physics trial. It is a replay of a captured trace.
- When presenting the video, explicitly say it shows direction `(1,0)` in the
  local frame for env3, i.e. push along positive X from the selected cube start
  position. In the four-direction video, env0 is `(0,-1)`, env3 is `(1,0)`,
  env4 is `(0,1)`, and env7 is `(-1,0)`.

Evidence:

- Selected source row is `diffik_probe_v3_reach16_seed779.csv:5`: env_id `3`,
  direction `(1,0)`, cube start `(0.353590250,-0.073313951)m`, displacement
  `0.036002159m`, controlled `1`, impact `0`, low-motion `0`, success marker
  `1`.
- Trace generation stdout lines 20-21 confirm local IsaacLab built-in
  `DifferentialIKController`, no RoArm-local IK loop, no training, no dataset
  generation, no grasp/attach/object posewrite, and `trajectory_variant=v3`.
- Trace summary lines 46-49 confirm trace CSV path, `trace_env_id=3`,
  `trace_frame_count=145`, and `training=false`.
- Render stdout line 447 confirms `frames=145`, MP4 path, trace path,
  `training=NO`, `dataset_generation=NO`, and `physics_recomputed=NO`.
- Render summary lines 19-27 confirm `30fps`, 145 written frames, `1280x720`,
  output MP4 path, `physics_recomputed=false`, trace CSV, and `training=false`.
- MP4 probe lines 1-8 confirm `opened=True`, `frame_count=145`, `width=1280`,
  `height=720`, `fps=30.0`, first frame decode OK, and file size `722185`
  bytes.
- Four-direction render stdout line 447 confirms `frames=145`, `env_count=4`,
  env IDs `[0, 3, 4, 7]`, `training=NO`, `dataset_generation=NO`, and
  `physics_recomputed=NO`.
- Four-direction real-RoArm render summary lines 12-19/91-112 confirm white
  background, black actual RoArm URDF STL mesh, gray table, pink cube, 2x2
  layout, 145 frames, `robot_visual_mode=black_roarm_urdf_stl_mesh_from_trace_joints`,
  mesh source `local_assets/roarm_m3/urdf/meshes`,
  `physics_recomputed=false`, and `training=false`.
- Four-direction MP4 probe lines 1-8 confirm `opened=True`, `frame_count=145`,
  `width=1280`, `height=720`, `fps=30.0`, first frame decode OK, and file size
  `1234819` bytes.
- Earlier black FK-proxy render artifacts were rejected/superseded because they
  were not actual RoArm geometry.

Implication:

- The video is appropriate for explaining what the v3 scripted controller is
  physically doing near the cube.
- It must be paired with the 10,240-trial audit for statistics; by itself it is
  an illustrative sample, not a robustness claim.

Sources:

- `claudedocs/session_20260527_cube3cm_diffik_v3_visualization.md`
- `sim_scripts/cube3cm_push_diffik_probe.py`
- `sim_scripts/cube3cm_push_diffik_render_trace.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_reach16_seed779.csv:5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_trace_posx_env3_seed779_stdout.out:20-21`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_trace_posx_env3_seed779_summary.json:46-49`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_posx_env3_seed779_stdout.out:447`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_posx_env3_seed779_summary.json:19-27`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_posx_env3_seed779_mp4_probe.out:1-8`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_fourdir_realroarm_env0_3_4_7_seed779_stdout.out:447`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_fourdir_realroarm_env0_3_4_7_seed779_summary.json:12-19`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_fourdir_realroarm_env0_3_4_7_seed779_summary.json:91-112`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_render_fourdir_realroarm_env0_3_4_7_seed779_mp4_probe.out:1-8`

## D106 - Professor cube3cm BC evidence requires dataset audit plus learned rollout audit

Date: 2026-05-28

Decision:

- A raw DiffIK trace or per-trial CSV is not enough to call something a training
  dataset. For professor cube3cm push/tap BC, require a separate dataset artifact
  with observations, actions, labels, metadata, split isolation, direction
  coverage, final teacher-quality filtering, manifest, and audit.
- A supervised BC MSE reduction is not enough to call the result a physics policy
  success. Require an IsaacLab rollout where the learned checkpoint controls
  joint deltas without the DiffIK controller, followed by a learned-rollout audit.
- For the local IsaacLab BC rollout script, write CSV/summary artifacts before
  optional simulator close. In this session explicit `sim_app.close()` hung after
  rollout completion; use the fixed artifact-first path and `--skip_sim_close`
  for this local evaluation route.
- Keep this professor cube3cm push/tap dataset/BC line separate from Track A
  grasp/dataset/training and from PPO/VLA learning claims.

Evidence:

- Dataset build log lines 1-6 report source trace rows `148480`, selected rows
  `46400`, balanced 80 trajectories per direction, train/val/test env split
  `224/48/48`, and `full_dataset_candidate=YES`.
- Dataset audit lines 1-7 report rows `46400`, env count `320`, frames/env
  `145`, schema/finite OK, split leakage OK, direction coverage OK, final rates
  controlled `1.0`, impact `0.0`, low-motion `0.0`, success `1.0`, mechanism OK,
  size OK, and `PASS_FULL_STATE_ACTION_DATASET_V2 full_dataset_ready=YES`.
- BC train log lines 1-4 report train/val/test rows `32480/6960/6960`, final
  test MSE `0.007494668`, mean test MAE `0.000745819rad`, and
  `PASS_BC_TRAINED_CHECKPOINT`.
- Progress logs showed the 4-env 20-step diagnostic reached `rollout_done` and
  wrote artifacts immediately after the artifact-first fix.
- Learned rollout audit lines 1-6 for 1024 envs, seed883 report
  `controller=BC_MLP_joint_delta_policy`, `learned_policy=True`,
  `diffik_controller_used=False`, mechanism OK, posewrite calls 0, controlled
  `0.945312500`, impact `0.012695312`, low-motion `0.026367188`, success
  `0.648437500`, and `PASS_LEARNED_BC_POLICY_ROLLOUT`.
- The same learned rollout audit line 5 preserves the caveat that `(1,0)`
  remains weakest: controlled `0.879844961`, impact `0.038759690`, low-motion
  `0.038759690`, success `0.453488372`.

Implication:

- It is now acceptable to call the new artifact a teacher-filtered state-action
  dataset v2 for the professor cube3cm push/tap branch.
- It is now acceptable to call the checkpoint a learned BC joint-delta policy
  with 1024-env IsaacLab rollout PASS for this branch.
- It is still not Track A grasp success, not PPO/RL, not VLA, not image-dataset
  readiness, and not 10k/100k learned-policy robustness.

Sources:

- `claudedocs/session_20260528_cube3cm_diffik_dataset_bc_policy.md`
- `sim_scripts/cube3cm_push_diffik_build_dataset.py`
- `sim_scripts/cube3cm_push_diffik_dataset_v2_audit.py`
- `sim_scripts/cube3cm_push_diffik_train_bc.py`
- `sim_scripts/cube3cm_push_bc_policy_rollout.py`
- `sim_scripts/cube3cm_push_bc_rollout_audit.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779_build.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779_audit.out:1-7`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779_bc_train.out:1-4`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v1_rollout4_seed882_step20_retry_progress.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v1_rollout1024_seed883_audit.out:1-6`

## D107 - Bucket-balanced BC improvement is not enough if per-bucket impact fails

Date: 2026-05-28

Decision:

- For the professor cube3cm branch, do not let improved overall learned BC success
  override per-direction/per-bucket safety failures.
- v3.2 teacher parameter sweeps must be cross-checked by seed and bucket, not only
  by overall `(1,0)` success. In this session, several low-x parameter candidates
  improved one seed or one metric while failing impact or another seed.
- A bucket-balanced dataset is useful and auditable, but it can make the learned
  policy more forceful in hard buckets. Treat bucket-balanced BC as a gate, not a
  permission to start PPO/RL scale-up.

Evidence:

- The first t270/p036 candidate was initially run with wrong base settings; stdout
  exposed `base_steps=55/35/30` and `max_diffik_joint_step_rad=0.012`, causing a
  false failure. Corrected reruns used `base_steps=220/90/40` and
  `max_diffik_joint_step_rad=0.035`, showing why stdout line checks are mandatory.
- Corrected t270/p036 raised `(1,0)` success on seed790 but also raised `(1,0)`
  impact to `0.154929577`, so it was rejected.
- Conservative t270/p030 and t257/p034 improved seed790 low-x success but did not
  robustly hold on seed791; no scripted v3.2 teacher candidate was accepted.
- Dataset v3 build lines 1-6 report 26,100 rows, 180 trajectories, 45 per
  direction, and `(1,0)` low/mid/high-x bucket counts `15/15/15`.
- Dataset v3 audit lines 1-8 pass schema, finite values, split leakage,
  direction coverage, teacher filtering, mechanism, size, and bucket/split-bucket
  checks.
- BC v2_bucket rollout 1024 seed883 improves overall success to `0.679687500`
  and `(1,0)` success to `0.527131783`, but bucket audit lines 7-9 fail the
  safety screen: low_x impact `0.068493151` and high_x impact `0.061855670`.
- Reducing rollout `policy_delta_clip_rad` to `0.035` improved overall success to
  `0.689453125`, but low_x impact stayed `0.068493151`; clip `0.030` made overall
  impact worse.

Implication:

- PPO/RL should remain blocked for this branch until a small learned rollout passes
  both overall metrics and per-direction/per-bucket impact gates.
- The next valid work is teacher/action-distribution redesign or safety-aware
  BC/RL warm-start objective, followed by another small 1024 frozen audit. Do not
  run 10k/100k learned-policy scaling from the current bucket-balanced BC v2.

Sources:

- `claudedocs/session_20260528_cube3cm_v32_bucket_bc_gate.md`
- `sim_scripts/cube3cm_push_diffik_bucket_audit.py`
- `sim_scripts/cube3cm_push_diffik_build_dataset.py`
- `sim_scripts/cube3cm_push_diffik_dataset_v2_audit.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v31_baseline_eval512_seed790_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v32cand_t270_p036_eval512_seed790_fixed_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v32cand_t257_p034_eval512_seed790_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v32cand_t257_p034_eval512_seed791_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v3_bucket_1024_seed779_build.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v3_bucket_1024_seed779_audit.out:1-8`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v3_bucket_1024_seed779_bc_train.out:1-4`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v2_bucket_rollout1024_seed883_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v2_bucket_rollout1024_seed883_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v2_bucket_clip035_rollout1024_seed883_audit.out:1-6`

## D108 - Safety gate can pass while low-x motion quality remains weak

Date: 2026-05-28

Decision:

- For the professor cube3cm branch, safety-aware BC and per-bucket action scaling can
  be used as a small learned-policy gate after bucket-balanced BC, but only with
  explicit per-bucket audit.
- Passing low_x/high_x impact gates is necessary, not sufficient. Keep low_x
  low-motion/success as a visible caveat and do not promote a two-seed 1024 result
  into 10k/100k learned-policy robustness.
- PPO/RL fine-tuning is still a separate explicit-approval step. This result
  supports the next small warm-start experiment; it is not permission to run large
  PPO/RL or to mix with Track A grasp.

Evidence:

- Safety-aware BC train lines 1-5 passed with rows `26100`, test MSE
  `0.018879525`, mean MAE `0.001225158rad`, and checkpoint md5
  `03b159809ddca64aad6d6449b7f44876`.
- Frozen learned rollout seed883 passed the overall learned-policy audit with
  controlled `0.953125000`, impact `0.004882812`, low-motion `0.030273438`, and
  success `0.662109375`.
- Seed883 bucket audit passed: `(1,0)` low_x impact `0.041095890`, high_x impact
  `0`, but low_x low-motion was `0.315068493`.
- Cross-seed seed884 also passed the overall learned-policy audit with controlled
  `0.943359375`, impact `0.010742188`, low-motion `0.024414062`, and success
  `0.662109375`.
- Seed884 bucket audit passed: `(1,0)` low_x impact `0.035714286`, high_x impact
  `0`, but low_x low-motion was `0.261904762`.
- Compared with BC v2 seed883, impact improved but low_x motion quality regressed:
  BC v2 low_x impact/low-motion/success was `0.068493151` / `0.082191781` /
  `0.410958904`, while safety BC v3 seed883 low_x was `0.041095890` /
  `0.315068493` / `0.315068493`.

Implication:

- The next valid work is either to recover low_x motion/success while preserving
  low impact, or to run an explicitly approved small safety-aware RL warm-start
  pilot from this checkpoint and repeat the same 1024 per-bucket audit.
- Do not run 10k/100k learned-policy robustness, PPO scale-up, dataset generation,
  or Track A runtime from this result without explicit approval and a written gate.

Sources:

- `claudedocs/session_20260528_cube3cm_safety_bc_gate.md`
- `sim_scripts/cube3cm_push_diffik_train_bc.py`
- `sim_scripts/cube3cm_push_bc_policy_rollout.py`
- `sim_scripts/cube3cm_push_diffik_bucket_audit.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v3_bucket_1024_seed779_bc_v3_safety_l2_train.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed883_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed883_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed884_audit.out:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed884_bucket.out:1-10`

## D109 - PPO warm-start smoke is connected, but teacher-off performance failed

Date: 2026-05-28

Decision:

- For the professor cube3cm branch, the safety-aware BC checkpoint cannot be treated as
  an rsl_rl PPO checkpoint. The bridge must go through environment-level BC teacher
  action/imitation reward or a separate residual-learning design.
- The approved small PPO warm-start smoke proves the local IsaacLab/GPU PPO path is
  connected, but it does not produce a successful learned policy.
- Teacher-off frozen audit is the only valid learned PPO performance check. The
  teacher-off 1024 rollout failed performance and per-bucket gates, so do not scale
  PPO/RL from `model_11.pt`.
- The direct-like teacher-on diagnostic partially recovers performance, which means the
  BC checkpoint is not dead. The current blocker is the mismatch between direct BC
  joint-target replay and the PPO env normalized action-loop/safety curriculum.

Evidence:

- Direct system Python failed before Isaac with `ModuleNotFoundError: No module named
  'gymnasium'`; valid local runtime used `conda run -n isaaclab`.
- PPO smoke12 seed885 ran on `cuda:0`, used BC teacher blend `0.35`, imitation reward
  `0.30`, completed 73,728 timesteps, and wrote `model_11.pt` md5
  `c9f945a4d1eacd817d4733e7d9b7e48e`.
- Training logs show the BC teacher was active: iteration 0 logged
  `cube_push_bc_teacher_blend_mean=0.3500`,
  `cube_push_bc_teacher_imitation_mse=0.5454`, and
  `bc_teacher_imitation_penalty=-0.1636`.
- Teacher-off 1024 eval seed886 was mechanism-clean but performance-failed:
  controlled `0.470703125`, impact `0.087890625`, low-motion `0.344726562`,
  success `0.078125000`.
- Teacher-off bucket audit failed: `(-1,0)` impact `0.230483271`, `(1,0)` success
  `0.070833333`, low_x success `0.197530864`, mid_x/high_x success
  `0` / `0.014492754`.
- Teacher-on short/safety-limited diagnostic was also weak: success `0.050781250`,
  impact `0.097656250`.
- Direct-like teacher-on diagnostic with 6s horizon, home reset, and relaxed action loop
  improved to controlled `0.792968750`, impact `0.054687500`, low-motion
  `0.175781250`, success `0.417968750`; `(1,0)` success was `0.516666667`.

Implication:

- Do not call `model_11.pt` a successful learned cube-push policy.
- Do not run 10k/100k learned-policy robustness, PPO scale-up, dataset generation, or
  Track A runtime from this PPO smoke.
- Next valid work is to redesign the BC teacher bridge/horizon/safety curriculum, then
  rerun teacher-off frozen 1024 overall and per-bucket audits before any larger RL.

Sources:

- `claudedocs/session_20260528_cube3cm_safety_rl_warmstart.md`
- `roarm_rl/roarm_cube_push_env.py`
- `roarm_rl/train_cube_push_ppo.py`
- `roarm_rl/eval_cube_push_policy.py`
- `sim_scripts/cube3cm_push_ppo_rollout_audit.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_smoke12_seed885_stdout.out:47-78`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_smoke12_seed885_stdout.out:121-155`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_smoke12_seed885_stdout.out:573-614`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_smoke12_seed886_eval1024_stdout.out:48-100`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_smoke12_seed886_eval1024_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_smoke12_seed886_eval1024_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_teacheron_seed887_eval512_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_teacheron_seed887_eval512_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_teacheron_directlike_seed888_eval256_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_teacheron_directlike_seed888_eval256_bucket.out:1-10`

## D110 - Matching the BC bridge action loop restores teacher-on, but PPO actor still fails teacher-off

Date: 2026-05-29

Decision:

- For the professor cube3cm branch, the BC teacher bridge should use the direct
  BC rollout's timing/action semantics when diagnosing teacher fidelity:
  direct step timing, home reset/no IK endpoint reset, joint-position-referenced
  joint deltas, relaxed smoothing/slowdown, and explicit low_x scaling.
- A teacher-on bridge PASS is not a learned PPO policy. It only proves the bridge
  can execute the BC teacher through the PPO env action path.
- PPO reward-side imitation alone did not initialize the actor in the small
  smoke8 diagnostic; teacher-off remained zero-motion.

Evidence:

- Added default-preserving bridge controls in `roarm_rl/roarm_cube_push_env.py`
  md5 `a0483108ef0fc8ab2f27a58b6edd8c13`, `train_cube_push_ppo.py` md5
  `7032616ded5617b546149227f4c0d110`, and `eval_cube_push_policy.py` md5
  `b10fad43cfd3b0ca543390ad6011135f`.
- Teacher-on direct-step/joint-pos bridge with lowx scale `1.0` passed the small
  128-env screen: controlled `0.992187500`, impact `0.007812500`, low-motion
  `0.007812500`, success `0.765625000`; bucket PASS with low_x success
  `0.538461538`.
- Existing `model_11.pt` under the redesigned action loop failed teacher-off 128:
  controlled `0`, low-motion `1`, success `0`.
- New 128-env PPO distillation smoke8 wrote `model_7.pt` md5
  `5ed5ac34dc624ac8c660d9176378b357`, but imitation MSE stayed around
  `0.56-0.59`.
- That smoke8 `model_7.pt` also failed teacher-off 128: controlled `0`,
  low-motion `1`, success `0`.

Implication:

- Do not run teacher-off 1024, PPO scale-up, 10k/100k learned robustness, dataset
  generation, or Track A runtime from `model_11.pt` or the smoke8 `model_7.pt`.
- Next valid work is true supervised actor/normalized-action distillation or a
  stronger actor initialization path, then a small teacher-off 128 audit before
  any 1024 audit.

Sources:

- `claudedocs/session_20260529_cube3cm_bc_teacher_bridge_redesign.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_bridge_directstep_jointpos_lowx100_seed890_eval128_summary.json:1-58`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_bridge_directstep_jointpos_lowx100_seed890_eval128_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_bridge_directstep_jointpos_lowx100_seed890_eval128_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_newloop_teacheroff_model11_seed891_eval128_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_warmstart_newloop_teacheroff_model11_seed891_eval128_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_newloop_distill_smoke8_seed892_stdout.out:3-30`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_newloop_distill_smoke8_seed892_stdout.out:102`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_newloop_distill_smoke8_seed892_stdout.out:144`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_newloop_distill_smoke8_seed892_stdout.out:349`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_newloop_distill_smoke8_seed892_model7_teacheroff_eval128_seed893_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_newloop_distill_smoke8_seed892_model7_teacheroff_eval128_seed893_bucket.out:1-10`

## D111 - One-step actor distillation lowers imitation error, but closed-loop teacher-off 128 still fails

Date: 2026-05-29

Decision:

- For the professor cube3cm branch, directly fitting the rsl_rl actor mean to the
  BC teacher's normalized joint-delta actions is necessary evidence, but it is
  not sufficient to claim a successful learned policy.
- The actor-distilled checkpoint moved teacher-off behavior away from pure
  zero-motion, but the closed-loop 128 gate still failed overall and per-bucket
  performance.
- A low one-step action MSE under teacher-collected states must not be treated as
  a rollout success claim; closed-loop state-distribution and target-update
  effects remain the blocker.

Evidence:

- Added `roarm_rl/distill_cube_push_actor.py` for supervised rsl_rl actor
  distillation. It collects teacher actions from the direct-step/joint-pos BC
  bridge and writes a normal rsl_rl checkpoint with actor observation
  normalization.
- Distillation seed894 used 128 envs x 600 steps = 76,800 samples from
  `model_7.pt`; `model_actor_distill.pt` md5
  `57811cfb054ca7ac39b134d1d97cd543`.
- Distillation metrics improved sharply: initial val MSE `0.169735238` to final
  val MSE `0.000794161`; teacher action abs mean `0.280680388`, actor action abs
  mean `0.281257778`.
- The only allowed teacher-off rollout gate, 128 envs seed895, was mechanism-clean
  but performance-failed: controlled `0.101562500`, impact `0`, low-motion
  `0.929687500`, success `0.031250000`.
- Per-bucket audit failed: low_x success `0`, mid_x success `0`, high_x success
  `0.076923077`; overall low-motion remained `0.929687500`.

Implication:

- Do not run teacher-off 1024, PPO scale-up, 10k/100k learned robustness, dataset
  generation, or Track A runtime from `model_actor_distill.pt`.
- Next valid work is a closed-loop/action-target analysis of why low-MSE one-step
  normalized-action imitation collapses to low-motion, or a stronger
  rollout-aware actor initialization, then another teacher-off 128 audit only.

Sources:

- `claudedocs/session_20260529_cube3cm_actor_distillation_gate.md`
- `roarm_rl/distill_cube_push_actor.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_distill_seed894/actor_distill_stdout.out:47-72`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_distill_seed894/actor_distill_metrics.json:19-24`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_distill_seed894/model_actor_distill_teacheroff_eval128_seed895_stdout.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_distill_seed894/model_actor_distill_teacheroff_eval128_seed895_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_distill_seed894/model_actor_distill_teacheroff_eval128_seed895_bucket.out:1-10`

## D112 - Waypoint observation plus on-policy actor distillation can pass the first teacher-off 128 gate

Date: 2026-05-29

Decision:

- For the professor cube3cm branch, the plain 28D observation with final push
  target fields was insufficient for stable rsl_rl actor imitation of the
  time/waypoint-conditioned BC teacher.
- A default-off waypoint-observation mode, `policy_obs_target_mode =
  bc_teacher_tcp_target`, is valid as a diagnostic/learned-policy bridge only
  when disclosed: it exposes the teacher's moving TCP waypoint in the existing
  target observation slots, while still using teacher-off actor actions.
- `model_actor_waypoint_lowx130.pt` passed the first teacher-off 128
  first-episode overall/per-bucket gate, but must not be called 1024-robust,
  10k/100k robust, Track A success, or PPO/RL/VLA final success.

Evidence:

- Trace of `model_actor_distill.pt` showed target application was not the main
  blocker: actor abs `0.181961636` vs teacher abs `0.514814639`, while
  effective-vs-actor MSE was only `0.001083134`; rollout remained low-motion
  with success `0.031250000`.
- The actor observation exposes final target fields in
  `roarm_rl/roarm_stack_env.py:517-529`, while the BC teacher uses
  `phase_alpha` and moving target features in
  `roarm_rl/roarm_cube_push_env.py:390-411`.
- Waypoint-only distillation improved seed901 teacher-off 128 to controlled
  `0.679687500`, impact `0`, low-motion `0.242187500`, success `0.273437500`,
  but failed low_x bucket.
- Waypoint on-policy DAgger1 improved seed904 to controlled `0.968750000`,
  impact `0`, low-motion `0.070312500`, success `0.617187500`, but low_x
  success `0.117647059` was below the bucket threshold.
- Low_x scale `1.3` on-policy distillation seed905 wrote
  `model_actor_waypoint_lowx130.pt` md5
  `606d19fff713e7468d395af4a027d08a`.
- Teacher-off 128 first-episode seed906 passed mechanism and bucket gates:
  controlled `0.937500000`, impact `0`, low-motion `0.093750000`, success
  `0.546875000`; bucket PASS with low_x success `0.571428571`, mid_x success
  `0`, high_x success `0.428571429`, and zero impact in all three posx buckets.

Implication:

- Next valid gate is an explicitly approved teacher-off 1024 first-episode audit
  of `model_actor_waypoint_lowx130.pt` using
  `policy_obs_target_mode=bc_teacher_tcp_target` and no teacher action blend.
- Until that 1024 audit passes, do not run PPO scale-up, 10k/100k learned
  robustness, dataset generation, or Track A runtime from this checkpoint.

Sources:

- `claudedocs/session_20260529_cube3cm_waypoint_actor_gate.md`
- `roarm_rl/roarm_cube_push_env.py`
- `roarm_rl/distill_cube_push_actor.py`
- `roarm_rl/eval_cube_push_policy.py`
- `roarm_rl/trace_cube_push_actor.py`
- `roarm_rl/analyze_cube_push_trace.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_trace_seed896/actor_trace_analysis.out:1-4`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_trace_waypoint_seed902/actor_trace_analysis.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_dagger_seed903/model_actor_waypoint_dagger1_teacheroff_eval128_seed904_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_dagger_seed903/model_actor_waypoint_dagger1_teacheroff_eval128_seed904_firstonly_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/actor_waypoint_lowx130_metrics.json:1-45`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval128_seed906_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval128_seed906_firstonly_bucket.out:1-10`

## D113 - Waypoint actor passed three teacher-off 1024 first-episode gates

Date: 2026-05-29

Decision:

- After explicit user approval, `model_actor_waypoint_lowx130.pt` is now a
  teacher-off learned-policy artifact with three 1024-env first-episode
  overall/per-bucket PASS results on seeds 907, 908, and 909.
- This justifies calling it a 3x1024 teacher-off robust learned-policy gate PASS
  for the professor cube3cm waypoint-observation branch.
- It must still not be called 10k/100k robust, dataset-ready, Track A evidence,
  or PPO/RL/VLA final success.

Evidence:

- Seed907 audit PASS: controlled `0.924804688`, impact `0`, low-motion
  `0.109375000`, success `0.511718750`; bucket PASS with low_x success
  `0.415730337`, mid_x success `0.303797468`, high_x success `0.215909091`.
- Seed908 audit PASS: controlled `0.925781250`, impact `0`, low-motion
  `0.114257812`, success `0.523437500`; bucket PASS with low_x success
  `0.412371134`, mid_x success `0.173333333`, high_x success `0.212765957`.
- Seed909 audit PASS: controlled `0.920898438`, impact `0`, low-motion
  `0.125976562`, success `0.506835938`; bucket PASS with low_x success
  `0.395348837`, mid_x success `0.182926829`, high_x success `0.151515152`.
- All three 1024 summaries used `bc_teacher_blend=0.0`, no BC teacher checkpoint
  during evaluation, `policy_obs_target_mode=bc_teacher_tcp_target`, and
  first-episode-only recording.

Implication:

- The next stricter claim requires explicit approval for a larger robustness
  audit such as 10k; do not run 10k/100k, PPO scale-up, dataset generation, or
  Track A runtime from this checkpoint without that approval.

Sources:

- `claudedocs/session_20260529_cube3cm_waypoint_actor_gate.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_seed907_firstonly_summary.json:1-60`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_seed907_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_seed907_firstonly_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_seed908_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_seed908_firstonly_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_seed909_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_seed909_firstonly_bucket.out:1-10`

## D114 - High-x actor retune rejected; gain 0.045 is a non-canonical candidate

Date: 2026-05-29

Decision:

- Do not replace `model_actor_waypoint_lowx130.pt` with the high_x scale `1.0`
  actor candidate. It passed one-step distillation metrics but failed the
  teacher-off 128 posx bucket gate.
- The same checkpoint with deployment gain `action_scale=0.045`,
  `max_joint_delta_per_step_rad=0.045`, and
  `joint_target_lead_limit_rad=0.0675` passed seed911 128 and seeds 907/908/909
  1024 overall/per-bucket gates.
- However, gain `0.045` is not a clear canonical replacement: it improves
  mid_x and overall success slightly, leaves high_x unchanged, and is a control
  gain change rather than a new learned actor.
- Keep canonical status on `model_actor_waypoint_lowx130.pt` with the previously
  verified gain `0.040`, and treat gain `0.045` as a valid non-canonical
  deployment candidate.

Evidence:

- Three-seed canonical aggregate showed remaining posx failures are displacement
  limited: all low_x/mid_x/high_x fail cases have `disp_lt_0p030=1.000000000`
  and no target-distance/speed failure.
- High_x scale `1.0` actor wrote checkpoint md5
  `12c98baec7deb17a96dd38fdb22b9a42`, with validation MSE
  `0.048445213586091995 -> 0.0002650115347933024`, but teacher-off 128 seed911
  bucket failed: low_x success `0.166666667`, high_x success `0`.
- Gain `0.045` passed teacher-off 128 seed911 and 1024 seeds 907/908/909
  overall/per-bucket. Across 3072 1024 trials, it changed overall success
  `0.513997396 -> 0.519205729`, posx success
  `0.280423280 -> 0.305555556`, mid_x success
  `0.220338983 -> 0.300847458`, and high_x success
  `0.197580645 -> 0.197580645`.
- Gain `0.050` passed seed911 128 and a seed907 1024 pilot, but was mixed and
  not continued.

Implication:

- Do not run 10k/100k, dataset generation, PPO scale-up, or Track A runtime
  from any of these candidates without explicit approval.
- Next useful work is either a stricter approved robustness audit of the
  canonical setup, or a targeted actor/observation redesign specifically for
  displacement-limited high_x without degrading low_x.

Sources:

- `claudedocs/session_20260529_cube3cm_waypoint_actor_gate.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_3seed_aggregate.out:1-14`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval1024_3seed_failure_modes.out:1-13`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_highx100_seed910/actor_waypoint_highx100_metrics.json:1-45`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_highx100_seed910/model_actor_waypoint_highx100_teacheroff_eval128_seed911_firstonly_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_gain045_vs_gain040_3seed_compare.out:1-12`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_candidate_summary.out:1-10`

## D115 - Canonical waypoint actor passed sharded 10k teacher-off gate

Date: 2026-05-29

Decision:

- After explicit approval, the canonical setup
  `model_actor_waypoint_lowx130.pt` md5
  `606d19fff713e7468d395af4a027d08a` with gain `0.040` passed a sharded
  10,240-trial teacher-off first-episode learned-policy robustness gate.
- The single-stage 10240-env attempt failed during IsaacLab environment creation,
  before policy rollout; treat that as a runtime/stage creation failure, not as
  a learned-policy failure.
- It is now valid to call the canonical setup a sharded 10k teacher-off robust
  learned-policy gate PASS for the professor cube3cm waypoint-observation branch.
- It is still not dataset-ready evidence, not Track A evidence, not PPO/RL/VLA
  final success, and not proof that mid_x/high_x success is solved.

Evidence:

- The single-stage 10240-env run failed with `Stage.GetPrimAtPath(Stage,
  NoneType)` during ground-plane setup and produced no rollout CSV/summary.
- Ten independent 1024-env first-episode shards, seeds 912-921, completed and
  combined into 10240 rows.
- Aggregate mechanism audit PASS: controlled `0.927148437`, impact
  `0.000097656`, low-motion `0.106054687`, success `0.524902344`.
- Aggregate bucket audit PASS: low_x success `0.406947891`, mid_x success
  `0.183497537`, high_x success `0.213625866`, all posx impact `0`.
- All ten individual shard bucket audits PASSed.
- Failure-mode audit confirmed posx failures remain displacement-limited:
  failed low_x/mid_x/high_x cases all have `disp_lt_0p030=1.000000000`.

Implication:

- Canonical waypoint actor evidence has advanced from 3x1024 to sharded 10k
  teacher-off robustness.
- Next work should not be dataset generation or Track A by implication. The next
  useful branch-specific improvement is a targeted redesign for displacement-
  limited mid_x/high_x success, with explicit small gates before any larger
  audit.

Sources:

- `claudedocs/session_20260529_cube3cm_waypoint_actor_gate.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_seed912_firstonly_stderr.out:1-26`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_10kshards_seed912_921_driver.out:1-20`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_firstonly_summary.json:1-77`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_firstonly_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_failure_modes.out:1-13`

## D116 - Professor cube push/tap reporting must not collapse to only the 3cm success marker

Date: 2026-06-02

Decision:

- For the professor cube3cm push/tap branch, do not treat the old 3cm
  `success_marker` as the sole objective.
- Keep `success_marker` as a strict task marker, but report a hierarchical push
  table: `1/5/10/20/30mm`, `disp/object_size`, controlled, no-impact,
  low-motion, and direction/posx buckets.
- Use `disp_along_push_m / cube_size_m` when comparing different object sizes.

Evidence:

- Code fixes the current cube at `CUBE_SIZE_M=0.030` and mass `0.020kg`.
- Env success requires controlled, no impact, `disp_along >= 0.030m`,
  target-distance tolerance, and speed cap. This is stricter than "object moved."
- Sharded 10k threshold analysis shows the canonical actor is strong at smaller
  stable pushes even where the 30mm marker is weak. For direction `(1,0)`,
  5mm/10mm/20mm rates are `0.906199678` / `0.842592593` / `0.770531401`, but
  30mm is only `0.266505636`.
- Posx mid/high buckets are near-perfect at 10mm and strong at 20mm, but fall
  sharply at 30mm.

Implication:

- Do not summarize this branch as "forward push cannot work" merely because the
  30mm marker is weak.
- Do not claim all directions are equally solved. At 30mm, `+y` is strong,
  `-y` is moderate, `-x` is moderate/weak, and `+x` is weak.
- Before changing cube size, explicitly log `cube_size_m`, `cube_mass_kg`,
  `density_kg_m3`, and `disp/object_size`.

Sources:

- `claudedocs/session_20260602_cube3cm_push_metric_reframe_targetext.md`
- `roarm_rl/roarm_cube_push_env.py:31`
- `roarm_rl/roarm_cube_push_env.py:60-77`
- `roarm_rl/roarm_cube_push_env.py:94-100`
- `roarm_rl/roarm_cube_push_env.py:691-714`
- `roarm_rl/roarm_cube_push_env.py:781-787`
- `roarm_rl/eval_cube_push_policy.py:219-240`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_threshold_analysis.out:1-9`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_failure_modes.out:1-13`

## D117 - Weighted mid/high actor and target-extension probes are small diagnostics, not scale-up candidates

Date: 2026-06-02

Decision:

- Reject the weighted mid/high actor candidate
  `model_actor_waypoint_midhighw200_post150.pt` despite good one-step fit.
- Treat the canonical target-extension probe
  `bc_teacher_midx_push_through_m=0.030`,
  `bc_teacher_highx_push_through_m=0.035` as a local diagnostic only.
- Do not run 1024/10k, dataset generation, PPO scale-up, or Track A runtime from
  either probe without a new small-gate design and explicit approval.

Evidence:

- Weighted candidate final validation MSE was small (`0.00023266756033990532`)
  and weighted validation MSE was small (`0.00021922370069660246`), but
  teacher-off 128 bucket screen failed: low_x success `0.083333333`, mid_x
  `0.090909091`, high_x `0.833333333`.
- Target-extension probe improved same-seed mid/high locally: mid_x
  `0.272727273`, high_x `1.000000000`, but still failed the 128 bucket screen
  because low_x success was `0.166666667`.
- Diagnostic trace showed contact reached and actor actions were applied, so the
  current issue is mainly displacement/gate design and distribution-specific
  action magnitude, not a dead action path.

Implication:

- Next work should first define the hierarchical push table and low_x handling,
  then run only a tiny 128 teacher-off gate.
- Do not jump from one 128 target-extension improvement to larger audits.

Sources:

- `claudedocs/session_20260602_cube3cm_push_metric_reframe_targetext.md`
- `roarm_rl/distill_cube_push_actor.py`
- `roarm_rl/roarm_cube_push_env.py`
- `roarm_rl/eval_cube_push_policy.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_midhighw200_post150_seed923/actor_waypoint_midhighw200_post150_metrics.json:22-47`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_midhighw200_post150_seed923/model_actor_waypoint_midhighw200_post150_teacheroff_eval128_seed911_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_midhighw200_post150_seed923/model_actor_waypoint_midhighw200_post150_teacheroff_eval128_seed911_firstonly_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_targetext_m030_h035_teacheroff_eval128_seed911_firstonly_summary.json:1-45`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_targetext_m030_h035_teacheroff_eval128_seed911_firstonly_audit.out:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_targetext_m030_h035_teacheroff_eval128_seed911_firstonly_bucket.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/diagnostic_trace_seed911/actor_trace_analysis.out:1-28`

## D118 - Professor 10cm object diagnostic uses a 1cm primary push gate, not full object-length displacement

Date: 2026-06-04

Decision:

- Interpret the professor's current `10*10*10` request as a separate
  10cm/0.72kg cube-like object push/tap diagnostic for the professor branch.
- Do not require moving the 10cm object by 10cm. The primary first gate is about
  1cm displacement: `cube_push_target_disp_m=0.010`,
  `cube_success_disp_m=0.010`, and `gate_disp_m=0.010`.
- Use the IsaacLab built-in `DifferentialIKController` probe path for the first
  teacher diagnostic; do not restart TCP/IK design from scratch unless the gate
  shows a concrete geometry/control failure.
- Do not run the tiny 128 DiffIK gate, any 1024/10k scale-up, dataset generation,
  PPO/RL scale-up, VLA training, or Track A runtime without explicit approval.

Evidence:

- The current 3cm evidence already shows that smaller displacement tiers are more
  informative than the strict 30mm marker for professor push/tap reporting.
- For a 10cm cube-like object, 0.72kg is close to density-preserving relative to
  the existing 3cm/20g object (`0.020 / 0.030^3 = 740.7kg/m^3`; 10cm at the same
  density is about 0.741kg), so it is a coherent weighted-object diagnostic.
- `sim_scripts/cube3cm_push_diffik_probe.py` now accepts explicit size/mass/gate
  args and logs density, `disp/object_size`, and threshold displacement rates.
- Local static verification passed: `py_compile`, `--help`, and
  `git diff --check`.
- After explicit approval, the first local GPU tiny 128 gate for 10cm/0.72kg
  seed930 failed the 1cm gate: summary lines 11-25 report controlled `0.1875`,
  `disp_along_push_mean_m=-0.0001524518520454876`, and
  `disp_ge_gate_rate=0.0`; lines 41-52 report final TCP target error mean
  `0.15134897292591631m`, low-motion `1.0`, and min TCP-cube distance mean
  `0.13646007765782997m`.

Implication:

- Do not scale the failed v1 10cm/0.72kg DiffIK diagnostic.
- The next valid work is small geometry/control diagnosis for the 10cm object
  contact path and DiffIK clipping, followed by another tiny gate only after
  explicit user approval.
- A future tiny 128 pass would only justify the next small design step. It would
  not be dataset readiness and would not approve 1024/10k data collection or RL
  scale-up.

Sources:

- `claudedocs/session_20260604_cube10cm_diffik_teacher_gate_prep.md`
- `sim_scripts/cube3cm_push_diffik_probe.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930_summary.json:11-52`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930.csv`
- `claudedocs/session_20260602_cube3cm_push_metric_reframe_targetext.md`
- `claudedocs/session_20260604_cube3cm_hierarchical_bucket_audit.md`

## D119 - Code changes before professor-branch reruns require pre-edit code review and deterministic geometry gates

Date: 2026-06-04

Decision:

- Before editing or generating code for the professor push/tap branch, perform a
  short code review of the relevant reset, target-generation, action/clipping,
  logging, and metric paths.
- For the 10cm/0.72kg diagnostic, do not return directly to randomized 128/1024
  gates. First use an object-size-aware reset, fixed easy cube position, fixed
  push direction, and side-face-center TCP height.
- Keep all new diagnostic knobs default-preserving for the existing 3cm branch.
- Do not run another GPU/IsaacLab gate, dataset generation, PPO/RL scale-up, VLA
  training, or Track A runtime without explicit approval.

Evidence:

- The first 10cm tiny gate failed with `disp_ge_gate_rate=0.0`, but code review
  showed a more basic issue: reset z still used the 3cm-derived `CUBE_CENTER_Z`
  constant while the probe only changed spawn/init-state size. A 10cm cube needs
  center z `TABLE_Z + 0.050`, not the old `TABLE_Z + 0.015`.
- The failed gate also mixed random x/y, random direction, top-margin target
  height, short v1 horizon, and DiffIK clipping. That makes it too broad for
  isolating whether a single easy front-face push works.
- Static patch added object-size-aware reset fields, fixed position/direction
  options, `side_center` TCP height mode, and reset-height logging. Static checks
  passed: `py_compile`, `--help`, and `git diff --check`.

Implication:

- The next valid approved runtime is a small fixed-position/fixed-direction
  geometry gate, preferably 16 envs or less, not 128/1024 randomized robustness.
- A pass on the fixed case only unlocks gradual reintroduction of randomization;
  it does not unlock dataset generation or RL scale-up.

Sources:

- `claudedocs/session_20260604_cube10cm_diffik_teacher_gate_prep.md`
- `roarm_rl/roarm_cube_push_env.py`
- `sim_scripts/cube3cm_push_diffik_probe.py`

## D120 - 10cm DiffIK probes must target the settled PhysX object pose, not only the reset buffer

Date: 2026-06-04

Decision:

- For 10cm/0.72kg professor-branch DiffIK diagnostics, do not generate TCP
  targets only from the reset-time `_cube_start_w` buffer.
- After reset and settle, use the actual settled `inner._sponge_pos_w` as the
  diagnostic start pose for TCP target generation and displacement metrics.
- Keep the next runtime as the same fixed-position/fixed-direction 16-env gate,
  only after explicit GPU escalation approval. Do not jump to randomized 128,
  1024/10k data, dataset generation, PPO/RL scale-up, VLA training, or Track A.

Evidence:

- The approved fixed 16 side-center gate used 10cm/0.72kg, fixed cube
  `(x=0.300,y=0.000)`, fixed push direction `(1,0)`, and a 1cm gate, but failed:
  controlled `0.0`, `disp_ge_gate_rate=0.0`, 1/5/10/20/30mm all `0.0`,
  `diffik_clip_rate_mean=1.0`, final TCP target error mean
  `0.1307708825916052m`, and min TCP-cube distance mean
  `0.07594270585104823m`.
- The summary/CSV reset-buffer start z was `0.03788299858570099`, while trace
  lines 2-5 showed the actual settled cube center z was about `0.049999m`.
- The same trace line 2 showed target z was still `0.03788299858570099`, proving
  the side-center TCP target used the reset buffer, not the settled PhysX object
  center.
- Trace lines 278-281 showed the TCP stayed around x `3.249m` while the final
  target x was `3.359999895095825`, so reach/clipping remains unsolved even after
  randomization was removed.
- Static patch updated `sim_scripts/cube3cm_push_diffik_probe.py` to copy
  `inner._sponge_pos_w` into `cube_start_w` / `_cube_start_w` after settle and to
  log `cube_start_z_mean_m`; `py_compile`, `--help`, and `git diff --check`
  passed.

Implication:

- The fixed 16 failure is still a valid failure, but not yet a clean mass/friction
  conclusion. It exposed a target-generation/settled-pose issue plus persistent
  DiffIK clipping.
- The next approved runtime should repeat the fixed 16 gate with the settled-start
  patch. A pass only unlocks gradual diagnosis, not scale-up; a fail should lead
  to trace/video reachability and joint-limit analysis.

Sources:

- `claudedocs/session_20260604_cube10cm_diffik_teacher_gate_prep.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed16_sidecenter_seed931_summary.json:11-76`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed16_sidecenter_seed931.csv:1-4`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed16_sidecenter_seed931_trace.csv:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed16_sidecenter_seed931_trace.csv:277-281`
- `sim_scripts/cube3cm_push_diffik_probe.py`
- `roarm_rl/roarm_stack_env.py:133-136`

## D121 - Settled-start fixed 16 still fails; next work is DiffIK body/Jacobian and actuator-path diagnosis, not data scale-up

Date: 2026-06-04

Decision:

- The settled-start patch fixed the immediate target-z mismatch, but did not make
  the 10cm/0.72kg fixed side-center push work.
- Do not interpret the current failure as "the object is too heavy" or "1cm is too
  much." The object still barely moves because the TCP does not reach the target
  path.
- The next professor-branch step is code review of IsaacLab DiffIK body/Jacobian
  mapping, TCP offset usage, position-only target convergence, joint-step
  clipping, and actuator target tracking. The next runtime should be a tiny
  trace/video diagnostic only after approval, not a 128/1024 gate.

Evidence:

- The approved settled-start fixed 16 rerun logged actual settled start
  `cube_start_z_mean_m=0.04999994789250195`, while the requested reset-derived
  `cube_center_z_m` remained `0.037883`; trace lines 2-5 showed target z now
  matches settled cube z around `0.050m`.
- Despite the corrected start pose, the run failed: controlled `0.0`,
  `disp_ge_gate_rate=0.0`, 1/5/10/20/30mm rates all `0.0`,
  `diffik_clip_rate_mean=1.0`, final TCP target error mean
  `0.12522956123575568m`, and min TCP-cube distance mean
  `0.07620850298553705m`.
- Trace lines 278-281 showed final target x `3.3600244522094727`, but TCP x
  stayed around `3.2484-3.2491` and TCP z around `0.1068-0.1075m`; the object
  moved only about 0.008-0.012mm along push in traced envs.
- A local non-GPU DLS check with `sim_scripts/roarm_kinematics.py` solved the
  same nominal side-center precontact/through TCP targets from HOME within
  1.5mm, so the project kinematic model does not say those points are impossible.

Implication:

- Do not generate 10,240 DiffIK data from this path; it would be failure data.
- Do not start PPO/RL or VLA scale-up from this teacher path.
- The next useful diagnostic is to make the IsaacLab DiffIK trace explain why the
  controller/actuator path diverges from the local kinematic feasibility result.

Sources:

- `claudedocs/session_20260604_cube10cm_diffik_teacher_gate_prep.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed16_sidecenter_settledstart_seed931_summary.json:13-77`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed16_sidecenter_settledstart_seed931_trace.csv:1-6`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed16_sidecenter_settledstart_seed931_trace.csv:277-281`
- `sim_scripts/cube3cm_push_diffik_probe.py`
- `sim_scripts/roarm_kinematics.py`

## D122 - 10cm DiffIK failure is target-geometry plus actuator/step-control, not object impossibility

Date: 2026-06-04

Decision:

- Keep the 10cm/0.72kg professor branch separate from Track A. Do not run
  128/1024/10k, dataset generation, PPO/RL, VLA, or Track A from these results.
- The current evidence does not support "0.72kg is too heavy" or "1cm is too
  much." It supports a narrower diagnosis: the original TCP through target was
  a far-face/cross-object target for a 10cm cube, while the default actuator and
  0.035rad step path cannot even reach the safer near-face contact target.
- `--through_target_mode near_face` and actuator override flags are diagnostic
  switches only. A drive-boost pass is not teacher-ready because it overshoots
  the 1cm objective.

Evidence:

- Code review: `sim_scripts/cube3cm_push_diffik_probe.py` now exposes
  `--through_target_mode` and arm actuator overrides at lines 51 and 58-61,
  applies actuator overrides before `gym.make` at lines 244-251, and computes
  far-face vs near-face through targets at lines 471-475. The run summaries log
  through mode and actuator values at lines 880 and 910-927.
- The default arm actuator remains weak for this diagnostic: `roarm_rl/roarm_stack_env.py`
  lines 173-180 configure arm stiffness `80.0`, damping `4.0`, effort limit
  `2.5`, and velocity limit `3.14`.
- Baseline 4-env fixed side-center trace with default actuator, far-face target,
  and cap `0.035` still failed: summary lines 11 and 25-28 show controlled
  `0.0`, clip `1.0`, `disp_ge_gate_rate=0.0`; lines 43 and 65-68 show final
  TCP target error mean `0.125396907m`, low-motion `1.0`, and min TCP-cube
  distance `0.076345483m`. Diagnostic lines 97-100 classify
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`.
- Long approach alone did not fix the default path. With cap `0.035` and
  `approach_steps=700`, summary lines 19, 21, 25-28, 43, and 65-68 still show
  controlled `0.0`, gate `0.0`, clip `1.0`, final TCP error `0.124230925m`,
  low-motion `1.0`, and min TCP-cube distance `0.075801797m`.
- Cap/drive positive controls prove the object can move, but not in a usable
  teacher way. cap `0.120` with default actuator moved mean `0.089894325m`
  and passed the 1cm gate on 4/4 envs, but summary lines 25-28 and 43 show
  clip `1.0`, `disp/object_size=0.898943245`, and final TCP error
  `0.094675537m`. Drive boost with the original far-face target moved mean
  `0.067951232m` on 3/4 envs; summary lines 2-11 record the boosted actuator,
  and lines 21, 35-38, 53, and 75-77 show controlled `0.75`, clip `1.0`,
  gate `0.75`, final TCP error `0.101294972m`, and low-motion `0.25`.
- Near-face target geometry reduces target error but does not solve default
  actuator/step tracking. Default near-face summary lines 21, 35-37, 53, 75-79,
  and 95 show controlled `0.0`, clip `1.0`, gate `0.0`, final TCP error
  `0.061259050m`, low-motion `1.0`, min TCP-cube distance `0.082745695m`, and
  `through_target_mode=near_face`. Near-face long-approach summary lines 19,
  21, 35-37, 53, 75-79, and 95 still show gate `0.0`, final TCP error
  `0.061997696m`, and min TCP-cube distance `0.083884733m`.
- Near-face plus drive boost passed 1cm but overshot. Summary lines 2-11 record
  the boosted actuator; lines 21, 36-38, 53, 75-79, 91, and 95 show controlled
  `1.0`, mean displacement `0.050082028m`, gate `1.0`, final TCP error
  `0.124849608m`, low-motion `0.0`, min TCP-cube distance `0.073166957m`, and
  `through_target_mode=near_face`. Per-env rows 2-5 all moved about 5cm, not
  about 1cm.

Implication:

- The next code work should design a controlled near-face contact controller:
  approach/contact phases, smaller/contact-aware push target increments,
  actuator/step schedule, and stopping on actual displacement/contact. It should
  remain a tiny fixed-geometry diagnostic until it can push near 1cm without
  large TCP error or 5-9cm overshoot.
- Do not report any drive-boost run as teacher-ready data or as Track A progress.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:51,58-61,244-251,471-475,880,910-927`
- `roarm_rl/roarm_stack_env.py:173-180`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_seed932_summary.json:11-95`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_seed932_trace_diagnostic_summary.json:97-185`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_cap120_seed932_summary.json:11-95`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_cap120_seed932_trace_diagnostic_summary.json:97-185`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_longapproach_seed933_summary.json:19-95`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_longapproach_seed933_trace_diagnostic_summary.json:97-185`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_driveboost_seed934_summary.json:2-125`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_nearface_seed935_summary.json:2-126`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_nearface_longapproach_seed936_summary.json:2-126`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_nearface_driveboost_seed937_summary.json:2-126`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_trace4_sidecenter_nearface_driveboost_seed937.csv:1-5`

## D123 - Professor 10cm push/tap primary metric is reaction event, not final displacement

Date: 2026-06-04

Decision:

- Supersede the D118 "primary 1cm push gate" only for the clarified professor
  push/tap objective. If the task is tap/reaction, final 1cm displacement is a
  secondary relocation metric, not the primary success criterion.
- Primary evidence for this branch should be reaction event: measured contact,
  transient displacement, cube z/lift response, or cube speed. It is acceptable
  for the cube to react, lift slightly, and settle back near the start if the
  professor objective is "push/tap happened" rather than "relocate by 1cm."
- Keep this separate from Track A, dataset generation, PPO/RL scale-up, and VLA.
  A fixed-geometry reaction pass only justifies a tiny randomized reaction screen
  after the reaction gate is defined.
- Massive IsaacLab randomization is acceptable as candidate discovery, but one
  lucky success is not a policy, not teacher-data readiness, and not sim-to-real
  evidence. It must be followed by held-out seeds, perturbation/domain
  randomization, and trace/video audits.

Evidence:

- The probe now exposes reaction thresholds at
  `sim_scripts/cube3cm_push_diffik_probe.py:61-63`, validates them at
  `:247-252`, stores transient state at `:589-605`, computes `reaction_event` at
  `:891-897`, records max/reaction row fields at `:932-936`, and summarizes max
  transient displacement and reaction-event metrics at `:1137-1147`.
- seed938 showed the near slowdown was too conservative: summary lines 21-31
  show measured-stop mode but no contact/stop; lines 50-52 show only
  `1.30385160446167e-05m` final displacement and final gate `0.0`; lines 68-71
  show final TCP error `0.06104395352303982m`, first contact `-1`, and first
  stop `-1`.
- seed939 removed near slowdown and reached contact/stop: summary lines 21-31
  show `contact_stop_seen_rate=1.0`; lines 50-52 show final displacement only
  `0.0014313608407974243m` and final gate `0.0`; lines 68-71 show first contact
  mean `261.5` and first stop mean `279.75`; lines 94-100 show object speed and
  measured contact evidence.
- seed940 measured-stop freeze is the key clarified-metric result. Summary lines
  21-32 show measured-stop freeze with stop seen `1.0`; lines 51-53 show final
  displacement `0.0014494359493255615m` and final gate `0.0`; lines 95-100 show
  max cube speed `0.14879385754466057m/s`, max 8-15mm rate `1.0`, max transient
  displacement mean `0.010990217328071594m`, no contact overshoot, and transient
  1cm rate `1.0`; lines 101-106 show measured contact and near-contact rates
  `1.0`; line 118 keeps the old success marker `1.0`.
- seed940 per-env CSV lines 2-5 show all four fixed envs had measured contact,
  contact stop, no overshoot, final displacement about `1.38-1.56mm`, max
  transient displacement about `10.81-11.26mm`, and max speed about
  `0.138-0.164m/s`.
- The trace analyzer still reports controller-quality blockers for seed940:
  diagnostic summary lines 2-29 show clipping in some joints, lines 97-100 list
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`. So this is reaction evidence, not a polished
  teacher.
- NVIDIA's Isaac Lab documentation frames the workflow around many parallel
  environments and GPU-accelerated robot learning, and the Isaac Gym paper
  reports 2-3 orders of magnitude speedups from a GPU-native simulation/training
  path. That supports scale as a search/training tool.
- Domain randomization and ADR papers support simulation scale only when the
  randomized distribution is designed to bridge the reality gap; they do not
  justify treating a single selected lucky rollout as robust real-world evidence.
- If the per-trial true reaction probability is `p`, the chance of at least one
  hit in `N=1,000,000` trials is `1-(1-p)^N`. That is about 63% for `p=1e-6`,
  about 9.5% for `p=1e-7`, and about 1% for `p=1e-8`. A single hit in 1M is
  therefore weak evidence of a very narrow success manifold unless it repeats on
  held-out seeds and perturbations.

Implication:

- Stop calling the professor 10cm branch failed solely because final 1cm
  displacement is not maintained. Under the clarified push/tap criterion, seed940
  is a fixed-geometry reaction-event pass.
- Do not jump directly to RL. First make the reaction gate explicit, rerun only a
  tiny randomized reaction screen if approved, and separate "reaction happened"
  from "usable teacher/policy/dataset."
- If the professor asks specifically for sustained relocation, re-enable final
  displacement as the primary gate; otherwise it is a secondary diagnostic.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:61-63,247-252,589-605,891-897,932-936,1137-1147`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940.csv:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_trace_diagnostic_summary.json:2-100`
- NVIDIA Isaac Lab docs, "How Isaac Lab Accelerates Reinforcement Learning":
  https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-first-robot-with-isaac-lab/02-how-isaac-lab-accelerates-reinforcement-learning.html
- NVIDIA Isaac Lab developer page: https://developer.nvidia.com/isaac/lab
- Makoviychuk et al., "Isaac Gym: High Performance GPU-Based Physics Simulation
  For Robot Learning": https://arxiv.org/abs/2108.10470
- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from
  Simulation to the Real World": https://arxiv.org/abs/1703.06907
- OpenAI et al., "Solving Rubik's Cube with a Robot Hand":
  https://arxiv.org/abs/1910.07113
- Mania et al., "Simple random search provides a competitive approach to
  reinforcement learning": https://arxiv.org/abs/1803.07055

## D124 - Reaction gate requires contact evidence and stays separate from teacher quality

Date: 2026-06-04

Decision:

- For the professor 10cm/0.72kg cube push/tap branch, do not count speed-only
  object jitter as a reaction/tap PASS. A reaction gate must require reaction
  evidence plus contact evidence, no posewrite, and no overshoot.
- Keep three labels separate:
  `reaction_gate_pass`, `final_relocation_pass`, and `teacher_quality_ready`.
  A reaction pass is enough to say "tap/push reaction exists" but not enough to
  generate data or start PPO/RL.
- `teacher_quality_ready` remains false while final TCP error and DiffIK clipping
  are high, even if reaction gate passes.

Evidence:

- Added `sim_scripts/cube10cm_reaction_event_gate_audit.py`, a non-GPU posthoc
  reader. Lines 1-5 state it reads existing logs only and does not run IsaacLab,
  train, generate data, or touch the robot.
- Lines 73-83 expose reaction/contact/overshoot/teacher-quality thresholds.
  Lines 118-138 compute reaction, contact evidence, overshoot, and transient
  gate from each row. Lines 140-167 enforce no-posewrite/controller checks and
  split `reaction_gate_pass`, `final_relocation_pass`, and
  `teacher_quality_ready`. Lines 173-212 write JSON evidence; lines 220-238 print
  the three audit lines.
- seed938 is the negative control: audit JSON lines 2-3 show computed reaction
  `0.5` but contact evidence `0.0`; lines 19-20 show reaction rate `0.5` and
  `reaction_gate_pass=false`; lines 24 and 28 show final TCP error
  `0.06104395352303982m` and teacher not ready.
- seed939 passes reaction but not teacher quality: audit JSON lines 2-3 show
  reaction/contact `1.0`; lines 19-20 show `reaction_gate_pass=true`; lines 22,
  24, and 28 show DiffIK clip `1.0`, final TCP error `0.06364072300493717m`, and
  teacher not ready; line 42 shows transient 1cm gate `0.0`.
- seed940 is the stronger reaction pass: audit JSON lines 2-3 show
  reaction/contact `1.0`; lines 14-20 show max displacement
  `0.010990217328071594m`, speed `0.14879385754466057m/s`, no overshoot, and
  reaction pass; lines 21-28 show stop/contact `1.0`, final displacement gate
  `0.0`, final TCP error `0.059237909503281116m`, DiffIK clip `1.0`, and teacher
  not ready; line 42 shows transient 1cm gate `1.0`.

Implication:

- The next valid GPU/IsaacLab action, only after explicit approval, is a tiny
  randomized reaction screen using this audit. It should be judged as
  reaction-screen evidence only.
- Do not start dataset generation, PPO/RL scale-up, VLA, Track A, 1024/10k, or
  a million-rollout sweep from seed939/940.
- If a future run has high speed but no contact evidence, treat it like seed938:
  not a push/tap PASS without manual trace/video confirmation.

Sources:

- `sim_scripts/cube10cm_reaction_event_gate_audit.py:1-5,73-83,118-138,140-167,173-212,220-238`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_reaction_gate_audit.json:1-44`

## D125 - Do not chase the 10cm randomized reaction gate by cap-only escalation

Date: 2026-06-05

Decision:

- For the professor 10cm/0.72kg randomized reaction screen, do not treat
  increasing `--max_diffik_joint_step_rad` alone as the next fix. The cap050
  diagnostic reduced DiffIK clip rate a little, but contact evidence got worse
  and teacher quality stayed false.
- The current failure looks direction/geometry dependent. Analyze or test
  direction-conditioned contact/reachability before another cap or drive boost
  sweep.
- Keep this separate from Track A, dataset generation, PPO/RL, VLA, 1024/10k,
  and large random search.

Evidence:

- seed941 randomized 16-env reaction screen failed D124: audit JSON lines 2-3
  show reaction `1.0` but contact evidence `0.625`; lines 17-20 show no
  posewrite, no overshoot, and `reaction_gate_pass=false`; lines 21-28 show
  contact stop `0.4375`, DiffIK clip `0.9706730805337429`, final TCP error
  `0.063001801721839m`, measured contact `0.625`, and teacher not ready.
- seed941 direction buckets showed `x+` contact `1.0`, `y-` contact `1.0`,
  `y+` contact `0.375`, and `x-` contact `0.0` from the local CSV rows.
- seed942 changed only the main DiffIK cap from `0.035` to `0.050` under the
  same randomized 16-env reaction setup. Summary JSON lines 48, 69, 92-100 show
  clip `0.9432692341506481`, final TCP error `0.05797268496826291m`, max
  displacement mean `0.004881829023361206m`, max gate `0.1875`, and measured
  contact `0.5`.
- seed942 reaction audit JSON lines 2-3 and 17-20 show reaction `1.0`, contact
  evidence `0.5`, no posewrite, no overshoot, and `reaction_gate_pass=false`;
  lines 21-28 show contact stop `0.1875`, final TCP error `0.05797268496826291m`,
  clip `0.9432692341506481`, measured contact `0.5`, and
  `teacher_quality_ready=false`.
- seed942 trace diagnostic lines 97-100 still report
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`; lines 168-184 show the worst clipped/raw
  deltas remain joint 2.

Implication:

- cap050 is a failed diagnostic, not a recovery path.
- The next valid work is direction/geometry-specific contact diagnosis or a
  clearly scoped tiny screen after explicit approval, not another cap-only
  escalation and not RL/data/Track A.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941.csv:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_summary.json:48,69,92-100`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_trace_diagnostic_summary.json:97-100,168-184`

## D126 - The 10cm randomized reaction failure must be bucketed by push direction

Date: 2026-06-05

Decision:

- Do not treat the 10cm/0.72kg randomized reaction failure as a homogeneous
  randomization failure. Bucket by push direction before changing controller or
  claiming readiness.
- The `y+` direction is now a confirmed weak bucket under the current near-face
  measured-stop freeze controller. It should be diagnosed as reach/geometry and
  actuator tracking, not solved by final-displacement metric changes.
- Keep this branch separate from Track A, dataset generation, PPO/RL, VLA, and
  large-scale randomized search.

Evidence:

- seed941 local direction breakdown showed `x+` contact `1.0`, `y-` contact
  `1.0`, `y+` contact `0.375`, and `x-` contact `0.0`.
- seed943 fixed `--fixed_push_dir 0 1` used the original cap `0.035`, same
  16-env 10cm/0.72kg near-face measured-stop freeze screen, and no Track A/data/RL.
- seed943 summary JSON lines 75-78 confirm fixed push direction `[0.0, 1.0]`.
  Lines 48, 69, 95-103, 112, 119, 123, and 142 show clip `1.0`, final TCP error
  `0.07060193479992449m`, max speed `0.10947006440255791m/s`, max z delta
  `0.011328218039125204m`, max displacement `0.004163078963756561m`, max gate
  `0.3125`, measured contact `0.375`, no posewrite, reaction `0.9375`, rollout
  posewrite false, and 16 trials.
- seed943 reaction audit JSON lines 2-3 and 17-20 show reaction `0.9375`, contact
  evidence `0.375`, no posewrite, no overshoot, and `reaction_gate_pass=false`.
  Lines 21-28 show contact stop `0.3125`, clip `1.0`, final TCP error
  `0.07060193479992449m`, measured contact `0.375`, and
  `teacher_quality_ready=false`.
- seed943 trace diagnostic lines 97-100 still report
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`; line 118 shows mechanism instrumentation was OK.

Implication:

- The next useful work is y+ reach/geometry diagnosis: target path, lateral/height
  offsets, workspace pose, and actuator tracking. It is not a cap-only problem and
  not an invitation to RL/data scale-up.
- Any future tiny GPU screen should be direction-bucketed and explicitly scoped.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941.csv:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_summary.json:48,69,75-78,95-103,112,119,123,142`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_trace_diagnostic_summary.json:97-100,118`

## D127 - Y+ next work is geometry/reach before data or RL

Date: 2026-06-05

Decision:

- For the professor 10cm/0.72kg y+ bucket, do not interpret reaction-like
  speed/z/tip motion as enough to proceed to dataset generation or RL. The
  no-contact rows have almost no displacement along the commanded push.
- Treat the next work as local y+ target-path and reach diagnosis: side-center
  height, lateral offset, workspace x/y pose, and actuator target tracking. This
  should happen before another candidate controller sweep, dataset generation,
  PPO/RL, VLA, Track A, 1024/10k, or broad randomized search.
- Any future GPU test should be tiny, direction-bucketed, and explicitly scoped
  to a single geometry/control hypothesis.

Evidence:

- Added `sim_scripts/cube10cm_yplus_geometry_reach_audit.py`, a local posthoc
  reader only. Lines 1-4 state it does not run IsaacLab, train, generate a
  dataset, touch the robot, or reconnect a remote machine. Lines 141-180 build
  the contact/no-contact, workspace-bin, trace-env, and interpretation summary.
  Lines 185-218 write JSON and print a four-line console summary.
- The seed943 y+ geometry audit JSON lines 2-23 show the contact group has 6
  rows and mean max displacement `0.010986278454462687m`. Lines 31-56 show the
  no-contact group has 10 rows, mean max displacement only
  `0.00006915926933288574m`, and higher final TCP error
  `0.07492788583040237m`.
- Audit JSON lines 58-122 show workspace asymmetry: `cube_y0_m<=0` contact
  `0.625`, `cube_y0_m>0` contact `0.125`, `cube_x0_m<0.25` contact
  `0.1111111111111111`, and `cube_x0_m>=0.25` contact
  `0.7142857142857143`.
- Audit JSON lines 126-139 show traced contact and no-contact groups both retain
  large final TCP-target vertical error; no-contact is worse
  (`0.06137282773852348m` mean abs z error), while final TCP-cube distance is
  still about `0.083m`.
- Audit JSON lines 141-240 show env0/env3 no-contact traced cases and env1/env2
  contact traced cases. Env0 final line 310 and env3 final line 313 keep
  side-center final z errors `0.052595339715480804m` and
  `0.07015031576156616m`; env1/env2 also keep high z errors but reach measured
  contact/stop in better workspace poses.
- This is consistent with D126 and the trace diagnostic modes:
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`, not with a solved teacher/data path.

Implication:

- The immediate research target is why y+ side-center near-face targets do not
  reliably create measured contact: compare target path, actual TCP height,
  lateral/xy offset, and actuator follow error across contact vs no-contact
  rows. Do this locally first.
- Do not start 10cm 10240 data generation or RL from the current y+ evidence.

Sources:

- `sim_scripts/cube10cm_yplus_geometry_reach_audit.py:1-4,141-180,185-218`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_yplus_geometry_reach_audit.json:1-245`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_trace_diagnostic_summary.json:97-100,118`

## D128 - Y+ target path exists; height/reach and clipping dominate next hypothesis

Date: 2026-06-05

Decision:

- Do not treat the fixed-y+ failure as a missing y-target-advance bug. The traced
  target path advances in world y by about `0.020000m` and keeps the side-center
  target z at the start-cube height.
- The unresolved y+ problem is that the actual TCP remains several centimeters
  above the side-center target, while joint clipping and actuator follow lag are
  still present in both contact and no-contact traced groups.
- The next GPU experiment, if explicitly approved, should be a tiny
  direction-bucketed geometry/control hypothesis such as target height,
  lateral/xy workspace pose, or actuator tracking. Do not go to 10cm 10240,
  dataset generation, PPO/RL, VLA, Track A, or broad random search.

Evidence:

- Added `sim_scripts/cube10cm_yplus_trace_path_actuator_audit.py`, a local
  CSV/trace reader only. Lines 1-4 state it does not run IsaacLab, use GPU,
  train, generate data, touch the robot, or reconnect a remote machine.
- The probe target path code uses the object half-size and lateral offset: lines
  497-520 compute pre/through targets, with side-center z from cube z and
  near-face target `cube - push_dir * (half_along - push_through)`.
- The trace-path audit script lines 110-168 build per-env target/TCP/joint
  summaries, lines 172-202 split contact vs no-contact traced envs, and lines
  223-263 write the path/actuator summary and interpretation.
- Audit JSON lines 2-54 show traced contact envs `[1, 2]` and no-contact envs
  `[0, 3]`. Both groups have target world-y delta
  `0.019999980926513672m`, final target z near start-cube z, and `clip_any=1.0`.
  Contact final z error is `0.051741816103458405m`; no-contact final z error is
  worse at `0.06137282773852348m`.
- Audit JSON lines 10-16 and 36-42 show final TCP error is mostly vertical:
  z-error fraction is `0.8440865598225584` for contact traced envs and
  `0.858887503603252` for no-contact traced envs.
- Audit JSON lines 112-142 and 834-865 show the no-contact traced env0/env3
  final z errors `0.052595339715480804m` and `0.07015031576156616m`. Lines
  287-296 and 1010-1018 show worst follow/raw delta remains joint 2 in both
  no-contact traced envs.
- Audit JSON lines 1022-1038 summarize the interpretation: short lateral-neutral
  target path, side-center z near start-cube height, final TCP several
  centimeters above target, and clipping/follow lag in both groups.

Implication:

- The next local/GPU design question should be: can y+ measured contact be made
  reliable by changing target height, lateral offset, workspace bucket, or
  actuator tracking while preserving no-posewrite and no-overshoot?
- Do not spend compute on RL/data scale-up until the y+ teacher can consistently
  create contact evidence with acceptable TCP error/clipping.

Sources:

- `sim_scripts/cube10cm_yplus_trace_path_actuator_audit.py:1-4,110-168,172-202,223-263`
- `sim_scripts/cube3cm_push_diffik_probe.py:497-520`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_yplus_trace_path_actuator_audit.json:1-1043`

## D129 - Height-only y+ correction is not a teacher/data fix

Date: 2026-06-05

Decision:

- Do not treat a positive target-height offset by itself as the recovery path for
  the professor 10cm/0.72kg fixed-y+ push/tap teacher.
- The seed944 height050 run lowered final TCP-target error, but it eliminated
  measured contact and failed the reaction gate. Low final TCP error without
  contact evidence is not teacher/data readiness.
- The next useful axis is a tiny lateral/workspace/actuator-tracking hypothesis,
  only after explicit approval. Do not start 10cm 10240 data generation, PPO/RL,
  VLA, Track A, or broad random search from seed944.

Evidence:

- seed944 used fixed `--fixed_push_dir 0 1`, 10cm/0.72kg, near-face
  measured-stop freeze, original cap `0.035`, and the diagnostic
  `--tcp_center_height_offset_m 0.050`.
- seed944 summary JSON lines 21-35 show the measured-stop controller and
  IsaacLab DiffIK controller; lines 47-49 show no dataset generation and
  DiffIK clip `0.9491987340152264`; lines 69-72 show final TCP error
  `0.022889409447088838m` but no contact/stop step; lines 95-103 show max speed
  `0.06781863939249888m/s`, max z delta `0.004550501937046647m`, max
  displacement only `0.000058706849813461304m`, and measured contact `0.0`;
  lines 112, 119, 123, and 142 show no posewrite, reaction `0.6875`, rollout
  posewrite false, and 16 trials.
- seed944 reaction audit JSON lines 2-3 and 17-27 show reaction `0.6875`,
  contact evidence `0.0`, no posewrite, no overshoot, reaction gate false,
  final TCP error `0.022889409447088838m`, clip `0.9491987340152264`, measured
  contact `0.0`, and `teacher_quality_ready=false`; line 41 shows transient
  1cm gate `0.0`.
- seed944 trace diagnostic JSON lines 97-100 still report
  `JOINT_STEP_CLIPPING_DOMINANT` and `ACTUATOR_TARGET_TRACKING_LAG`; lines
  167-184 show worst clipped/follow/raw joints are still active.
- seed944 y+ trace-path audit JSON lines 20-35 show all traced envs are
  no-contact, final target z is about `0.050000098533928394m` above start cube z,
  final TCP-cube distance remains about `0.08266180194914341m`, and final error
  is still mostly z fraction `0.8597798536432198`; lines 1027-1035 show target
  world-y still advances about `0.02000001072883606m`.

Implication:

- The +5cm diagnostic rejected the "raise target height and proceed" shortcut.
  It may help the controller track a target, but it does not create cube contact.
- The next small experiment, if approved, should avoid another height-only move
  and instead isolate fixed workspace x/y, small lateral offset, or actuator
  tracking while keeping reaction/contact/no-overshoot gates.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_summary.json:21-35,47-49,69-72,95-103,112,119,123,142`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_reaction_gate_audit.json:1-42`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_trace_diagnostic_summary.json:97-100,167-184`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_yplus_trace_path_actuator_audit.json:20-35,1027-1035`

## D130 - Good y+ workspace restores reaction/contact but not teacher quality

Date: 2026-06-05

Decision:

- Treat fixed y+ workspace as a real contact discriminator, not just seed943
  random noise. A good bucket near `cube_x=0.295m`, `cube_y=-0.044m` restores
  measured contact and reaction under the current near-face measured-stop freeze
  controller.
- Do not mistake this for 10cm teacher/data/RL readiness. The run still has high
  final TCP error and DiffIK clipping, and final 1cm relocation remains false.
- The next useful work is either to map/generalize the workspace window or to fix
  actuator tracking in this now-contacting bucket. Do not start 10cm 10240 data
  generation, PPO/RL, VLA, Track A, or broad random search from seed945.

Evidence:

- seed945 fixed `--fixed_push_dir 0 1`, `--fixed_cube_x_m 0.295`,
  `--fixed_cube_y_m -0.044`, 10cm/0.72kg, near-face measured-stop freeze,
  original cap `0.035`, and height offset `0.000`.
- The probe code already supports this controlled workspace test: parser lines
  43-45 define fixed cube x/y and fixed push dir; lines 268-276 apply them to the
  env config; lines 497-520 compute the near-face side-center target from cube
  pose and object half-size; lines 657-686 apply measured-stop, step scales, and
  DiffIK clipping.
- seed945 summary JSON lines 47-53 show no dataset generation, clip `1.0`, final
  displacement `0.009423360228538513m`, and final 1cm gate `0.0`; lines 69-77
  show final TCP error `0.0655147316865623m`, fixed cube x/y, and fixed y+; lines
  95-103 show max speed `0.13854316715151072m/s`, max z delta
  `0.015773175051435828m`, max displacement `0.009829461574554443m`,
  transient gate `0.1875`, and measured contact `1.0`; lines 112, 119, 123, and
  142 show no posewrite, reaction `1.0`, rollout posewrite false, and 16 trials.
- seed945 reaction audit JSON lines 2-3 and 17-28 show reaction `1.0`, contact
  evidence `1.0`, no posewrite, no overshoot, reaction gate true, final TCP error
  `0.0655147316865623m`, clip `1.0`, measured contact `1.0`, and
  `teacher_quality_ready=false`; line 42 shows transient 1cm gate only `0.1875`.
- seed945 y+ geometry audit JSON lines 2-33 show all 16 rows are in the contact
  group, with mean cube position `x=0.2950093150138855`,
  `y=-0.044006768614053726`, mean max displacement
  `0.009829461574554443m`, and final TCP error `0.0655147316865623m`; lines
  57-80 show `cube_y0_m<=0` contact `1.0`; lines 123-136 show traced contact
  `n=4` and no-contact `n=0`.
- seed945 trace-path audit JSON lines 2-34 show traced contact envs `[0,1,2,3]`,
  target world-y delta `0.02000001072883606m`, final z-error fraction
  `0.8575224903003924`, and `clip_any=1.0`; lines 1027-1035 show the target path
  still advances in world y and keeps side-center z.
- seed945 trace diagnostic JSON lines 97-100 still report
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`; lines 168-184 show worst clipped/follow/raw
  deltas remain active.

Implication:

- The 10cm y+ failure is now narrowed: workspace placement can recover
  contact/reaction, but the teacher still needs tracking/relocation cleanup
  before any 10240/data/RL step.
- A sensible next tiny test is either a minimal workspace boundary check around
  this good x/y point, or an actuator-tracking change inside this good bucket.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:43-45,268-276,497-520,657-686`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_summary.json:47-53,69-77,95-103,112,119,123,142`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_reaction_gate_audit.json:1-43`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_yplus_geometry_reach_audit.json:1-136`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_yplus_trace_path_actuator_audit.json:1-34,1027-1035`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_trace_diagnostic_summary.json:97-100,168-184`

## D131 - Lateral -2cm in the good y+ workspace reaches 1cm but still is not data-ready

Date: 2026-06-05

Decision:

- Add and preserve the default-zero `--base_lateral_offset_m` knob so the base
  10cm y+ path can test lateral alignment directly, not only posx variants.
- In the good y+ workspace (`cube_x=0.295m`, `cube_y=-0.044m`), a lateral offset
  of `-0.020m` is the strongest local 10cm y+ candidate so far: it passes
  reaction, contact, final 1cm gate, transient 1cm gate, no-posewrite, and
  no-overshoot.
- Do not call seed946 teacher/data/RL ready. Final TCP error and DiffIK clipping
  remain high, and trace diagnostics still report target not reached, clipping,
  and actuator lag.

Evidence:

- Code added a default-preserving `--base_lateral_offset_m` parser argument at
  line 51, prints it at line 378, applies it to the base trajectory lateral
  tensor at line 439, and records it in summary JSON at line 1104.
- Direction reasoning from seed945 trace showed final TCP was consistently offset
  to +x relative to the target. For y+ pushes, `lateral_dir=(-1,0)`, so a
  negative lateral offset moves the target in +x. seed946 therefore tested
  `--base_lateral_offset_m -0.020`.
- seed946 summary JSON lines 48-55 show no dataset generation, clip `1.0`,
  final displacement `0.011250250041484833m`, final gate `1.0`, and normalized
  displacement `0.11250250041484833`; lines 70-80 show final TCP error
  `0.06282096705399454m`, fixed good workspace x/y, and fixed y+; lines 96-105
  show max speed `0.13885411759838462m/s`, max z delta
  `0.016476489370688796m`, max displacement `0.011251196265220642m`, max gate
  `1.0`, and measured contact `1.0`; lines 113, 120, 124, and 143 show no
  posewrite, reaction `1.0`, rollout posewrite false, and 16 trials.
- seed946 reaction audit JSON lines 2-8 and 17-28 show reaction `1.0`, contact
  evidence `1.0`, final relocation pass true, no posewrite, no overshoot,
  reaction gate true, final TCP error `0.06282096705399454m`, clip `1.0`,
  measured contact `1.0`, and `teacher_quality_ready=false`; line 42 shows
  transient gate `1.0`.
- seed946 y+ geometry audit JSON lines 2-33 show all 16 rows in the contact
  group, max displacement `0.011251196265220642m`, final TCP error
  `0.06282096705399454m`, and no no-contact rows; lines 123-128 show traced
  contact `n=4`, final xy error `0.021463414385781712m`, and final z error
  `0.05878029018640518m`.
- seed946 trace-path audit JSON lines 2-34 show all traced envs `[0,1,2,3]`
  contact, target world-y delta `0.01827782392501831m`, final z-error fraction
  `0.9393462154426221`, and clip_any `1.0`; lines 1027-1035 show final target x
  moved to about `+0.01934826374053955m` relative to the cube.
- seed946 trace diagnostic JSON lines 97-100 still show
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`; lines 168-184 show joint 2 remains the worst
  clipped/follow/raw-delta joint.

Implication:

- The y+ issue is no longer "cannot push a 10cm cube" in the good workspace. It
  can create controlled reaction and 1cm displacement with lateral alignment.
- The remaining blocker for dataset/RL is teacher quality and robustness:
  clipping, link target tracking, and whether the lateral/workspace candidate
  survives beyond this one fixed 16-env screen.
- The next valid tiny step is either an actuator/IK tracking cleanup inside this
  seed946 candidate or a minimal robustness check of the same candidate; it is
  not 10cm 10240, dataset generation, PPO/RL, VLA, Track A, or broad search.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:51,378,439,1104`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_summary.json:48-55,70-80,96-105,113,120,124,143`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_reaction_gate_audit.json:1-43`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_yplus_geometry_reach_audit.json:1-33,123-128`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_yplus_trace_path_actuator_audit.json:1-34,1027-1035`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace_diagnostic_summary.json:97-100,168-184`
