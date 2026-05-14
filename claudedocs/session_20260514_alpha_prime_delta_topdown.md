# Session 2026-05-14 — (alpha') Skill 3 Basin Sweep + (delta) Top-down Chain

## Context

From 5/13 chain run #7: force-set robot_dof_targets fix made Skill 0 work (200→21 step,
max_err 1.56°, tcp_err 10.1mm), but Skill 1 descent failed (sponge side-collision, sponge
shifted +12mm z / -44mm y). User decision (HARD RULE #18): proceed with (alpha') Skill 3
basin sweep, then (delta) top-down redesign. Step-by-step, with cross-verification
and skepticism.

## (alpha') Results — Skill 3 Basin of Attraction Sweep

### Method
- Skip scripted Skill 0/1/2 entirely. Force-set env at (P6v14a training entry + perturbation).
- Reset → `_grasped=True, _was_grasped=True` latch → robot at IK(target + (dx, 0, +0.05+dz))
  → sponge force-set at TCP local → settling null env.step → P6v14a actor inference 200 step.
- 6-point grid: dx in {0, +15, +30, +45}mm, dz in {0, +20}mm.
- Implementation: [roarm_rl/chain_skills.py](../roarm_rl/chain_skills.py) `run_basin_sweep_isaac()` + `_run_single_basin()` + `_force_set_p6v14a_entry()`.

### Verdict table

| dx_mm | dz_mm | verdict           | success_step | release_step | final_d_xy | min_d_xy |
|-------|-------|-------------------|--------------|--------------|------------|----------|
| 0     | 0     | SUCCESS           | 15           | 13           | 227.6mm    | 5.2mm    |
| +15   | 0     | RELEASE_NO_LAND   | -            | **197**      | 147.5mm    | 16.5mm   |
| +30   | 0     | SUCCESS           | 28           | 13           | 178.5mm    | 14.6mm   |
| +45   | 0     | RELEASE_NO_LAND   | -            | 13           | 175.7mm    | **1.7mm**|
| 0     | +20   | SUCCESS           | 16           | 13           | 188.2mm    | 5.9mm    |
| +30   | +20   | SUCCESS           | 43           | 13           | 187.0mm    | 14.6mm   |

### Critical analysis

1. **Release timing portable 5/6** — at dx=0/30/45/dz=20, gripper opens at step 13 (consistent,
   memorization-free generalization). Only (15, 0) is outlier (release_step=197 — actor stalls).
   Non-monotonic basin (45mm works, 15mm doesn't) suggests brittle non-linear policy regions.
2. **All 6 runs final_d_xy=147-227mm** — single-step "SUCCESS" verdict in my code is misleading.
   `min_d_xy < 30mm` in 5/6 runs (sponge PASSES THROUGH target zone), but bounces away after.
3. **Root cause of bounce**: P6v14a continues issuing actions after release (200-step horizon).
   Robot post-release motion KNOCKS the falling sponge. The training success criterion
   (`_place_counter ≥ 50`) requires sponge to STAY near target — env penalizes post-release
   disturbance during training, but the trained policy still disturbs in our chain context.

### Basin conclusions for chain design

- **P6v14a release primitive IS portable** within at least 30-45mm xy / +20mm z offset.
- Skill 0/1/2 don't need extreme precision — 30mm xy tolerance OK.
- BUT need **early-terminate Skill 3** after release + small buffer to avoid post-release
  sponge knock-away.

## (delta) Top-down Chain Implementation

### Design
- Add q_high waypoint: TCP at +0.150m world (+115mm above sponge top +35mm) → vertical clearance.
- IK warm-start chain (HOME → q_high → q_hover → q_grasp) keeps wrist_p posture consistent.
- Skill 0 = HOME → q_high (instead of HOME → q_hover).
- Skill 1a = q_high → q_hover (descent stage 1, normal tol 0.03 rad).
- Skill 1b = q_hover → q_grasp (descent stage 2, **TIGHT tol 0.005 rad**, max 200 step).
- Skill 1c = close gripper (q → 45.84°).
- Skill 2 = grasp → transport_hover (unchanged).
- Skill 3 = P6v14a inference, **early-terminate at release_step + 15 buffer** (alpha' finding).
- Skill 4 = retreat to TCP +150mm above place (avoid sponge knock-away after release).

### Local dry-run sanity
- wrist_p range high/hover/grasp = **3.59° (OK < 5°)** — IK posture consistency verified.
- Final TCP residual 0.46mm — IK accurate.

### B200 chain run result (sponge_xy=(0.25, -0.04), 5/14 12:50 KST)

```
Skill 0: 21 step, max_err 1.38°, tcp_err 8.2mm                     ✓ FAST CONVERGE
Skill 1a: 14 step, max_err 1.71°                                   ✓ FAST CONVERGE
Skill 1b: 200 step (max), max_err 4.65°, tcp_err_pre_close=18.88mm ❌ STALLED
Skill 1c: 80 step, gripper_q=45.84°, d_sponge_tcp=32.7mm, grasped=False ❌ GRASP FAIL
  tcp_after1=(+250.7, -40.1, +51.9)mm  sponge_after1=(+266.0, -34.5, +23.5)mm
Skill 2: 6 step, grasped=False (carrying nothing)
Skill 3: SUCCESS step 1 (false-positive, sponge happened to be near target after collision push)
         release detected step 15, terminated step 30 (+15 buffer)
         post-Skill3 d_xy=53.8mm, d_z=0.4mm  CHAIN_SETTLED=NO
Skill 4: retreat done, final_d_xy=53.7mm, CHAIN_FINAL_SUCCESS=NO
```

### Critical analysis vs 5/13 baseline

| Metric | 5/13 baseline | 5/14 (delta) | Change |
|--------|---------------|--------------|--------|
| Skill 1 descent z stall | tcp_after1 z=+51.8mm | tcp_after1 z=+51.9mm | **SAME** |
| Skill 1 max joint err   | 4.14°                | 4.65°                | similar |
| Sponge lateral knock    | -44mm Y              | +16mm X / -6mm Y     | **7x improvement** |
| Sponge vertical lift    | +12mm                | +12mm                | SAME |
| Grasp success           | False                | False                | SAME |
| Chain final success     | N/A (early fail)     | NO                   | partial |

### Critical conclusion

- **Top-down approach REDUCED lateral knock 7×** (44mm → 6mm Y).
- **BUT descent STALL z=+51.9mm is IDENTICAL to 5/13** — fundamental physics issue,
  independent of approach direction.
- **Cross-verify hypothesis**: 19mm z gap (target +33mm) unbridgeable because gripper
  fingers physically contact sponge during descent. PD steady-state error 4° = motor torque
  balance against sponge reaction force.
- **xy precision excellent** (0.7mm error) — top-down geometry works for lateral alignment.
- **Only the final 19mm vertical penetration into sponge zone fails**.

## Hypothesis for next iteration (delta.2)

**GRIPPER_OPEN_DEG = -10°** (max open per JOINT_LIMITS_DEG ["gripper"]: (-10, +100)).

Current GRIPPER_OPEN_DEG = 0 (slightly open within env's "open" range q < 22.9°). Jaw width
at q=0 unknown but likely < 22mm sponge width (per tech_gripper_grasp_anchors.md: cmd 5° = 3mm jaw).

If q=-10° → jaw width > 22mm + margin, fingers should clear sponge sides during descent.
Robot can then descend the last 19mm freely.

Risk: q=-10° might be physically infeasible (joint hit hard stop), or jaw width still
< 22mm. Cheap to test (~6 min).

## Options for user decision

| Option | Description | ETA | Confidence |
|--------|-------------|-----|------------|
| **(δ.2) Wider gripper open** | GRIPPER_OPEN_DEG: 0 → -10°. Re-run chain. | 6 min | MEDIUM (untested jaw geometry) |
| (δ.3) Multi-stage descent | Ramp Skill 1b target gradually (not force-set FINAL) | ~half day | HIGH (PD-friendly) |
| (γ) PPO Skill 1 training | Train descent+grasp as PPO primitive, replace scripted Skill 1 | 1-2 weeks | HIGH (paper-quality) |
| (β) Sim physics tuning | Adjust sponge friction/restitution/mass | 1-2 days | LOW (geometry, not physics) |

**Recommendation**: (δ.2) first (cheap test). If fails → (δ.3). (γ) last resort.

## (delta.2) Result — gripper -10° BLOCKED by URDF clamp (2026-05-14 PM)

**Edit**: `roarm_rl/chain_skills.py` L69 `GRIPPER_OPEN_DEG = 0.0` → `-10.0`.
md5: `2773d0ddef624f3a80f1fa3df992ad49` → `6d7c06623b12c9ad005174c3851164c5`.
Synced to B200 + `launch_chain_topdown.sh` EXPECTED_CHAIN_MD5 updated.

### B200 chain run (sponge_xy = (0.25, -0.04))

| Skill | Result | Note |
|-------|--------|------|
| 0 (HOME→high) | 21 step, max_err 1.38°, tcp_err 8.2mm | force-set OK |
| 1a (high→hover) | 14 step, max_err 1.71° | OK |
| **1b (hover→grasp)** | **200 step (max), max_err 10.00°, tcp_err_pre_close 18.88mm** | gripper q stuck `-0.0°` |
| 1c (close) | 80 step, max_err 4.76°, gripper_q 45.84°, d_sponge_tcp 33.2mm, **grasped=False** | sponge not in jaw |
| 2 (transport) | 6 step (sponge already gone) | downstream fail |
| 3 (P6v14a) | SUCCESS step 1 (false positive — d_xy 16.6, d_z 12.1), release at s=15, terminate s=30 | OOD sponge displacement |
| 4 (retreat) | 150 step, final_d_xy=49.7mm, **CHAIN_FINAL_SUCCESS=NO** | — |

### Critical evidence — URDF gripper clamp confirmed

- Trace at Skill 1b s=120, s=160 (frozen state): `q_deg=[-9.1,+22.0,+121.7,+13.8,+0.0,-0.0]` — gripper actual `-0.0°`, **target -10° unreached**.
- `max_err_deg = 10.00` = exactly `|target_-10 - actual_0|`, dominated by gripper joint.
- URDF inspection: `assets/roarm_m3/urdf/roarm_m3.urdf` defines `<joint name="link5_to_gripper_link" ...><limit lower="0" upper="1.571" effort="2.5" velocity="3.14"/></joint>`. Lower=0 hard-coded.
- Env (`roarm_stack_env.py` ~L408-412): `torch.clamp(targets, robot_dof_lower_limits, robot_dof_upper_limits)` enforces URDF lower=0 every step.
- → `GRIPPER_OPEN_DEG = -10.0` action target is clamped to 0.0 every physics step. Test of "wider jaw" hypothesis is impossible without URDF mod.

### Comparison (δ.1 vs δ.2) — bit-identical physics

| Metric | (δ.1) target=0° | (δ.2) target=-10° |
|---|---|---|
| gripper actual | 0° | 0° (clamped) |
| tcp_after1 z | +51.9mm | +51.9mm |
| tcp_err_pre_close | 18.88mm | 18.88mm |
| Skill 1b descent steady-state | stuck at +51.9mm | stuck at +51.9mm |

→ Descent stall is **gripper-state independent**. Jaw-width hypothesis untestable but the 19mm vertical gap is NOT explained by jaw geometry alone (since gripper actual was identical in both runs).

### Decisions captured

- **D006** (DECISIONS.md): URDF gripper lower=0 clamps open targets. Do not modify URDF; P6v14a retrain risk. Future scripted skills: gripper_open_target = 0.0, not negative.
- **D007** (DECISIONS.md): Already known force-set pattern formalized; force-set required for all deterministic scripted skills.

### Next options (Skill 1b stall fix, gripper-independent)

| Option | Description | ETA | Confidence |
|--------|-------------|-----|------------|
| **(δ.4) NEW** | Skill 1b multi-stage z: +63 → +50 → +40 → +33mm | ~2 hr | MEDIUM (PD-friendly) |
| (δ.3) FULL | Multi-stage target ramping (`current + delta` instead of one-shot q_grasp) | ~half day | HIGH |
| (γ) PPO Skill 1 | Train descend+grasp as PPO primitive | 1-2 weeks | paper-quality |
| (β) Sim physics | effort_limit↑, friction tuning | 1-2 days | LOW (geometry-blind) |

**Recommendation**: (δ.4) NEW first (cheap, same PD-negotiation mechanism as (δ.3)). FAIL → (δ.3) FULL.

## HARD RULE compliance

- **#4** External citation: none added this session.
- **#8** Archive: 5/13 late evening L83 → MEMORY_archive_20260517.md (6→5 entries ✓).
- **#11** /half-clone: not invoked.
- **#14** fail-fast guard + no `2>&1` + no pipe-to-source: all ssh commands compliant.
- **#15** cu128 sm_100 alive: B200 isaacsim_5_1 env used.
- **#17** state-only RL 28-dim narrow inline: maintained.
- **#18** User explicit: "α' first then δ, step-by-step, 비판적/분석적/의심" — followed.
- **#19** edge-stand 47mm: maintained.
- **#26** B200 physics-only RL: state-only via Annotator/RTX bypass — maintained.

## Inventory

### Local (Lenovo)
- [roarm_rl/chain_skills.py](../roarm_rl/chain_skills.py) — 696 LOC (was 489 5/13). md5 `2773d0ddef624f3a80f1fa3df992ad49`.
  - Added: `_force_set_p6v14a_entry()`, `_run_single_basin()`, `run_basin_sweep_isaac()`
  - Added: `q_high_deg` waypoint, `target_q_skill0_high/1a_to_hover/1b_to_grasp/1c_close` methods
  - Modified: Skill 0/1/2/3 in `run_chain_isaac()`, added Skill 4 retreat, Skill 3 early-terminate
- [launch_basin_sweep.sh](../launch_basin_sweep.sh) — md5 `a361aa673f82a663be8cd39ff4d0dee6`
- [launch_chain_topdown.sh](../launch_chain_topdown.sh) — md5 `4bc965f98f128f78f80f13a577ab986c`

### B200 (sync verified)
- `code/roarm_rl/chain_skills.py` md5 match
- `launch_basin_sweep.sh` md5 match
- `launch_chain_topdown.sh` md5 match
- `logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt` (Skill 3 source, unchanged)
- `/tmp/basin_sweep_20260514.{out,err}` (basin sweep log)
- `/tmp/chain_topdown_20260514.{out,err}` (top-down chain log)

## Next session immediate

(δ.2) test was BLOCKED by URDF clamp (see "(delta.2) Result" section). New decision required.

Active recommendation: **(δ.4) NEW — Skill 1b multi-stage z-target (+63 → +50 → +40 → +33mm)**, ~2hr, MEDIUM confidence.

Fallback: (δ.3) full multi-stage target ramping if (δ.4) fails.

Code change for (δ.4):
- `roarm_rl/chain_skills.py` Part 2 `run_chain_isaac()` Skill 1b: split single force-set into ≥3 intermediate z waypoints between TCP +63mm and +33mm.
- Revert `GRIPPER_OPEN_DEG` from `-10.0` back to `0.0` (D006: negative target = noise, no effect).
- Update `EXPECTED_CHAIN_MD5` in `launch_chain_topdown.sh` after edit.

Test plan:
- B200 chain run with same sponge_xy=(0.25, -0.04) for apples-to-apples vs (δ), (δ.2).
- PASS criteria: Skill 1b tcp_err_pre_close < 5mm AND grasped=True AND CHAIN_FINAL_SUCCESS=YES.
- FAIL → pivot to (δ.3) full ramping or geometry investigation (Skill 1b s=120/160 frozen-state suggests pure PD stall, not collision).
