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

---

## APPENDIX — (δ.2) doc errata (correction 2026-05-14 PM, append-only)

Lines 138-139 of the (δ.2) result table report:
- "0 (HOME→high) | 21 step, max_err 1.38°, tcp_err 8.2mm | force-set OK"
- "1a (high→hover) | 14 step, max_err 1.71° | OK"

**Both rows are INCORRECT** — those numbers are from (δ.1) and were copied into the
(δ.2) table by mistake. The user's direct B200 read of `/tmp/chain_topdown_v2.out`
on 2026-05-14 PM confirmed actual (δ.2) Skill 0/1a both hit `max_steps` (200 / 150)
with `max_full_err_deg ≈ 10.00` dominated by the unattainable gripper target
(`GRIPPER_OPEN_DEG = -10°` clamped to actual `0°` by URDF; `run_skill_closed_loop`
break condition `np.max(np.abs(err))` includes gripper joint per
`roarm_rl/chain_skills.py` L405-406).

Therefore (δ.2) NOT cosmetic-only: the negative gripper target inflated
`max_full_err_deg` and forced all scripted skills to `max_steps`. Physics result
identical to (δ.1) but runtime/convergence diagnostics polluted. D006 implication
strengthened: future scripted skills MUST use `GRIPPER_OPEN_DEG = 0.0` AND
`run_skill_closed_loop(..., exclude_gripper=True)`.

Corrected rows for (δ.2) table:
- 0 (HOME→high)  | **200 step (max)**, max_full_err ≈ 10° (gripper-inflated)
- 1a (high→hover) | **150 step (max)**, max_full_err ≈ 10° (gripper-inflated)
- 1b/1c/2/3/4 rows in original table remain accurate.

Do not edit lines 138-139 in place (append-only project rule); this errata block
is the authoritative correction.

---

## (δ.4) Result — Skill 1b multi-stage z descent diagnostic (2026-05-14 PM)

### Edits applied

1. `chain_skills.py` L69 `GRIPPER_OPEN_DEG`: `-10.0 → 0.0` (D006 + max_err noise prevention).
2. Planner cached intermediate IK waypoints: `q_1b1_deg` (TCP +50mm), `q_1b2_deg` (TCP +40mm), warm-started from `q_hover`/`q_1b1` respectively. `q_grasp_deg` warm-started from `q_1b2_deg`.
3. `run_skill_closed_loop(..., exclude_gripper=False)`: option added. When True, `break_err = max(abs(err[:5]))` (joint 0-4 only); gripper q always logged separately. Returns dict `{steps, max_arm_err_rad, max_full_err_rad, gripper_q_rad, gripper_err_rad, final_q_rad}`.
4. Skill 0/1a/1b{1,2,3}/2/4 callers updated to `exclude_gripper=True`. Skill 1c kept `exclude_gripper=False` (gripper close IS the objective).
5. Per-substage diagnostic helper `_diag_log()`: actual TCP z, target z, arm_err, gripper_q, sponge xyz.
6. md5: `chain_skills.py = be689ea9a812f2d8d1470246559d207f` (998 LOC, was 696 LOC).
7. `launch_chain_topdown.sh` EXPECTED_CHAIN_MD5 updated, md5 `c81398f58865ec1c73d99e8d7b2d90ee`.
8. B200 sync verified (md5 match both files).

### B200 chain run (sponge_xy=(0.25, -0.04), `/tmp/chain_topdown_v3.{out,err}`)

| Skill / sub-stage | target z | steps | arm_err | gripper_q | **TCP_actual z** | xyz_err | comment |
|---|---|---|---|---|---|---|---|
| 0 (HOME→high)        | +150mm | **21**  | 1.38° | 0° | +155.79mm | 8.19mm | early break OK ✓ |
| 1a (high→hover)      | +63mm  | **14**  | 1.71° | 0° | +70.75mm  | 7.80mm | early break OK ✓ |
| **1b1 (hover→+50)**  | +50mm  | 200 max | 0.32° | 0° | **+51.79mm** | 1.98mm | nearly at target (1.8mm short) |
| **1b2 (+50→+40)**    | +40mm  | 200 max | 2.96° | 0° | **+51.72mm** | 11.74mm | **stall — identical depth to 1b1** |
| **1b3 (+40→+33)**    | +33mm  | 200 max | 4.90° | 0° | **+51.80mm** | 18.81mm | **stall — same depth, deeper target ignored** |
| 1c (close gripper)   | 45.84° | 80      | 4.89° | 45.84° | +51.8mm | — | `grasped=False`, d_sponge_tcp=33.4mm |
| 2 (transport)        | +63mm  | 6       | 1.70° | 45.84° | — | 10.7mm | downstream of grasp fail |
| 3 (P6v14a)           | —      | 33      | — | release@s=18 | — | — | early-terminate ok; CHAIN_SETTLED=NO |
| 4 (retreat)          | +150mm | 44      | 1.45° | 0° | — | — | CHAIN_FINAL_SUCCESS=NO, final_d_xy=100.7mm |

Skill 1b summary: total_steps=600, per_stage=(200,200,200), final_tcp_err_pre_close=18.81mm, `stall_signature=TRUE`.

### Critical diagnostic finding — stall location precisely identified

- **Physical barrier = TCP z ≈ +51.8mm**, INVARIANT under target depth (+50/+40/+33mm).
- Sub-stage 1b1 (+50mm target) reached +51.79mm (1.8mm short → arm_err 0.32° barely above tol 0.005 rad = 0.29°).
- Sub-stage 1b2/1b3 with deeper targets (+40/+33mm) saw arm_err GROW (0.32° → 2.96° → 4.90°), yet TCP z stayed at +51.8mm. PD torque ↑ with deeper target, but contact reaction force scales identically → equilibrium at same z.
- xy precision = 0.6-0.8mm (xyz_err minus z_err). top-down geometry succeeded; only z is blocked.
- Sponge xy invariant across 1b1/1b2/1b3 (+266, -34.5)mm — robot does not push sponge further; pure z-contact equilibrium.
- 5/13 baseline (+51.8mm) + (δ.1) (+51.9mm) + (δ.2) (+51.9mm) + (δ.4) (+51.8mm) all bit-identical → **invariant across sub-stage splitting / gripper target / approach geometry / 4 sessions**.

### Pre-experiment user prediction validated

User (HARD RULE #18): "지금 냄새는 'step size가 커서 실패'보다는 '마지막 접촉 장벽이 실제로 있음' 쪽이 더 강해. 실패하면 stage 개수 튜닝 계속하지 말고, 바로 geometry/contact deep-dive 또는 δ.3 full ramping으로 가야 해."

**Validated**: (δ.3) full ramping is ALSO predicted useless by the +51.8mm-invariant result. PD step-increment ramping would converge to the same contact equilibrium because the limiter is reaction force, not target overshoot.

### D006/D007 + exclude_gripper effect — verified

- (δ.2) ran Skill 0 → 200 step max, Skill 1a → 150 step max (gripper-inflated err).
- (δ.4) runs Skill 0 → 21 step early-break, Skill 1a → 14 step early-break.
- `exclude_gripper=True` confirmed restoring proper convergence on arm-only objective.
- wrist_p range across high/hover/1b1/1b2/grasp = **4.57°** (OK < 5°).

### Next-step options for (δ.5) deep-dive

stage tuning rejected by user pre-commitment. Cheap diagnostics first (~1.5h total):

| Candidate | Investigation | Cost | Info value |
|---|---|---|---|
| **5-D** | sponge spawn z + top z measured from sim `root_pos_w` (vs assumed +11.4 / +34.9mm) | 30min | HIGH — baseline for all others |
| **5-E** | finger_tip world FK at TCP z=+51.8mm (sim direct read), vs sponge top z | 1h | HIGH — primary collision evidence |
| **5-F** | gripper jaw separation (URDF FK) at `q_gripper=0`, compared to sponge 22mm width | 30min | HIGH-cheap |
| 5-A | sponge collision mesh + contact force sensor in sim (Isaac Sim contact viewer) | 1-2h | MEDIUM (after 5-D/E/F) |
| 5-B | effort_limit=2.5 Nm vs sponge reaction force; URDF mod evaluation | 2-3h | MEDIUM (gated on geometry result) |
| 5-C | sponge mass/friction/restitution sweep | 1-2h | MEDIUM (P6v14a-invariant params) |

Then based on outcome:
- finger_tip presses sponge TOP → redefine TCP_GRASP_Z OR change finger collision OR redesign grasp pose.
- finger_tip presses sponge SIDE + jaw width < 22mm → URDF gripper open limit change → P6v14a retrain risk → (γ) PPO Skill 1 primitive likely the cheaper path.
- effort_limit bottleneck → URDF effort increase (mild retrain risk).

Reserve: (γ) PPO Skill 1 descend+grasp primitive (1-2 weeks, paper-quality).

### Inventory delta (Local + B200 md5 verified)

- `roarm_rl/chain_skills.py` md5 `be689ea9a812f2d8d1470246559d207f` (998 LOC; was 696 LOC / `6d7c06623b12c9ad005174c3851164c5`)
- `launch_chain_topdown.sh` md5 `c81398f58865ec1c73d99e8d7b2d90ee` (was `a5208ac9c0530bc40c3bfff85c8118be`)
- B200 sync verified for both files; EXPECTED_CHAIN_MD5 in launcher matches chain_skills.py md5.
- B200 logs: `/tmp/chain_topdown_v3.out` (full chain stdout) + `/tmp/chain_topdown_v3.err` (Isaac Sim warnings only, no failure).

### HARD RULE compliance (delta.4)

- #4: no external citations introduced.
- #8: MEMORY archive pending — must move oldest of 5 entries to `MEMORY_archive_YYYYMMDD.md` when prepending the new 5/14 PM entry.
- #11: `/half-clone` declined when stop-hook fired at 85% context (this session's end-of-session update protocol used instead).
- #14: fail-fast guard + no `2>&1` + no pipe-to-source — all B200 ssh commands compliant.
- #15: cu128 sm_100 alive (B200 isaacsim_5_1 env).
- #17: state-only RL 28-dim narrow inline maintained (Skill 3 obs unchanged).
- #18: user-explicit "(a)=4-stage OK / (b)=exclude_gripper 같이 추가 / (c)=append-only 정정" all followed; user pre-commit "stage 튜닝 금지 후 geometry/contact deep-dive로" honored.
- #19: edge-stand 47mm sponge geometry unchanged.
- #26: B200 physics-only state-only RL (until 5/19) — maintained.

---

## (δ.5) Result — 5-D/5-E/5-F contact geometry diagnostic (2026-05-14 PM)

### Edits / run

- Added diagnostic logging to `roarm_rl/chain_skills.py`:
  - 5-D: sponge raw/cache root z, top z, TCP-vs-top gap, `_grasped/_was_grasped`.
  - 5-E: `gripper_link` collision mesh world bbox at Skill 1b stall.
  - 5-F proxy: `gripper_link.stl` local bbox. Note: URDF/USD has single
    `gripper_link` collision mesh, not separate left/right finger links, so true
    inner jaw gap is not directly represented by body names.
- Local checks passed: `python -m py_compile roarm_rl/chain_skills.py`,
  `python roarm_rl/chain_skills.py --dry-run`.
- md5: `chain_skills.py = 03169d005c4d39fa10583047e8957961`;
  `launch_chain_topdown.sh = 2d52c0efd2ca1d5bab78f3e029185a47`.
- B200 run: `/tmp/chain_topdown_v4.{out,err}`.

### Key evidence

| Diagnostic | Result |
|---|---|
| 5-D initial write | raw sponge root `(+250.0,-40.0,+11.4)mm`, expected center OK |
| 5-D after Skill 0/1a settling | sponge root `(+266.0,-34.5,+23.5)mm`; sponge top `+47.0mm` |
| Skill 1b1 target +50 | TCP `+51.79mm`; TCP minus sponge top `+4.79mm` |
| Skill 1b2 target +40 | TCP `+51.72mm`; TCP minus sponge top `+4.73mm` |
| Skill 1b3 target +33 | TCP `+51.80mm`; TCP minus sponge top `+4.81mm` |
| 5-E gripper mesh | `gripper_link` mesh min-z `+47.4mm`, max-z `+127.9~128.0mm` |
| 5-E top gap | mesh min-z minus sponge top `+0.4mm` at all 1b stages |
| 5-E xy overlap | AABB overlap about `60.0mm × 22.0mm` (directly over sponge width) |
| latch check | `_grasped=False`, `_was_grasped=False` through Skill 1b |

### Conclusion

The D008 `+51.8mm` equilibrium is now localized: the gripper_link collision
envelope bottoms out at the settled sponge top (`+47.4mm` vs `+47.0mm`) while the
TCP remains about `+4.8mm` above that top. The robot is not failing because of
waypoint step size, action interface, or grasp latch. It is a top-contact
geometry/collision bottleneck before grasp.

This matches the intended high-level strategy (top-down approach avoids side
knocking), but the current TCP/gripper geometry places the collision mesh on the
sponge top before the TCP reaches the planned `+33mm` grasp pose.

### Next recommendation

Do not generate four-sponge well/hash stacking demos yet. First fix the top-down
pick primitive:

1. G1 quick geometry patch: raise/redefine `TCP_GRASP_Z`, shift TCP relative to
   sponge center, or alter wrist/gripper approach so the collision mesh clears
   the sponge top while still closing around the 22mm width.
2. Re-run B200 chain with same diagnostics; PASS only if Skill 1b reaches grasp
   pose, `grasped=True`, and no side knock/top stall.
3. Then resume four-sponge workspace planning: white table, black robot, gray
   background, four pink edge-stand sponges arranged in the well/hash layout;
   pick each sponge from above and place/stack sequentially.

### Inventory delta

- `roarm_rl/chain_skills.py` md5 `03169d005c4d39fa10583047e8957961`.
- `launch_chain_topdown.sh` md5 `2d52c0efd2ca1d5bab78f3e029185a47`.
- B200 logs: `/tmp/chain_topdown_v4.out`, `/tmp/chain_topdown_v4.err`.
