# Phase 1 Step E — P1→P2→P3 curriculum + Phase 1.B design (COMPLETE)

**Date**: 2026-05-07 late-night (continuation from Step D)
**HARD RULE**: #26 (B200 physics-only Isaac Sim RL priority, 5/19 deadline)
**Status**: ✅ Phase 1.A curriculum complete. Best ckpt = **P3 model_1100 (99.68% success)**.

---

## TL;DR

| Phase | Best ckpt | Wall time | success_rate | sponge_h | Notes |
|---|---|---|---|---|---|
| **P1 reach** | model_499 | 3:19 (500 iter) | **97.75%** | 391 mm | emergent grasp via kinematic-attach side-effect |
| **P2 lift** (resume P1) | model_998 | 3:04 (500 iter) | **99.22%** | 644 mm | clean lift, +1.5pp over P1 |
| **P3 grasp+success** (resume P2) | **model_1100** | 6:39 (1000 iter trained, but **regressed after iter 1100**) | **99.68%** | 427 mm | 🏆 BEST overall |
| ↳ P3 final 1997 | — | — | 26.87% | 119 mm | ❌ collapsed (action noise 2.68→6.31) |

✅ HARD RULE #26 abort gates all cleared with 12 days of buffer remaining.
Total B200 compute: ~13 minutes (P1+P2+P3 train) + ~3 minutes (5 evals).

---

## Eval methodology — `roarm_rl/eval_policy.py` (NEW this session)

### Why a new script (vs reusing Isaac Lab `play.py`)?

Isaac Lab's `play.py` runs interactively — does not aggregate success rate. Custom `eval_policy.py` rolls out frozen policy on `num_envs` parallel envs, captures per-env stats AT episode-end (before reset clobbers `_success_flag`), aggregates over `num_rollouts` episodes/env.

### Two technical fixes inside `eval_policy.py`

1. **No `env.reset()` after `runner.load(ckpt)`** — Isaac Lab `joint_acc` becomes an inference tensor after `runner.load()` makes its first inference pass on policy load, so `_data.joint_acc[env_ids, joint_ids] = 0.0` inside `write_joint_velocity_to_sim` raises `RuntimeError: Inplace update to inference tensor outside InferenceMode`. Workaround: skip explicit reset; force a warmup truncation by setting `episode_length_buf = max_episode_length` before first `policy(obs)` step (which itself runs inside `with torch.inference_mode()`, so the in-step `_reset_idx` is safe).

2. **Hooked `_reset_idx` for capture** — replace `inner_env._reset_idx` with a wrapper that records `(_success_flag, dist, sponge_h, _grasped)` AT episode-end before delegating to `orig_reset()`. The first warmup reset is filtered out via `warmup_done` flag.

### Eval config used (consistent across P1/P2/P3)

```
--num_envs 256 --num_rollouts 4  → 1024 trials per ckpt
--reward_phase {1,2,3} (matches the trained policy)
--seed 42
```

Wall time per eval: ~30-35 s (1024 trials of 400 steps each, B200 ~250K steps/s).

### Output buffering trap (lesson)

Initial eval run produced an empty log because `python -m roarm_rl.eval_policy ... > log 2>&1` block-buffers stdout. **Fix**: always launch B200 Python with `python -u` (or `PYTHONUNBUFFERED=1`) when redirecting to file. Now fixed in standard ssh entry pattern.

---

## P3 regression — root cause + remediation

### What happened (training-time metric trajectory)

| iter | noise std | sponge_h (m) | rolling success | grasped_frac | ep_len |
|---:|---:|---:|---:|---:|---:|
| 998 (P2 final → P3 init) | 2.68 | 0.055 | 0.0000 | 0.027 | 19 |
| 1000 | 2.68 | 0.361 | 0.001 | 0.852 | 67 |
| 1050 | 2.87 | 0.254 | 0.011 | 0.615 | 92 |
| **1100** | **3.07** | **0.177** | 0.010 | 0.652 | 95 |
| 1200 | 3.43 | 0.079 | 0.007 | 0.517 | 146 |
| 1300 | 3.69 | 0.074 | 0.001 | 0.893 | 320 |
| 1500 | 4.68 | 0.081 | 0.0004 | 0.917 | 343 |
| 1997 (final) | 6.31 | 0.095 | 0.0004 | 0.925 | 360 |

(Note: these are "rolling per-step success" — biased low for P3 because successful episodes terminate immediately, leaving few `_success_flag=True` frames in the buffer. Eval-time success rates above are the unbiased numbers.)

### Diagnosis (two independent failure modes compounding)

1. **Local-minimum exploit**: in P3, `terminated = _success_flag` ends episodes on success. `success_bonus=10` (one-shot) is far less than the per-step continuous reward (`5 × lift + 2 × grasp` ≈ 4.5/step × 350 steps = 1575). Policy correctly learns NOT to lift past 100 mm to avoid early termination — instead, it grasps the sponge at table height and milks `grasp_bonus` for full 400 steps. Average sponge_h converged to 95 mm (just below threshold).

2. **Adaptive KL noise blow-up**: with `desired_kl=0.01, schedule="adaptive"`, when policy converges to the local-min and surrogate loss flattens, rsl_rl pushes `action_noise_std` upward each iteration (no KL hits the target). std climbed monotonically 2.68 → 6.31 over 1000 iterations. By iter 1300 the policy is essentially executing random ±6 rad action perturbations, which destroys whatever fine grasp/lift behavior remained.

### Remediation options (NOT applied this session — Phase 1.A target met by ckpt 1100)

For future P3 / Phase 1.B runs, change one or more of:

| Fix | Code change | Trade-off |
|---|---|---|
| **A. Don't terminate on success** | `roarm_pick_env.py:_get_dones` → return zeros for terminated even in P3, just give one-shot bonus | sustained lift reward continues; eliminates exploit; recommended |
| **B. Larger success bonus** | `cfg.success_bonus: 10 → 100` | weaker fix; still creates a cliff at 100 mm |
| **C. Per-step "altitude" reward** | replace `lift = max(0, sponge_z − TABLE_Z)` with `lift = sponge_z + small_height_bonus_above_threshold` | smoother gradient |
| **D. Freeze noise** | `RoArmPickPPORunnerCfg.policy.init_noise_std=0.5`, `algorithm.schedule="fixed"` | prevents blow-up but loses exploration |
| **E. Smaller `desired_kl`** | `algorithm.desired_kl: 0.01 → 0.005` | gentler adaptive schedule |

**Recommended for next P3 run**: A + E together. A removes the local-min exploit; E softens the KL-driven noise schedule so a temporary loss-flat patch doesn't snowball.

---

## Phase 1.B — 4-sponge # tower stacking env (DESIGN, not implemented)

### Geometry (HARD RULE #19/#20, frozen 5/03 evening)

Edge-stand sponge: 47 mm tall × 22 mm wide × 125 mm long.

```
TOP VIEW (# tower, center at (+0.280, 0.000) m base coord)

        Y
        ↑
        │   L1.sp1   L1.sp2          z up
        │  ┌──┐    ┌──┐                ↑  L2.sp3 ━━━━━━━━━━━━━━━ (z=0.0705)
   +0.04│  │  │    │  │                │  L2.sp4 ━━━━━━━━━━━━━━━
        │  │  │    │  │                │   ↕ L1.sp1│      │L1.sp2 (z=0.0235)
        │  └──┘    └──┘                │     ║      ║         ║
        │═══════════════ ← L2.sp3      │     ║      ║         ║       table
        │═══════════════ ← L2.sp4      └─────────────────────────→ X
        └──────────→ X
   −0.04
```

| Element | base-coord pos (x, y, z m) | edge-stand orientation | wrist_roll @ place |
|---|---|---|---|
| L1.sp1 (south of center, X-aligned long axis) | (+0.280, −0.0435, +0.0235) | long-axis = X | 0° |
| L1.sp2 (north of center, X-aligned long axis) | (+0.280, +0.0435, +0.0235) | long-axis = X | 0° |
| L2.sp3 (west, Y-aligned long axis, on top of L1) | (+0.2465, 0.000, +0.0705) | long-axis = Y | 90° |
| L2.sp4 (east, Y-aligned long axis, on top of L1) | (+0.3135, 0.000, +0.0705) | long-axis = Y | 90° |
| TCP grasp z (per HARD RULE #19) | — | — | +0.033 m world (top 70% grip) |
| TCP place L1 z | — | — | +0.033 m |
| TCP place L2 z | — | — | +0.080 m |

Total tower height 0.094 m, footprint 0.125 × 0.125 m.

### Task formulation (proposed: progressive curriculum)

| Sub-phase | # sponges | Targets | Rationale |
|---|---|---|---|
| **1.B-α single place** | 1 sponge in source region | place at L1.sp1 fixed target (no L1.sp2 yet) | warmup; resume from P3 ckpt 1100 |
| **1.B-β L1 only** | 2 sponges in source | L1.sp1 + L1.sp2 (two targets) | learn 2-step sequencing |
| **1.B-γ L1 + 1 L2** | 3 sponges | L1.sp1, L1.sp2, L2.sp3 (cross orientation) | learn wrist_roll switch |
| **1.B-δ full # tower** | 4 sponges | full L1+L2 | final task |

Each sub-phase resumes from the previous's final checkpoint.

### Env extension (RoArmStackEnv, proposed code path)

`roarm_rl/roarm_stack_env.py` (NEW), modeled on `roarm_pick_env.py`:

#### Observation (47-dim, state-only, HARD RULE #17)
| Block | Dim | Source |
|---|---|---|
| joint_pos scaled to [-1,1] | 6 | `_robot.data.joint_pos` |
| joint_vel × 0.1 | 6 | `_robot.data.joint_vel` |
| 4 sponges × pos_local (3) | 12 | `_sponges[i].data.root_pos_w − env_origin` |
| 4 sponges × quat (4) | 16 | `_sponges[i].data.root_quat_w` |
| `tcp_to_active_target` (3) | 3 | active sub-target this step |
| `current_stage` one-hot (4) | 4 | which sponge is "next to place" |
| **Total** | **47** | |

For 1.B-α/β/γ pad with zeros for unused sponges to keep obs shape constant across sub-phases.

#### Action: same as Phase 1.A — 6-dim joint position delta target.

#### Stage advance logic (in `_compute_intermediate_values`)
```
For active stage k ∈ {0..3}:
  target_xy = TARGETS[k]  # L1.sp1, L1.sp2, L2.sp3, L2.sp4
  target_z  = +0.033 if k<2 else +0.080
  if !stage_completed[k]:
    if (sponge[k] within ±15mm of target_xy)
       and (|sponge_z − target_z| < 10mm)
       and (gripper_open)  # released sponge
       and (sponge_stable_for_30_frames):
      stage_completed[k] = True
      next_stage = k+1  # advance
```

#### Reward (per active stage k)
```
r =
  −‖tcp − sponge[k]‖                           # reach active source sponge
  + 5.0 · max(0, sponge_z[k] − TABLE_Z)        # lift it
  + 2.0 · grasp_bonus[k]                       # grasp it
  − 1.0 · ‖sponge[k] − target[k]‖              # carry-to-target reward
  + 50.0 · stage_just_completed[k]             # one-shot per-stage bonus
  + 100.0 · all_stages_completed_first_tick    # final tower bonus
  − 0.1  · (placed_sponge_displaced_after_completion)  # stability penalty
```

Critical: **do NOT terminate on success of intermediate stages** — only after stage 3 (or never; rely on truncation). This avoids the local-min exploit observed in P3.

#### Spawn (per HARD RULE #20)
- 4 sponges spawn in 4 of the source regions R1-R4 (one per region) — random assignment
- Random yaw per sponge (so wrist must rotate to align)
- Tower zone (rect around (+0.28, 0)) excluded from spawn

#### Wrist orientation: free-form
The policy must learn to orient wrist_roll to 0° (L1) or 90° (L2). No explicit constraint — placement reward + stability check encode this implicitly.

### Open design questions (need decision before coding)

1. **Pre-placed scaffolding for 1.B-γ/δ?** Should L1 sponges spawn already-placed in tower (kinematic-pin until policy releases), or must policy place them itself?
   - Pin: faster learning of L2 placement
   - No-pin: end-to-end task; harder

2. **Kinematic-attach vs proper grasp?** Phase 1.A uses kinematic-attach when (dist<25mm) ∧ (gripper>0.4 rad). For stacking, this is fine for grasp, but for **release** at placement: should the placed sponge stay rigid (no physics) until next step, or fall under gravity? (Gravity = realism but can topple stack.)

3. **Real-to-sim transfer plan**: same edge-stand mass/friction as Phase 1.A but now stacking dynamics matter (stack stability under perturbation). Probably need `restitution=0` and high friction (already set).

4. **Curriculum jumps**: when 1.B-α reaches 95% success, freeze and start 1.B-β? Or interleave for 50/50 episodes?

### Implementation roadmap (estimate)

| Step | Effort | Dependency |
|---|---|---|
| 1.B-α env wrapper + sanity test | ~3 hours coding + 1 hour B200 sanity | — |
| 1.B-α PPO 500 iter (resume P3 ckpt 1100) | ~4 min B200 | env ready |
| 1.B-α eval | 30 sec | — |
| 1.B-β env extension (2 targets, stage logic) | ~2 hours | 1.B-α verified |
| 1.B-β PPO 1000 iter | ~7 min | env ready |
| 1.B-γ wrist_roll switch + L2.sp3 | ~3 hours | 1.B-β verified |
| 1.B-γ PPO 1000 iter | ~7 min | — |
| 1.B-δ full tower | ~2 hours | 1.B-γ verified |
| 1.B-δ PPO 1000-2000 iter | ~15 min | — |
| **Total** | **~10-15 hours coding + ~35 min B200** | — |

**Buffer vs HARD RULE #26 deadline**: 12 days remaining. Even at conservative 2-3 hour/day cadence, 1.B should fit comfortably with ~7-8 days slack for Blender visualization, paper draft setup, real-deploy translation.

---

## Checkpoint inventory (Step E)

### B200 paths

```
$ROARM_B200_ROOT/logs/roarm_rl/
├── roarm_pick_p1_500iter_seed0/
│   ├── model_0..499.pt         (12 ckpts)
│   └── events.out.tfevents.*   (528 KB)
├── roarm_pick_p2_500iter_seed0_resumeP1/
│   ├── model_500..998.pt       (11 ckpts)
│   └── events.out.tfevents.*   (530 KB)
└── roarm_pick_p3_1000iter_seed0_resumeP2/
    ├── model_1000..1997.pt     (21 ckpts; recommended: model_1100.pt)
    └── events.out.tfevents.*

$ROARM_B200_ROOT/logs/phase1/
├── eval_p1_499.log             (97.75% success)
├── eval_p2_998.log             (99.22% success)
├── eval_p3_1050.log            (99.54% success)
├── eval_p3_1100.log            (99.68% success ⭐)
├── eval_p3_1200.log            (93.02% success)
├── eval_p3_1997.log            (26.87% success — collapsed)
├── train_p2_500iter.log
└── train_p3_1000iter.log
```

### Recommended best ckpt for any downstream Phase 1.B work

```
$ROARM_B200_ROOT/logs/roarm_rl/roarm_pick_p3_1000iter_seed0_resumeP2/model_1100.pt
```
99.68% success, 427 mm sponge lift, 99.56% grasp@reset, 1024 trials.

---

## Files added/modified this session

### Local repo (`/home/cgxr/Documents/Robotics/RoArm_Project/`)
| File | Status | Purpose |
|---|---|---|
| `roarm_rl/eval_policy.py` | NEW | Frozen-policy eval with reset-idx hook |
| `claudedocs/phase1_step_e_complete_20260507.md` | NEW | This doc |

### B200 (`$ROARM_B200_ROOT/`)
| Path | Status |
|---|---|
| `code/roarm_rl/eval_policy.py` | NEW (transferred via `roarm_rl/transfer_to_b200.sh`) |
| `logs/roarm_rl/roarm_pick_p2_500iter_seed0_resumeP1/` | NEW (P2 train) |
| `logs/roarm_rl/roarm_pick_p3_1000iter_seed0_resumeP2/` | NEW (P3 train) |
| `logs/phase1/eval_p{1,2,3}_*.log` | NEW (5 eval logs + 2 train logs) |

### NOT modified (intentionally)
| File | Why preserved |
|---|---|
| `roarm_rl/roarm_pick_env.py` | P3 termination-on-success exploit identified but not patched — Phase 1.B will introduce the fix; preserving Phase 1.A reproducibility for 4090 comparison |
| `roarm_rl/agents/rsl_rl_ppo_cfg.py` | adaptive-KL noise blow-up not mitigated yet — same reasoning |

---

## HARD RULE #26 deadline tracking

| Day | Target | Status |
|---|---|---|
| Day 0 (5/07) | Phase 0 V1 launch | ✅ done early |
| Day 1 (5/08) | Steps A-C URDF→USD | ✅ done 5/07 |
| Day 3 (5/09) | env wrapper + 1 ep rollout | ✅ done 5/07 |
| Day 7 (5/13) | PPO 1K iter pass | ✅ done 5/07 (P3 hit 1000 iter, 99.68% peak at 1100) |
| **Day 12 (5/19)** | **result + report** | **5+ days ahead with Phase 1.A complete; Phase 1.B (full # tower) planned to fit in remaining time** |

---

## Continuation prompt (next session — Phase 1.B start)

```
RoArm M3 — Phase 1.B-α start (single-target place stacking entry).

Status 5/07 late-night #4 (Step E): Phase 1.A curriculum COMPLETE.
  - Best ckpt: $ROARM_B200_ROOT/logs/roarm_rl/roarm_pick_p3_1000iter_seed0_resumeP2/model_1100.pt
  - 99.68% success, 427mm sponge lift, 1024 trials.
  - eval_policy.py + reset-idx hook proven on P1/P2/P3.

Next session (1.B-α single-place):
  1. Read claudedocs/phase1_step_e_complete_20260507.md (this doc — Phase 1.B design section)
  2. Decide open design questions (esp. #1 pre-placed scaffolding, #2 kinematic vs gravity release)
  3. Create roarm_rl/roarm_stack_env.py:
     - 1 sponge (existing R1-R4 spawn) + 1 fixed L1.sp1 target (+0.280, -0.0435, +0.0235 base)
     - obs 47-dim (joint+joint_vel+4*sponges+target_vec+stage_one_hot, pad zeros for sponges 1-3)
     - action 6-dim (same)
     - reward = reach + lift + grasp + carry + place_50_bonus (+ stability penalty)
     - DO NOT terminate on intermediate stage success (avoid P3 local-min exploit)
  4. Sanity test 1 env × 50 step
  5. Scale 4096 envs sanity
  6. PPO 500 iter resume from P3 ckpt 1100 (smoke), eval, plan 1.B-β

HARD RULE #14 fail-fast guard, #11 NO /half-clone, #15 cu128 sm_100 verify, #17 visual RL X.
HARD RULE #26 deadline 5/19 — currently 12 days ahead.

B200 entry: ssh JHPark; set -e; source env.sh; [[ -z $ROARM_B200_ROOT ]] && exit 1;
            micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1;
            export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1;
            cd $ROARM_B200_ROOT/code
```

## Cleanup TODO before next session

- [ ] Decide: keep all 21 P3 ckpts or prune to {1050, 1100, 1200, 1997} (~16 MB savings, marginal)
- [ ] Optional: re-run P3 with fix A+E (no-terminate + KL=0.005) to validate the fix before applying to 1.B
- [ ] Document Phase 1.B-α open design Q's outcomes in next session's continuation
