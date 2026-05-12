# 랩미팅 5 slide — Phase 1.B-α RL on B200 (2026-05-12)

> **Context**: HARD RULE #26 lock-in (5/07 night ~ 5/19), B200 physics-only Isaac Sim/Lab RL 단일 전속. P6v7 ε ablation 학습 진행 중.

---

## Slide 1 — B200 Isaac Sim & Lab 활용 (state-only RL)

**자원 매핑 (5/07 night ~ 5/19 LOCK-IN)**

| 자원 | 역할 | Render | RL |
|---|---|---|---|
| B200 GPU0 (sm_100, cu128) | Physics-only Isaac Sim RL 전속 | **OFF** | PPO, # tower stacking |
| RunPod A6000 | HOLD | — | — |
| 로컬 4090 | Deploy / Blender fallback | (Optix 검증 후) | — |

**Render OFF 구현** (`roarm_rl/train_ppo.py:47`):
```python
app_launcher = AppLauncher(headless=True, enable_cameras=False)
```

**왜 state-only인가**: HARD RULE #17 — Tiled Camera Annotator → CUDA tensor mapping SIGSEGV (NVIDIA Discussion #4339, L40s/H100/H200/B200 visual RL 미지원). state-only는 Vulkan ICD + PhysX만 사용 → SIGSEGV path 회피. 5/07 night PIVOT 후 Phase 0 V1 통과.

**Task** (HARD RULE #19/#20): 우물정자 # tower stacking, edge-stand 47mm sponge, L1.spot1 target `(+0.280, −0.0435, +0.0114) m`. 1-sponge → L1 single placement (Phase 1.B-α).

---

## Slide 2 — PPO 알고리즘 + 학습 setup

**RL framework**: rsl_rl 3.1.2 OnPolicyRunner, Isaac Lab `DirectRLEnv`, 6-DOF RoArm-M3 articulation.

| 항목 | 값 | 근거 |
|---|---|---|
| Envs | **4096** (parallel) | B200 80GB VRAM 활용 |
| Throughput | **240~258K steps/s** | 1000 iter = 98.3M timesteps ≈ 7min wall |
| Observation | **28-dim** | joint q (6) + joint qd (6) + sponge_pos_local (3) + sponge_quat (4) + tcp→sponge (3) + target_pos_local (3) + sponge→target (3) |
| Action | **6-dim joint delta** (no IK) | `q_target += action_scale × clamp(action, -1, 1)`, action_scale = 0.1 rad |
| episode_length_s | 4.0 (400 step @ 100Hz) | Phase 1.A 그대로 |
| desired_kl | 0.005 | Phase 1.A fix |
| init_noise_std | 0.8 | Phase 1.A |

**Phase warm-start chain** (P3 Phase 1.A → P4 → P5 → P6 점진 reward shaping):
- **P4** = reach + lift + grasp + success (Phase 1.A 그대로, target 무시)
- **P5** = P4 + nav_reward (grasped 상태에서만 −‖sponge − target‖)
- **P6** = P5 + place_bonus (near + open + stable) + place_success_bonus

근거 (`claudedocs/phase1_balpha_design_decisions_20260508.md`): Phase 1.A new_1100 transient dip 교훈 — reward 한 번에 크게 바꾸면 ~100 iter transient → 점진 phase 도입.

---

## Slide 3 — 성공 작업 (Phase 0 → P5까지 PASS)

**Phase 0 — State-only RL launch 검증 (5/07 night ~ Day 1 EOD)**
- V1 PASS: `headless=True, enable_cameras=False` + Vulkan ICD + PhysX articulation step → 100 step rollout, no SIGSEGV.
- HARD RULE #17 narrow 정정 정당화 (visual RL 한정, state-only는 작동).

**Phase 1.A (precursor) — Reach/Lift/Grasp CONVERGED**
- 22-dim obs, P3 reward, single sponge pick. new_1497 best ckpt = lift/grasp success ~96%.
- → Phase 1.B-α 28-dim warm-start의 base policy.

**P4 (stack warm-start) — PASS**
- Phase 1.A의 P3 reward 그대로 + 28-dim obs 확장 (target 정보 추가).
- 결과: 22→28 expand 후에도 lift/grasp 능력 유지. grasped_frac ≈ 0.93 안정.

**P5 (nav) — sponge_target_dist ~600mm → ~100mm (≈83%↓)**
- P5v2 1500 iter, nav_reward 작동 확인. sponge_target_dist final = **103mm** (R1-R4 spawn 초기 ~600mm 기준).
- ⚠️ caveat: target 100mm 안쪽 진입은 아직 X (place_dist_thresh 25mm). nav 작동 정성적 PASS, 정량 squeeze는 P6 issue.

**핵심 자산**: P5v2 model_1499 — grasp + nav 능력 보존된 정책. 모든 P6 실험의 resume base.

---

## Slide 4 — 실패 작업: P6 7회 + 4단계 chicken-and-egg 진단

**핵심 fail metric** (모든 P6 버전 공통): `place_success_rate ≈ 0` 또는 ≤1.5%.

| Ver | 변경 | place_succ | sponge_height | 진단 |
|---|---|---:|---:|---|
| P6v1 | place_dist=25mm 원본 | 0.000 | — | std 5.28→7.38 발산 |
| P6v2 | thresh 100mm + std reset 1.5 + entropy 0.001 | 0.000 | — | std 안정化 1.46. **place_cond fire 0회** |
| P6v3 | place_cond에서 gripper_open 제거, sponge_grounded 추가 | 0.000 | 104mm | sponge 공중 10cm hover |
| P6v4 | release-path reshape (near_gate, lower_reward, open_bonus×10) | 0.000 | 132mm (악화) | dist 105→144mm 멀어짐 |
| P6v5 | was_grasped latch + actor.6.bias[5]=0 reset | 0.000 | 125mm | bias가 50 iter 만에 재saturate |
| P6v6 | ManiSkill REPLACE tower 구조 (8-cap) | **0.0148** | 98mm | iter 0 4.22% → iter 999 1.48% (감소) |
| P6v7 | ungrasp_signal sign fix (`(high−q)/(high−low)`) | 학습 중 | — | 5/13 evening launched |

**4단계 chicken-and-egg 진단 chain**:

1. **CE #1 (P6v2→v3)**: `actor.6.bias[5]=+0.8446` + std=1.5 → gripper close saturate → `gripper_open` AND-condition 1000 iter fire 0회 → place_bonus signal 영원히 0.
2. **CE #2 (P6v3→v4)**: gripper_open 분리 후 `sponge_grounded` (z<+30mm) 신규 bottleneck → sponge가 +100mm hover 유지 → grounded fire 0회. Hold path = lift(+5) + grasp(+2) + nav(−0.5) ≈ **+6.5/step**, release path는 ~30 step credit gap.
3. **CE #3 cliff effect (P6v4→v5)**: `near_gate` sharp binary cutoff → near zone 진입 시 reward 절벽 (far +1.65/step → near −0.50/step) → PPO advantage 음수 → far zone 머묾, sponge 더 높이/멀리.
4. **Stage-3 TRAP / Reward Misspecification (P6v5→v6→v7)**:
   - was_grasped latch + bias reset (P6v5) → bias가 **50 iter 만에 다시 양수** (reward gradient 일관 close 방향). 표면 fix 무효.
   - 정량 진단: stage 3 hold 누적 = 6.59/step × 75% × 400 = **1976** vs stage 4 transition = 8/step × 1.5% × 400 = **48** → hold가 **41× 우세** → globally optimal.
   - ManiSkill REPLACE tower (P6v6, max 8/step cap) 도입 → iter 0 4.22% 즉시 발생 BUT iter 999 1.48%로 회귀.
   - ⚠️ **잠재 bug 발견 (P6v6 결과 polling, 5/13)**: `ungrasp_signal = (q−low)/(high−low)` 정의가 RoArm gripper convention (q LOW=OPEN, q HIGH=CLOSED)과 반대 sign — closed일 때 ungrasp_signal=0.94 (release 의미인데 실제 closed).

---

## Slide 5 — 현재 P6v7 ε ablation + 분기 + Future Work

**P6v7 (학습 중, 5/13 evening launched)**:
- Patch: `ungrasp_signal = (high−q)/(high−low)` (sign flip), resume P6v6 model_999, no reset_std, no bias reset, 1000 iter ETA ~7min.
- Hypothesis: ε 단독으로 release 인센티브 sign 정상화 → gripper_open ↑ → stage 4 transition ↑.

**사용자 명시 ablation 분기 (iter 999 결과 기준)**:

| 분기 | 조건 | 다음 step |
|---|---|---|
| **(A) FULL SUCCESS** | stage4 > 5% AND open > 10% | P7 squeeze (place_dist 100→50→25mm curriculum) |
| **(B) PARTIAL** | stage4 2-5% | resume + fine-tune (action_penalty 축소, episode 100 step) |
| **(C) FAIL** | stage4 ≤ 2% AND P6v6과 차이 미미 | α (episode 400→200) + Fix A (success_zone=50mm jackpot+20) 결합 P6v8 |
| **(D) 회귀** | stage4 < 1% OR open=0 | sign fix가 또 다른 bug 노출 — root cause 재분석 |

**5/19 deadline (6일 남음) 시나리오**:
- (A)/(B) → 결과물 = "B200 state-only RL로 stacking 학습 가능" 1차 demo + P7 curriculum 진입.
- (C)/(D) → α + Fix A P6v8 1회 시도 (~10min wall). 추가 fail 시 사용자 보고 → Blender Optix fallback 또는 RunPod resume.

**Future Work (5/19 이후 자동 release)**:
1. **HARD RULE #21 3-way 비교 재개**: Pure VLA (BC) / Pure RL (이 라인) / Hybrid (DAPG/AWAC BC warmstart + RL finetune).
2. **HARD RULE #22 4-Axis Matrix**: (real scale) × (sim scale) × (backbone: SmolVLA/OpenVLA-OFT/π0) × (training: BC/Hybrid/Pure RL).
3. **HARD RULE #24 v7 collection**: 200ep stacking single-step, 5×5 grid stratified, 2 viewpoints.
4. **Real-to-sim hybrid evaluation**: 본 RL 정책 → real RoArm deploy → sim2real gap 정량 (HARD RULE #21 핵심 contribution).

**메시지**: 4단계 chicken-and-egg 진단 chain을 거치며 reward 구조 misspecification (hold-path globally optimal)이 핵심 lesson. ManiSkill REPLACE 도입 + ε sign fix가 누적 진단 결과. 5/19까지 1-2 iteration 더 시도 가능.

---

## Appendix — 핵심 수치 cross-verify

| Source | 검증 항목 | 값 |
|---|---|---|
| `train_ppo.py:47` | `headless=True, enable_cameras=False` | ✓ |
| `roarm_stack_env.py:89-90` | action_space=6, observation_space=28 | ✓ |
| `phase1_balpha_p6v6_session_20260513_result.md` | P6v6 iter 999 stage4=1.48%, ungrasp=0.94 (sign bug) | ✓ |
| `phase1_balpha_p6v5_session_20260512_result.md` | P6v5 iter 50 gripper_open 54%→3% (bias re-saturate) | ✓ |
| `phase1_balpha_p6v4_session_20260509.md` | P6v4 dist 105→144mm cliff effect | ✓ |
| `phase1_balpha_p6v3_session_20260508_evening.md` | sponge_height 104mm hover, grounded fire 0 | ✓ |
| `phase1_balpha_p6v2_session_20260508_late.md` | actor.6.bias[5]=+0.798 close-saturate root cause | ✓ |
| HARD RULE #26 (MEMORY.md) | 5/19 deadline, B200 state-only RL 전속 | ✓ |
