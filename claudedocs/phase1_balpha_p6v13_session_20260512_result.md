# Phase 1.B-α P6v13 Session Result (2026-05-12 오후)

## 요약 (TL;DR)

**🚨 V2 + V3 fix BACKFIRED — 신규 failure mode "Zone Avoidance" 등장**

P6v12 η-v1 (transient +10 no-gate, stage 2 cap 2.0)의 결함을 fix하려 V2 (gripper_open gate + stage 3 close-cap 3.0) + V3 (sponge_stable vel 0.05→0.10) 적용. 결과: PPO가 zone 진입 자체를 회피 (`is_on_target_rate` 40.6% → **0%**). 5번째 reward farming local optimum 등장 — pure reward shaping 한계 확정.

## Configuration

- Resume: `p6v12_eta_stage2cap_stage3transient_resumeP6v11/model_999.pt`
- Reset_std: 1.0 → **1.5** (escalate from P6v12 final 0.88, +70% exploration boost)
- Entropy: 0.001 / Episode 2.0s (200 step) / num_envs 4096 / max_iterations 1000
- env md5: `5f0b6e047c0e38ca67e55ae94e165a0f` (local↔B200 verified)
- Launch PID: 2188003, wall **6:44** @ ~245K steps/s

### Patches (roarm_rl/roarm_stack_env.py)

```python
# V3 (line 578-582): sponge_stable threshold relax
- sponge_stable = sponge_vel_mag < 0.05
+ sponge_stable = sponge_vel_mag < 0.10  # capture release bounce-down 150ms window

# V2 (line 605-623): gripper_open AND-gate + stage 3 close-cap 3.0
- just_on_target = is_on_target & ~self._stage3_fired
- self._stage3_fired = self._stage3_fired | is_on_target
- stage3_r = 6.0 + 0.5*ungrasp + 0.5*static + 10*just_on_target.float()
+ just_on_target = is_on_target & gripper_open & ~self._stage3_fired
+ self._stage3_fired = self._stage3_fired | (is_on_target & gripper_open)
+ stage3_r_open  = 6.0 + 0.5*ungrasp + 0.5*static + 10*just_on_target.float()
+ stage3_r_close = torch.full_like(stage3_r_open, 3.0)
+ stage3_r = torch.where(gripper_open, stage3_r_open, stage3_r_close)
```

## iter 999 Final Metrics (vs P6v12)

| Metric | P6v12 (η-v1) | P6v13 (V2+V3) | Δ |
|---|---|---|---|
| Mean reward | 854 | **894** | +5% (오해 유발 — Path A 강화) |
| action_std | 0.88 | 1.18 | std 유지 (reset 1.5 영향) |
| gripper_open_rate | 0.064 | **0.061** | -0.003 (FLAT) |
| **is_on_target_rate** | **0.406** | **0.0000** | 🚨 **40% → 0%** |
| sponge_target_dist | 0.105m | **0.167m** | 🚨 **+59% farther** |
| xy_offset | 0.082 | 0.104 | +27% worse |
| z_offset | 0.048 | 0.110 | +130% worse (hover 더 높음) |
| **stage2_grasp_frac** | 0.454 | **0.865** | +90% (transport 학습 강화) |
| **stage3_neartgt_frac** | **0.406** | **0.0000** | 🚨 zone 진입 0회 |
| stage4_success_frac | 0.0002 | 0.0000 | both ≈ 0 |
| jackpot_fire_rate | 0 | 0 | sponge_stable 조건 미충족 |

## Iter Trajectory (sanity gate FAIL)

| iter | reward | gripper_open | on_target | stage4 | sponge_target_dist |
|---|---|---|---|---|---|
| 0 | 18 | **0.577** | 0.000 | 0 | 181mm (random reset_std=1.5) ✓ |
| **1** | 110 | **0.067** | 0.003 | 0 | 214mm | 🚨 **8.6× drop → sanity gate FAIL** |
| 5 | 385 | 0.071 | 0.39 | 0 | 200mm |
| 10 | 565 | 0.070 | 0.39 | 0 | 123mm |
| 50 | ~590 | 0.067 | 0.40 | 0 | 105mm |
| 500 | ~580 | 0.067 | 0.39 | 0.0001 | 105mm |
| **999** | **894** | **0.061** | **0.0000** | **0** | **167mm** |

**Critical observation**: iter 5-500은 transitional state (P6v12 ckpt 유산), iter 500 이후 정책이 zone avoidance로 deep shift. on_target 0.4 → 0.0 collapse.

## Root Cause 분석 (Critical Thinking)

### Reward Math (산수 cross-verify)

Zone 외부 (d=167mm) 정책 위치 선택 결정 트리:
- **외부 zone hover (d=167mm)**: stage 2 r = 4 + 3·(1−tanh(5·0.167)) = 4 + 3·0.317 = **4.95/step**
- **내부 zone closed (V2 close-cap)**: stage 3 close = **3.0/step** ← **외부보다 1.95 낮음**
- **내부 zone open**: stage 3 open = 6.0 + ungrasp(0~1) + static(0~1) = 6.5~7.0/step (drift risk)

→ PPO 1-step advantage analysis:
- Enter zone closed: **-1.95/step net loss**
- Stay outside: **status quo (best deterministic)**
- Enter zone open: **+1.55/step** but requires release exploration (rare)

→ 정책이 **zone 진입 회피** 학습 → on_target rate 0%.

### 5번째 Reward Farming Local Optimum

| Version | Farm location | Reward/step | Fix attempt | New farm |
|---|---|---|---|---|
| P6v6/v7/v8 | Stage 3 closed-hover near-target | 6.5 | — | (initial) |
| P6v9/v10 | Stage 3 closed-hover (same) | 6.5 | ε ungrasp sign + γ transport | (same) |
| P6v11 | **Stage 2 near-zone hold (no cap)** | 7.0 | β jackpot 5.0 | (escalation) |
| P6v12 | **Stage 3 close-hover (transient close 카운트)** | 6.5 + 17 first fire | η stage 2 cap 2.0 + transient +10 | (moved to stage 3) |
| **P6v13** | **Stage 2 OUTSIDE zone hold (d=167mm)** | 5.0 | V2 close-cap 3.0 + V3 vel relax | **(moved to outside zone!)** |

**5회 연속**: 매번 새로운 reward farming local optimum 발견 → pure reward shaping 본질적 한계.

### 본질적 문제 (Bootstrap Problem)

Stage 4 fire 조건: `is_on_target & gripper_open & sponge_stable`
- 각 component independent prob (P6v13 iter 999): 0 × 0.061 × 0.157 = **0** (joint ≈ 0)
- 학습 신호 없음 → PPO가 release path 학습 불가 → V2 close-cap 효과 측정 불가

V2 close-cap이 **이론적으로** open >> close advantage 생성하더라도 **stage 4 fire 안 됨 = path B 미실현 = PPO가 path B 존재 자체 학습 못 함**.

**Pure PPO + sparse stage 4 reward + manipulation = inherent exploration challenge** (HER/curriculum/demo 필요).

## P6v14 권장 방향 (사용자 결정 필요)

### Option A: 마지막 reward shaping 1회 (P6v14)
- V4: stage 3 close = **5.0** (stage 2 outside-zone 매칭, neutral entry)
- V5: stage 3 open transient = **30** (was 10, 강한 magnet)
- V6: stage 4 jackpot 20 → **100** (one-time big signal)
- 위험: 6번째 farming local opt 등장 가능. 5/19까지 ~7회 시도 가능 (1 run ~7min).
- 정량 기대: 산수 상 1-step open 매력 +25 first fire / +2 sustained, vs close -2. 시도 가치 있음.

### Option B: Curriculum (sponge init near target)
- _reset_idx 수정: sponge spawn region을 target ±5cm 좁힘 (Phase 0)
- Stage 4 fire 가능성 ↑ (transport 거의 0, release만 학습) → signal 확보
- 학습 converge 후 spawn region 점진 확대 (curriculum scheduler)
- 위험: 코드 변경 50-100 LOC + curriculum config 필요. 학습 시간 2-3× (multi-phase).
- 정량 기대: stage 4 fire rate iter 100에 5-10% 가능 (independence 회복).

### Option C: BC warm-start (v6 demos)
- v6 50ep real demos → state-only RL action label 추출 → BC pretrain actor
- PPO finetune 시 release path 이미 학습된 상태에서 시작
- 위험: train_ppo.py 대폭 변경 (~200 LOC), v6 22-dim ↔ stack 28-dim mapping
- 시간: 1-2일 셋업 + 학습

### Option D: 정지 + Pure RL 실패 보고
- 5/19 deadline 7일 잔여, P6v6~P6v13 모두 실패 paper trail로 보고
- 교수님께 "Pure RL state-only Isaac Sim B200 stacking은 inherent exploration failure" 명시
- HARD RULE #26 release 후 v7 collection / 4-axis matrix 진행
- 위험: 본 phase에서 결과물 0

## 추천 (Critical Analytical Stance)

**Option A + B 병렬** 권장:
- A를 1회 시도 (~10min): 5/19까지 시간 있고 산수 상 가능성 있음. 실패 시 즉시 B로 pivot.
- B를 동시 셋업: A가 fail 시 즉시 launch. Curriculum이 RL manipulation의 SOTA 접근 (DrS / Adroit / SoftGym 모두 curriculum).
- C는 시간 over-budget. D는 사용자 의지에 따라.

**5/19 deadline에서 가장 안전한 path = B (Curriculum)**: 실패해도 "tried curriculum" 학습 가능. Pure shaping은 5번 실패로 saturated.

## B200 Inventory

- `logs/roarm_rl/p6v13_v2_etav2_v3_velrelax_resumeP6v12/` — 22 ckpts (model_0~999)
- `logs/phase1Balpha/train_p6v13.{out,err}`
- `$ROARM_B200_ROOT/launch_p6v13.sh` — V2+V3 patch + sanity gate 명세

## HARD RULE 준수

- #8: 7건 archive 완료 (5/13 P6v6 / 5/13 evening P6v7 / 5/14 P6v7 / 5/14 evening P6v8 / 5/15 P6v8 / 5/12 night P6v6 / 5/12 P6v5 → MEMORY_archive_20260512.md). 현재 5 full bodies compliant.
- #11: /half-clone X 0회.
- #14: fail-fast guard 모든 ssh (set -e + source env.sh + ROARM_B200_ROOT/whoami/hostname 검증).
- #15: cu128 sm_100 alive (P6v13 wall 6:44 = 추가 검증).
- #17: state-only 28-dim only (visual RL 미사용).
- #18: 사용자 명시 V2+V3 진행 (Plan B = Agent A 권장).
- #19/#20: edge-stand 47mm / # tower geometry 그대로.
- #26: B200 physics-only RL 5/19 deadline **7일 잔여**.

## 다음 세션 즉시 명령

```bash
# Result polling (이미 완료)
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh; \
  [[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1; \
  ls \$ROARM_B200_ROOT/logs/roarm_rl/p6v13_v2_etav2_v3_velrelax_resumeP6v12/"'
```

## 사용자 confirm 대기

- Option A (P6v14 마지막 shaping) / Option B (Curriculum) / Option A+B 병렬 / Option C (BC) / Option D (stop) 선택
- 랩미팅 자료 (5/13): 8 PNG 충분 vs 3D scene render 추가 생성
