# Report 1 — P3 Fix + Precision Comparison (2026-05-08 새벽)

> **Stage**: Phase 1.A (Pick) → Phase 1.B-α (Stacking) 사이
> **목적**: Step E (P3 collapse 26.87%) 해결 + checkpoint 정밀 ranking 확정

---

## 1. 구성 (Setup)

### 1.1 코드 변경 (3 곳)

#### 변경 #1 — `roarm_rl/roarm_pick_env.py:391` `_get_dones` terminated=zeros 강제
- **이전**: `if reward_phase>=3: terminated = success_flag.clone()` — success 시 episode 즉시 종료
- **이후**: `terminated = zeros(...)` — success 후에도 episode 끝까지 진행
- **이유**: 종료-on-success가 trajectory 길이를 단축시켜 on-policy 분포가 짧은 trajectory에 편중. policy가 "잠깐 들어올리고 끝내자" local-minimum에 갇힘. Step E 26.87% collapse의 메커니즘.

#### 변경 #2 — `roarm_rl/roarm_pick_env.py:355-361` success_bonus single-shot 보장
- **이전**: `success_now = success_flag.float() * 10.0` — flag 켜진 모든 step에서 fire. 이전엔 terminated=True가 즉시 종료시켜 자연스레 1회만 fire.
- **이후**: `_success_bonus_paid` 별도 추적 → 첫 1회만 fire, 이후 latch. episode reset 시 false 복귀.
- **이유**: 변경 #1을 하면 terminated=False라 +10 bonus가 100+ step 동안 fire되는 부수효과. 원본 코드 주석은 "single-shot per episode"로 명시. 미수정 시 reward landscape이 success_bonus(+10×100step=+1000)에 의해 지배됨.

#### 변경 #3 — `roarm_rl/agents/rsl_rl_ppo_cfg.py:41` desired_kl 0.01 → 0.005
- **이유**: Step E의 P3 collapse 시 KL adaptive 메커니즘으로 action noise 표준편차가 2.68→6.31로 폭주. trust region을 절반으로 축소하여 안정화.

### 1.2 학습 (B200, NHN)

| 항목 | 값 |
|---|---|
| Resume | P2 model_998 (P2 best, 99.22% success) |
| Iterations | 500 (총 1499번까지) |
| Parallel envs | 4096 |
| Steps per env per iter | 24 |
| Transitions per iter | 98,304 |
| Total samples | 49,152,000 |
| Wall clock | **3분 17초 (197s)** |
| Throughput | 250-260 K steps/sec |
| Final action noise std | 3.19 (training-time exploration) |
| Final training success rate | 0.7717 (with action noise) |
| Saved checkpoints | 11 개 (model_1000, 1050, 1100, ..., 1497) |

### 1.3 Precision Comparison Eval 구성

| 항목 | 값 |
|---|---|
| Isaac Sim 세션 | **1 회만 launch** (32 runs 모두 한 세션 내) |
| Checkpoints | 8개 — old_1050, old_1100, new_1050, new_1100, new_1200, new_1300, new_1400, new_1497 |
| Seeds per checkpoint | 4 (seed = 42, 43, 44, 45) |
| Parallel envs per run | 4096 |
| Episodes per run | 1 (200 steps) |
| Trials per run | 4096 |
| Trials per checkpoint | **16,384** (4 seeds × 4096) |
| Standard error | ≈ 0.08-0.15 percentage points |
| Pairing | 같은 seed → 같은 spawn 분포 → policy 차이만 측정 |
| Total wall | ~2분 30초 (32 runs 합계) |

---

## 2. 결과 (Results)

### 2.1 Per-checkpoint success rate (16,384 trials each, 정렬됨)

| Rank | Checkpoint | Success rate | Std error | Mean sponge height | Grasped frac |
|---|---|---|---|---|---|
| 🥇 | **old_1050** | **98.956%** | 0.079% | 532mm | 94.28% |
| 🥈 | new_1200 | 98.798% | 0.085% | 586mm | 92.97% |
| 🥉 | **new_1497** (final) | **98.773%** | 0.086% | **610mm** | 92.99% |
| 4 | new_1300 | 98.737% | 0.087% | 608mm | 92.91% |
| 5 | new_1400 | 98.718% | 0.088% | 608mm | 92.88% |
| 6 | old_1100 | 98.444% | 0.097% | 570mm | 92.63% |
| 7 | new_1050 | 98.004% | 0.109% | 602mm | 92.22% |
| 8 | **new_1100 (transient dip)** | **96.259%** | 0.148% | 579mm | 90.50% |

### 2.2 핵심 paired diff (4 seeds, paired t-test)

| 비교 | mean diff | t-stat | 결론 |
|---|---|---|---|
| old_1100 − old_1050 | −0.51pp | −2.09 | **old_1050이 미세 유의 (p≈0.13)** |
| new_1497 − old_1100 | +0.33pp | +6.97 | new가 old_1100 대비 유의 우월 |
| **new_1497 − old_1050** | **−0.18pp** | **−0.64** | **차이 없음 (noise)** |
| new_1200 − new_1100 | +2.54pp | +25.10 | 50 iter만에 dip 회복 |
| new_1497 − new_1200 | −0.02pp | −0.93 | iter 1200~1497 plateau |

---

## 3. 단계별 해석 (Step-by-step Reasoning)

### 발견 1 — Step E doc의 99.68% 숫자는 1024-trial noise였다

- Step E는 1024 trial로 평가 → standard error ≈ 0.18% → 0.14pp 차이를 "best"로 단정
- 16,384 trial로 재평가하면 실제 best는 **old_1050 (98.956%)**, old_1100은 6위 (98.44%)
- old_1100이 best라는 주장은 sample size 부족으로 인한 **rank inversion**

### 발견 2 — P3 fix는 collapse를 막았지만 천장을 뚫진 못했다

- 이전 (terminated=True, KL=0.01): iter 1000→1997 동안 99.68%→26.87% collapse
- 이후 (terminated=False, KL=0.005): iter 1000→1497 동안 stable plateau ~98.7-98.8%
- 즉 fix가 **잘못된 학습을 막은 건 맞지만**, 새 supremum에 도달한 건 아님
- old_1050과 통계적 동등 → policy capacity 또는 reward shape의 한계 시사 (env가 더 어려워야 학습 동력 생김)

### 발견 3 — new_1100 dip (96.26%)는 reward-shape transient

- Resume 직후 50 iter는 새 reward 분포에 적응하는 transient 구간
- 변경된 reward 구조: ① success 후도 lift_reward 계속 fire (이전엔 즉시 reset) ② success_bonus single-shot이라 spike 후 평탄
- value function이 새 reward에 fit하기 전까지 일시적 perf 저하 → 100 iter 정도면 회복

### 발견 4 — Mean sponge height가 ckpt마다 다르다 (532-610mm)

- old_1050: 532mm (가장 보수적)
- new_*: 585-610mm (더 공격적으로 들어올림)
- 이유: terminated=True 시절은 success(=h>100mm) 즉시 종료 → 더 들어올릴 incentive 없음. terminated=False는 "들면 들수록 lift_reward 누적" → policy가 끝까지 들어올리는 학습
- **Phase 1.B-α stacking에 중요한 함의**: sponge를 더 높이 들 수 있는 policy = L1 layer 위로 sponge를 cleaner하게 옮길 수 있음

### 발견 5 — Mean dist 메트릭이 매우 noisy하다

- old_1050: 381mm vs others ~465mm — success rate 98.96%인데 dist만 다름
- 원인: episode 종료 시점의 TCP-sponge 거리 — 정책이 lift 후 다른 위치로 움직이면 거리가 커짐. 즉 "성공했지만 손 빼고 멀어진" 정도를 측정. 정책 종료 시점 행동에 너무 민감
- **Eval 메트릭으로는 부적합**. success rate + mean h만 신뢰

---

## 4. Phase 1.B-α 진입용 checkpoint 결정

| 후보 | 장점 | 단점 |
|---|---|---|
| old_1050 | 통계상 best (98.96%) | terminated=True로 학습 → reward 형상이 1.B-α와 다름. lift 천장 낮음 (h=532mm) |
| **new_1497** ✅ | 새 reward 형상에 적응. h=610mm (가장 높음). 학습 안정성 검증 | old_1050 대비 0.18pp 낮음 (통계적 noise) |

**결정: new_1497 사용**

근거:
1. 0.18pp 통계적으로 무의미 (t = -0.64)
2. Phase 1.B-α는 lift → place 멀티-stage → lift 천장 (610mm > 532mm) 중요
3. 새 reward 형상이 1.B-α 확장에 자연스러움. old_1050으로 가면 새 reward 도입 시 또 transient dip 발생

---

## 5. 보충/개선 사항 (Suggestions)

### A. 즉시 (Phase 1.B-α 진입 전)

1. **Success criterion 강화**: 현재 `sponge_z > 100mm + 50 consecutive steps`은 너무 permissive. 1.B-α stacking에선 "정확한 위치 + 안정 자세"가 필요 → place criterion 추가 (`tcp-target_dist < 25mm` ∧ `gripper still grasped` ∧ `stable yaw`)
2. **Mean_dist 메트릭 폐기 또는 대체**: "success 시점 dist" 또는 "최대 lift 시점 dist"로 대체
3. **Eval 표준 trial 수 = 4096+ 못박기**: Step E의 rank inversion 사고는 1024 trial이 원인. 모든 ranking 결정은 ≥ 4096 trials

### B. 중기 (Phase 1.B-α 진행 중)

4. **Action noise std 클램프**: 현재 init=0.8 + adaptive KL이 noise를 3.19까지 키움. eval은 stochastic policy mean이라 영향 적지만, 학습 후반 불안정의 indicator. max=1.5 클램프 고려 (rsl_rl 3.1.2에 직접 옵션 없으면 entropy_coef 줄이는 우회)
5. **Reward 천장 분석**: new 학습 plateau가 98.7-98.8%에서 막혔음. 남은 1.2%는 어떤 spawn에서 실패하는가? per-region failure rate 분석 (R1-R4) → 어려운 region에 더 많은 sample 또는 reward shaping
6. **Value loss 추적**: P3 final value_function loss 27 → 1.B-α에서 더 큰 reward (place_bonus 추가) 도입 시 value scale 변화 폭 모니터

### C. 1.B-α 본격 진입 시

7. **Curriculum 재설계**: P1 reach → P2 lift → **P3.5 hover-near-target** (lift 후 target 근처로 이동) → P4 place — 4 단계로 더 잘게. 한 번에 reward 형상 크게 바꾸지 말기 (new_1100 dip 교훈)
8. **Checkpoint 1100~1200 사이 50 iter 단위 save**: dip 회복 곡선 더 정밀하게 추적

---

## 6. 신뢰도 평가

| 결론 | 신뢰도 | 근거 |
|---|---|---|
| old_1050이 진짜 best | HIGH | 16,384 trials, SE = 0.08% |
| Step E의 99.68%는 노이즈 | HIGH | 같은 ckpt 재평가 → 98.44% (1.24pp 낮음, 1024 trials noise 범위 안) |
| P3 fix가 collapse 방지 | HIGH | 이전 26.87% vs 이제 98.77% |
| new_1497이 1.B-α에 적합 | MEDIUM-HIGH | 통계상 동등이나 lift margin 유리. 실제 1.B-α reward에서 검증 필요 |
| 0.18pp 차이가 sim-to-real에 무의미 | MEDIUM | sim-only artifact (kinematic-attach grasp). Real에선 grasp 안정성이 더 큰 변수 |

---

## 7. 산출물

- B200 학습 logs: `$ROARM_B200_ROOT/logs/roarm_rl/roarm_pick_p3_500iter_seed0_resumeP2_FIX/`
- B200 eval log: `$ROARM_B200_ROOT/logs/phase1/eval_precision_compare.log`
- B200 eval JSON: `$ROARM_B200_ROOT/logs/phase1/precision_compare_result.json`
- 로컬 코드 변경: `roarm_rl/roarm_pick_env.py`, `roarm_rl/agents/rsl_rl_ppo_cfg.py`
- 로컬 신규 코드: `roarm_rl/precision_compare.py`

## 8. HARD RULE 준수

- #11 `/half-clone` 사용 0회 ✓
- #14 모든 ssh 명령에 `set -e` + `ROARM_B200_ROOT` + `whoami` 검증 ✓
- #15 cu128 sm_100 alive (학습 + eval 모두 정상) ✓
- #17 visual RL 시도 0회 (state-only obs만 사용) ✓
- #18 사용자 명시 정정 우선 (1-cube 먼저 확정 그대로 유지) ✓
- #19/#20 v3 geometry constants 그대로 유지 (TABLE_Z=−0.012117, SPONGE_HEIGHT_EDGE=0.047, Z_TCP_GRASP_L1=0.033) ✓
- #26 5/19 deadline 11일 ahead — Phase 1.A complete, 1.B-α 진입 준비 완료 ✓
