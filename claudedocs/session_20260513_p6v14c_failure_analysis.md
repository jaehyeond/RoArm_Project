# P6v14c Phase 0a' Failure Analysis (5/13 evening session)

**Run**: `logs/roarm_rl/p6v14c_phase0a_prime_hover_resumeP6v14a/` (500 iter, ~3.5min B200)
**Resume**: P6v14a/model_499.pt (release-aware policy)
**Config**: pregrasp_hover=ON, post_grasp_cap=3.0, annulus 0.05-0.07, entropy 0.003, reset_std 2.0
**Outcome**: **FAIL** — iter 50 sanity gate FAIL, ~100 iter 만에 P6v14b와 동일한 lock-in 도달

---

## Executive Summary

**P6v14c는 P6v14b 실패의 변형이 아니라, "starting policy was good but RL destroyed it" 패턴**. iter 0에서 **stage4 36.5%, jackpot 1.7% fire** 달성 → iter 1에서 PPO update 한 번에 gripper_open 0.78→0.21 collapse. Bridge 자체는 작동했으나 **RL의 advantage 계산이 starting policy를 1 iter 만에 무너뜨림**. Reward shape engineering으로 풀 수 없는 영역 확정 → BC pivot 필요.

---

## Metric Trajectory (Critical Iters)

| iter | stage4_succ | stage2_grasp | jackpot | grasped | gripper_open | upright |
|---|---|---|---|---|---|---|
| **0** | **0.3653** | 0.0496 | **0.0170** | 0.0807 | **0.7835** | 0.4424 |
| **1** | 0.3446 | 0.3760 | 0.0001 | 0.6696 | 0.2100 | 0.2770 |
| 5 | 0.1465 | 0.6278 | 0.0000 | 0.8317 | 0.1207 | 0.2685 |
| 10 | 0.0037 | 0.7637 | 0.0000 | 0.8496 | 0.1119 | 0.2749 |
| 20 | 0.0030 | 0.7471 | 0.0000 | 0.8578 | 0.1049 | 0.2812 |
| 50 | 0.0039 | 0.7225 | 0.0000 | 0.8724 | 0.0920 | 0.2889 |
| 100 | 0.0027 | 0.6815 | 0.0000 | 0.8882 | 0.0783 | 0.3371 |
| 200 | 0.0029 | 0.6461 | 0.0000 | 0.8929 | 0.0749 | 0.3913 |
| 300 | 0.0055 | 0.8390 | 0.0000 | 0.8934 | 0.0783 | 0.3857 |
| 400 | 0.0047 | 0.8651 | 0.0000 | 0.8996 | 0.0742 | 0.2982 |
| **499** | 0.0105 | 0.8735 | 0.0000 | 0.9132 | 0.0692 | 0.1409 |

CSV: [p6v14c_data/p6v14c_metrics.csv](p6v14c_data/p6v14c_metrics.csv) (500 iter full trajectory)
Raw log: [p6v14c_data/train_p6v14c.out](p6v14c_data/train_p6v14c.out)

---

## 성공한 부분 (왜 성공했는지)

### ✅ S1: pregrasp_hover bridge — iter 0 stage4 36.5%

**Quantitative evidence**: iter 0 stage4_success_frac = **0.3653**, jackpot_fire_rate = **0.0170** (4096 env × 200 step × 24 substep / iter 중 **약 1.7% fire**).

**Why it succeeded**:
- P6v14a policy obs distribution: TCP+5cm 위치, gripper q=0.8 (closed-hold), sponge in-hand
- P6v14c initial state: TCP+5cm 위치 (동일), gripper q=0.0 (OVERRIDE OPEN), sponge on table near target
- **공통**: TCP가 target 근처 5cm 위 → P6v14a의 "descent to target then release" motor primitive 그대로 실행
- 결과: P6v14a가 학습한 "open gripper near target = release"를 sponge가 in-hand 아닐 때도 적용
- Sponge가 이미 target xy 근처 (annulus 0.05-0.07) → upright 떨어지면 stage 4 condition 충족

**Conclusion**: **Bridge mechanism 본질적으로 작동**. Pre-launch math가 가정한 path B (release path)가 iter 0에 실제로 fire함. 문제는 PPO가 이를 보존 못 한 것.

### ✅ S2: Bug #2 fix (upright check) — 정확히 작동

**Quantitative evidence**: 
- iter 499: stage2_grasp_frac=0.87 (deep farming) but **stage4_success_frac=0.011** (tipping success로 위장 안 됨)
- iter 499: upright_rate=0.14 (sponge가 누워있어도) stage4 success 0.011 — Bug #2 fix 없었으면 falsely fire

**Why it succeeded**:
- `success_now = is_on_target & upright` AND-gate
- sz_world_z = 1 - 2*(qx²+qy²) > 0.90 threshold 적용
- 옆으로 누운 sponge → upright fail → stage4 success 안 fire

**Validation**: Phase 0b P6v14b가 옆으로 눕히고 stage4 success 받는 8th farming의 9th 변형으로 도피했다면 이번 P6v14c iter 499에도 그 패턴 나왔을 것. Bug #2 fix가 그 도피로를 차단함을 정량 검증.

### ✅ S3: post_grasp_cap=3.0 — P6v14b 8th farming "grasp+move-away" 차단

**Quantitative evidence**:
- P6v14b stage2_grasp_frac final: 0.871 (cap 없음, stage 2 reward 5.28/step)
- P6v14c stage2_grasp_frac final: 0.874 (cap 3.0)
- 비슷한 frequency이지만 P6v14c는 reward 절대값 ~57% 감소 → wasted reward 줄임

**Why it partially succeeded**:
- "grasp+move-away" (d > 0.1 zone)이 cap 적용으로 ~3.0/step bounded
- P6v14b "outside cap zone에서 자유롭게 5.28/step" → P6v14c "어디서든 3.0/step"
- 변형이지만 같은 attraction basin

**왜 결국 실패했는지**: cap=3.0이 stage 1 max(2.0)보다 +1.0 더 높음. **Cap < stage 1**가 아니라 **Cap > stage 1** → attraction basin 차단 X. Path A''=583 vs Path B=1597 margin +174%는 episode-sum 비교지 PPO advantage 비교 아님.

---

## 실패한 부분 (왜 실패했는지)

### ❌ F1: PPO update 1 iter 만에 starting policy 파괴 — **가장 critical**

**Quantitative evidence**:
- iter 0 → iter 1 (1 update): gripper_open_rate **0.7835 → 0.2100** (-74% in 1 iter)
- iter 0 → iter 1: grasped_frac **0.0807 → 0.6696** (+730%)
- iter 0 → iter 5: stage4_success **0.3653 → 0.1465** (-60%)
- iter 0 → iter 10: stage4_success **0.3653 → 0.0037** (-99%)

**Root cause**: PPO advantage 계산이 reward gradient를 즉시 grasp 방향으로 attract:
- Stage 1 (descent): reach_r ~1.0-2.0/step
- Stage 2 (grasp+hold): cap=3.0/step
- Stage 3 (drag to target xy): ~1.0/step + small bonus
- Stage 4 (release): jackpot=150 one-shot + post-jackpot continuation

PPO advantage A(s,a) = Q(s,a) - V(s):
- iter 0에서 grasp 시도 → Q jump 1.0→3.0 = **+2.0 marginal advantage**
- iter 0에서 release 시도 → Q stays 1.0 unless jackpot fires (1.7% chance) → +148.4 jackpot OR -0.0
- **1.7% × 148 = +2.5 expected advantage but variance 거대**
- PPO는 high-variance signal (release jackpot) vs low-variance signal (grasp +2.0)에서 후자로 lock-in

**Why RL fundamentals problem, not reward shape**:
- Cap 낮추기 (cap=0.5) → grasp jump +1.0 → still attractive, 또한 grasp gradient 자체도 약해져 학습 어려움
- Bonus-only (transition +5) → 일회성 spike but **sustained reward 없으면 sub-optimal trajectory에서 멈춤**
- 본질적 issue: **release path는 chain of 5+ sequential rare events** (descent → grasp → drag → align → release). 각 transition이 rare → expected return 매우 낮음.

### ❌ F2: stage2_grasp_frac trajectory — non-monotonic but ending at 0.87

**Quantitative evidence**:
- iter 0: 0.0496 (거의 안 잡음)
- iter 1: 0.376 (+656% jump in 1 iter)
- iter 10: 0.764 (+93% in 9 iters)
- iter 200: 0.646 (감소 — exploration 시도)
- iter 300: 0.839 (+30% — 다시 farming lock-in)
- iter 499: 0.874

**Why this trajectory**:
1. **iter 0-10**: PPO greedy grasp lock-in (reward gradient 우세)
2. **iter 200**: entropy_coef 0.003 + reset_std 2.0의 exploration 시도. 일부 env가 release 시도하니 stage2 frequency 감소.
3. **iter 300+**: jackpot fire 0 → release exploration이 sustained reward 없어 다시 grasp farming으로 돌아옴.

**Conclusion**: Entropy 증가가 exploration window 잠시 열지만, **release path가 PPO advantage에서 잡히지 않으니** 결국 grasp 기본값으로 복귀.

### ❌ F3: jackpot_fire_rate trajectory — iter 0 0.017 → iter 1+ 0.0

**Quantitative evidence**:
- iter 0: 0.0170 (P6v14a behavior 잔존)
- iter 1: 0.0001 (-99.4% in 1 iter)
- iter 2-499: 0.0000 with very occasional 0.0001 spike

**Why catastrophic**: PPO 단 1 update에서 release behavior가 사라짐. iter 0에 **이미 release를 demonstrate**했음에도 PPO는 그 path를 advantage 낮다고 판단 → policy distribution에서 deweighting.

**Critical 의미**: P6v14a policy의 release behavior가 "P6v14c initial state에서 valuable"이라는 정보가 PPO에 전달 안 됨. **value function update가 1 iter 만에 grasp-favor로 shift**.

### ❌ F4: upright_rate 후반 collapse 0.39 → 0.14 (iter 200 → 499)

**Quantitative evidence**:
- iter 200: 0.3913
- iter 400: 0.2982
- iter 499: 0.1409

**Why**: stage2_grasp_frac 후반 saturation (0.87) → grasp-then-move 빈도 증가 → sponge 옆으로 누이는 빈도 증가. Bug #2 fix가 이 누운 sponge를 stage4 success로 false fire하지 않게 막음 (=S2). 하지만 underlying tipping 빈도 자체 증가.

**Conclusion**: Bug #2 fix는 success metric 무결성 보장. 하지만 **policy가 sponge를 옆으로 눕히는 행동을 학습** — 우물정자 # tower에 부적합.

---

## 8th Farming Pattern — 정확한 형태 확정

P6v14b → P6v14c 둘 다 동일 attraction basin으로 수렴. 정량 분석 후 **정확한 형태**:

> **"Sustained grasp-hold in any reward zone"** — sponge in hand 상태로 어디든 (cap zone 안이든 밖이든) 머물면 PPO가 stage 2 reward를 sustained advantage source로 인식. Release transition은 jackpot fire 아니면 reward 0 → PPO advantage에서 invisible.

**왜 이전 8th farming variant ("grasp+move-away" with stage2=5.28)와 본질 같음**:
- P6v14b: cap 없으니 어디든 farm, reward 5.28/step
- P6v14c: cap 3.0이지만 어디든 farm, reward 3.0/step
- **차이는 절대값만, 패턴 동일**
- 이것이 "9th farming"이 아닌 **"8th farming variant with bounded magnitude"**

---

## RL Fundamentals Problem — 왜 reward shape engineering으로 못 푸나

**4회 시도 패턴**:
| Run | Approach | Outcome |
|---|---|---|
| P6v14a | Pre-grasp specialist (sponge in hand, release only) | SUCCESS in narrow domain (release rate ~50%) |
| P6v14b | Cold-start full chain | FAIL (1000 iter, jackpot 0, gripper_open collapse in 5 iter) |
| P6v14c | P6v14a resume + pregrasp_hover bridge + post_grasp_cap | FAIL (iter 0 stage4 0.37, iter 1 collapse, iter 499 stage4 0.01) |

**Conclusion**: 
- **Reward shape engineering 한계 도달**: cap 절대값 조정 (0, 3.0, 5.28) 모두 같은 attraction basin
- **Marginal advantage 분석**: stage 1→2 transition은 항상 +1~2 marginal positive. Release path는 rare sparse jackpot에 의존 → PPO advantage 0 averaged.
- **RL exploration starvation**: 5-step sequential rare events (descent → close → grasp → drag → release) 각 transition rare → joint probability ~0.

**왜 RL fundamentals problem**:
- PPO는 advantage-based on-policy. **sparse multi-step task에서 정의상 어려움**.
- Sparse reward를 dense로 변환하는 reward shaping은 **shaping reward 자체가 attraction basin** 됨 (potential-based reward shaping 이론).
- 본질적 해결: **Demonstrations 사용** (BC + RL hybrid) — 이는 NVIDIA Isaac Lab 공식 stack task에서도 BC만 사용하는 이유.

---

## Bridge to Option D (BC Pivot)

**BC가 본질적으로 다른 이유**:
1. **Supervised**: advantage 게임 안 함. Demonstration trajectory를 직접 mimicking.
2. **Multi-step trajectory 일관성 보장**: Demo가 descent → grasp → drag → release 전체 chain → BC policy가 chain 전체 학습.
3. **PPO finetune 시 starting point 안정**: BC가 stage 4 reach 가능한 policy 만든 후 PPO는 미세조정만. Initial advantage estimate가 grasp-only가 아니라 full-chain 기준 → DAPG/AWAC behavior cloning regularizer로 forgetting 방지.

**Why this should work given P6v14c data**:
- iter 0 jackpot 0.017 = P6v14a policy가 P6v14c environment에서 stage4 reach 가능 증명
- BC가 demo로부터 P6v14a-like full-chain policy 학습 → **iter 0 stage4 0.5+** (P6v14a보다 풍부한 demo로)
- PPO finetune with DAPG → iter 0의 좋은 starting point가 **catastrophic forgetting 안 됨** (DAPG가 BC loss 유지)

---

## Lessons Learned (Hard Rule 후보)

1. **iter 0 메트릭은 starting policy의 진짜 성능** — iter 50+에서 sanity gate 만으로 RL 진단 부족. **iter 0 jackpot fire rate가 BC bridge 가치 정확히 측정**.
2. **PPO 1-iter update가 starting policy 파괴 가능** — high-quality starting policy를 갖고도 sparse multi-step task에서 RL이 망가뜨림.
3. **Reward cap < stage 1 max 필요조건이지 충분조건 아님** — cap=3.0 > stage 1 max (2.0)이 farming 유도하지만, cap=0.5라도 release path가 PPO advantage에 보이지 않으면 다른 attractor로 lock-in.
4. **Marginal advantage 분석 = episode-sum 분석보다 PPO 행동 예측에 정확** — Pre-launch math의 episode-sum B/A''=+174%가 무력함을 정량 검증.

---

## Decision: Option D BC Pivot — 시작

(별도 문서로 step-by-step plan 작성됨: `session_20260513_option_d_bc_pivot_plan.md` — 작성 예정)
