# Phase 1.B-α P6v11 결과 — C (β+δ+γ continue + reset_std 1.5 escalate) 부분 학습 BUT (C) BIAS RE-SATURATION 재현, stage 4 release 0 (5/12)

## TL;DR

- 🟡 **P6v11 학습 COMPLETE** PID 2175407, **wall 6:50** @ 243K steps/s (4096 envs × 1000 iter × 199 step = 98.3M timesteps), 21:08~21:15Z, 23 ckpts. β + δ + γ continue + reset_std 1.5 escalate combo.
- 🟢 **β + δ + γ 패치 검증 OK (iter 0 log)**: reset_std P6v10 [0.92, 0.87, 0.93, 0.82, 0.93, 0.88] → [1.5×6] ✓, **bias[5] -0.0242 → 0.0** ✓ (other dims 보존). `success_jackpot 5.0` env code (md5 `ec82ed18313cb377214773afa00b9696` local↔B200 일치).
- 🟢 **부분 학습 진전 (z/xy 모두)**: sponge_target_dist **0.101→0.094 (-7mm)**, **xy_offset 0.073→0.067 (-6mm)**, **z_offset 0.051→0.048 (-3mm)**, **is_on_target 0.91→1.61% (1.8×)**, **is_success_zone (50mm 3D) 46→53% (+7pp)**.
- 🔴 **(C) BIAS RE-SATURATION 재현 확정 (P6v5에 이어 2번째)**: iter 0 gripper_open=**0.561** ✓ → iter 1 **0.0641** (**1 iter 만에 17× ↓**, P6v5는 50 iter), iter 999 **0.0634**. P5 phase 50 iter → P6v6+ REPLACE tower 1 iter = **더 빠른 re-saturation**.
- 🔴 **β jackpot 5.0 fire 0회 1000 iter 내내** (산수 검증대로): success_now = `is_on_target AND gripper_open AND sponge_stable` joint AND ≈ 0 → reward 효과 0.
- 🔴 **stage4_success 0** (1000 iter 0 fire, place_success 0.0000).
- 🟡 **δ bias reset 효과 0 사전 산수 검증**: P6v10 bias[5]=-0.0242 (P6v5 시점 +0.8446 → 25× 감소). mean action +1.36 close (std 0.88 + P(open)=6.1% z_score 역산) = driving force = weights (not bias). δ reset 0.024 = ε 변화.
- ✅ **HARD RULE 준수**: #8 archive 1단계 (이번 세션 5/11 P6v5 후보), #11 /half-clone X 0회, #14 fail-fast guard 모든 ssh, #15 cu128 sm_100 alive (P6v11 wall 6:50 추가 검증), #17 state-only 28-dim, **#18 사용자 명시 "C 옵션 진행" + reset_std 미선택 → claude critical sanity 후 1.5 escalate (zero-cost δ 발견 evidence)**, #19/#20 sponge edge-stand 47mm + tower geometry 그대로, #26 5/19 deadline **7일 ahead**.

## Iter Trend Table

| Metric | iter 0 | iter 1 | iter 50 (est) | iter 500 (est) | iter 999 | Δ vs P6v10 iter 999 | 해석 |
|---|---:|---:|---:|---:|---:|---:|---|
| Mean action std | **1.50** | - | - | - | **1.34** | +0.45 (P6v10 0.89) | **std reset 1.5 → 1.34 = +50% maintained exploration** ✓ |
| Mean reward | 18.82 | 120.94 | ~683 | ~1030 | **1058.59** | +3.6 (P6v10 1055) | 거의 동일 (P6v10 saturate point) |
| tcp_sponge_dist | 0.178 | - | - | - | 0.025 | -2mm marginal | reach 안정 |
| **sponge_target_dist** | 0.181 | 0.207 | ~0.150 | ~0.105 | **0.094** | 🟢 -7mm (P6v10 0.101) | δ + std boost로 transport 추가 학습 |
| **sponge_height** | 0.054 | 0.089 | ~0.080 | ~0.075 | **0.071** | 🟢 -4mm (P6v10 0.075) | z drop 약간 ↑ |
| grasped_frac | 0.048 | 0.720 | ~0.85 | ~0.86 | **0.865** | 동일 | grasp 안정 |
| **gripper_open_rate** | **0.561** ✅ | **0.064** ❌ | ~0.065 | ~0.064 | **0.0634** | 동일 🔴 | **1 iter 만에 17× ↓ (C) BIAS RE-SAT** |
| sponge_stable | 0.621 | - | - | - | 0.145 | +0.008 | sponge 흔들림 동일 |
| near_target (100mm 3D) | 0.055 | - | - | - | **0.656** | +0.019pp (P6v10 0.637) | marginal +2pp |
| **is_success_zone (50mm 3D)** | 0.001 | - | - | - | **0.526** | +0.065pp (P6v10 0.461) | 🟢 +7pp |
| **is_on_target (strict)** | 0.0001 | - | - | - | **0.0161** | +0.0070 (P6v10 0.0091) | 🟢 **1.8× ↑** |
| **xy_offset_mean (m)** | 0.172 | - | - | - | **0.0671** | -0.006 (P6v10 0.073) | 🟢 -6mm |
| **z_offset_mean (m)** | 0.030 | - | - | - | **0.0479** | -0.003 (P6v10 0.051) | 🟢 -3mm |
| **jackpot_fire** | 0 | 0 | 0 | 0 | **0** | **0회 1000 iter** | 🔴 산수대로 0 |
| stage1_reach_frac | 0.952 | - | - | - | 0.135 | 동일 | 정상 |
| stage2_grasp_frac | 0.048 | - | - | - | **0.849** | -0.007 | hold 여전 dominant |
| stage3_neartgt_frac (strict) | 0.0001 | - | - | - | **0.0161** | +0.007 (P6v10 0.0091) | 🟢 1.8× |
| stage4_success_frac | 0 | 0 | 0 | 0 | **0** | 동일 🔴 | 0 fire 1000 iter |
| ungrasp_signal | 0.979 | - | - | - | 0.183 | +0.003 | closed gripper saturate |

## 🚨 Critical 진단 (Step-by-step)

### 1. δ bias reset 효과 0 확정 (사전 산수 → 실험 일치)

**사전 산수 (학습 전)**:
- P6v10 model_999 actor.6.bias[5] = **-0.0242**
- P6v5 (5/11) 시점 bias[5] = +0.8446
- mean action 추정: gripper_open_rate=0.061 + std=0.88 → P(action<open_threshold)=6.1% normal z_score=-1.55 → **mean action ≈ +1.36 close-strong**
- bias[5]=-0.0242 → 0.0 reset = action mean +0.024 변화 = ε
- 결론: driving force = **actor weights (not bias)**

**실험 결과**: P6v10 iter 999 gripper_open=0.061 → P6v11 iter 999 0.0634 (Δ +0.002 negligible). 산수 일치.

**Why P6v5 (5/11) bias=+0.84 ↑ P6v10 bias=-0.024**?
- P6v5 = P5 phase, hold-path globally optimal → bias 양수 → close 학습
- P6v6+ = REPLACE tower, Path B (release) globally optimal +42% → bias 학습 신호 약화 → bias drift to 0
- BUT weights는 obs와 coupled → close 학습이 weights에 누적 (1 iter 만에 saturate)

### 2. (C) BIAS RE-SATURATION 분기 재현 (더 빠름)

| Phase | bias reset 후 close 재학습 속도 | 원인 |
|---|---|---|
| **P6v5 (5/11, P5 phase)** | **50 iter** | Hold-path 4-7/step, dense gradient |
| **P6v11 (5/12, P6v6+ REPLACE tower γ patch)** | **1 iter** ⚠️ | Stage 2 4-7/step REPLACE = same dense gradient + reward magnitude 더 큼 |

P6v11 iter 0 → 1 변화:
- gripper_open 0.561 → 0.0641 (17× ↓)
- grasped 0.048 → 0.720 (15× ↑)
- mean reward 18.82 → 120.94 (6× ↑)

**= 1 iter PPO update가 close → grasp → stage 2 reward (4-7/step) capture로 정책 빠르게 학습**.

### 3. β jackpot 5.0 fire rate 0 (산수 검증 일치)

**사전 산수**: success_now = is_on_target × gripper_open × sponge_stable, joint AND probability ≈ 0 (음의 correlation):
- is_on_target=0.0091 × gripper_open=0.061 × stable=0.137 = 0.0000076 (independent 상한)
- 실제 correlation 음 (stage 3 진입 = grasped = closed gripper) → actual ≈ 0

**실험**: jackpot_fire 1000 iter 내내 0. **β 효과 0 확인**.

### 4. 분기 (C)에도 부분 학습 진전 (δ + reset_std 1.5)

P6v11 (D 영역에 가까운 BUT C 분기) 학습 진전:
- sponge_target_dist -7mm, xy -6mm, z -3mm
- is_on_target 0.91% → 1.61% (1.8×)

**원인**: reset_std 1.5 boost 후 entropy decay 1.5→1.34 = **P6v10 시점 0.89보다 큰 std 유지** → action distribution wider → 행동 다양성 → transport 정밀도 ↑.

**그러나 release path는 학습 안 됨** (close-saturated weights × stage 4 sparse fire).

### 5. 핵심 root cause 재진단

**Stage 2 reward 4 + 3*place_progress = 4-7/step이 close 행동의 dense immediate reward로 작동**:
- close gripper → grasp (15%→72% in 1 iter) → stage 2 fire (4-7/step) **즉시 reward capture**
- vs open gripper → no grasp → stage 1 reach (0-2/step) **즉시 reward loss**
- 1-step PPO advantage estimate: close >> open
- → PPO가 1 iter 만에 close 학습 saturate

**REPLACE tower가 stage 2 hold path를 globally optimal 아니게 만들었지만 (Path B +42%)**, **1-step advantage는 여전히 close 우위** → PPO는 1-step gradient 따라 close 학습 → exploration 부족 → Path B 미발견 → stage 4 0.

## 분기 판정 cross-check

| 분기 | 조건 | 실제 P6v11 | Match |
|---|---|---|---|
| (A) ⭐⭐⭐⭐ SUCCESS | stage4>5% AND on_target>10% AND z_off<30mm | 0% / 1.61% / 48mm | ❌ |
| (B) PARTIAL | 5%>stage4>1% AND z_off<50mm | 0 (B 미달) / **48mm ✓** | 거의 (stage4=0) |
| **(C) BIAS RE-SATURATION** | gripper_open<0.10 iter 999 | **0.0634 ✅** | **✅** |
| (D) FAIL slow | stage4<1% AND on_target>5% AND z_off<50mm | 0 / 1.61% (5% 미달) / 48mm | ❌ |

**Verdict (C) BIAS RE-SATURATION 분기 재현**. P6v5 + P6v11 = 2번째.

## 다음 P6v12 fix 후보 (사용자 confirm 필요)

### 진짜 root cause = **stage 2 reward 4-7/step**의 1-step advantage close 우위

| Fix | 메커니즘 | 1-step margin (close vs open) | Path A vs Path B EV | 위험 | HARD RULE #18 |
|---|---|---|---|---|---|
| **(η) stage 2 near-cap + stage 3 transient bonus** ⭐⭐⭐⭐ | sponge_near (100mm) 시 stage 2 = 2.0 cap, stage 3 entry시 transient +10 = 16.5/step | close-near 2.0 vs open-near 1.76 (stage 1) **마진 0.24** OR open-near on_target 16.5 (stage 3) **마진 -14.5** | A=340 / B=1769 (Path B +1429) | medium | confirm 필요 |
| (ζ) stage 2 base ↓ (4→2 keep weight 3) | 2 + 3*p (2-5) — base 약화 | close-near 5 vs open-near 1.76 마진 +3.24 (was +3.83) | A=755 / B=1300 (Path B +545) | low-medium | confirm 필요 |
| (θ) actor weight reset actor.6.weight[5, :]=0 | gripper output mapping zero | mean action drift to 0 → P(open)=50% temporarily | recovery 50-100 iter (P6v5 lesson) | high (grasp 무너짐) | confirm 필요 |
| (ι) action mask curriculum | gripper output random force-sample first N iter | exploration 강제 | recovery 50 iter | high (코드 추가 필요) | confirm 필요 |
| (κ) jackpot 50 escalate (β 강화) | one-time +50 | fire rate 0 (joint AND ≈ 0) → 효과 0 | 변화 없음 | low BUT 무효 | confirm 필요 |

### 권장 P6v12: **(η) stage 2 near-cap + stage 3 transient +10 결합**

**근거**:
1. P6v11 실험 결과 = 1-step PPO advantage가 close 우위 → 이걸 직접 깨는 fix만 효과적.
2. Stage 2 near-cap 2.0 = sponge_near 시 close incentive 약화. Sponge_far 시 4-7 transport gradient 유지 (γ 효과 보존).
3. Stage 3 transient bonus +10 = on_target 진입 시 매 step +10 추가 → close-near 2 vs open-near 16.5 = **1-step margin -14.5 (open >> close)** → PPO가 release path 학습.
4. Path B EV 1769 vs Path A EV 340 = +1429 (5×). PPO가 명확히 release 선택.
5. 5/19 deadline 7일 잔여, P6v11 wall 7분 × 7회 = P6v12-P6v18 escalate room 있음. (η) fail 시 (θ) weight reset, (ι) action mask 순으로 escalate.

**Falsifiability P6v12 iter 999**:
- **(A) SUCCESS**: stage4>5% AND on_target>10% AND z_off<30mm
- (B) PARTIAL: 5%>stage4>1% AND z_off<50mm
- (C) BIAS 재재현: gripper_open<0.10 → 3번째 fail → root cause 재진단 (1-step gradient만 봐서는 부족)
- (D) FAIL slow: stage4<1% AND on_target>5% → margin 충분치 않음 → bonus 강화

### 대안: 더 보수적 (ζ) stage 2 base 4→2

**근거**: γ patch가 P6v9 → P6v10 transport 학습 driver 입증. base 약화는 close-incentive marginal 약화. 위험 작음. 그러나 1-step margin still close (+3.24 → 약함이지만 still positive). (D) FAIL slow 가능성 50%.

### 권장 우선순위: η > ζ > θ > ι > κ

## HARD RULES 준수

- **#8**: archive 1단계 (5/09 처리 완료), 5/11 P6v5 entry archive 후보 (다음 step에서 진행).
- **#11**: /half-clone X 0회 (Stop hook context 75% 안정).
- **#14**: fail-fast guard 모든 ssh `set -e + source env.sh + [[ -z $ROARM_B200_ROOT ]] && exit 1 + [[ user != sogang_jhki ]] && exit 1`. NVML mismatch error stderr 무시 (HARD RULE #15 known issue, P6v9 6:50 + P6v10 7:00 + **P6v11 6:50 wall 성공 = 3회 실증**).
- **#15**: cu128 sm_100 alive (P6v11 학습 = 추가 검증).
- **#17**: state-only 28-dim only.
- **#18**: 사용자 명시 "C 옵션 (β+δ+γ) 진행" + reset_std 미선택 → Claude critical sanity (P6v10 bias[5]=-0.024 발견) 후 1.5 escalate. Evidence 기록 = launch script 코멘트 + 본 doc. **P6v12 (η) stage 2 near-cap + transient bonus = reward 구조 신규 변경 → 사용자 confirm 필요**.
- **#19/#20**: sponge edge-stand 47mm + tower geometry 그대로.
- **#26**: 5/19 deadline **7일 ahead** (오늘 2026-05-12 21:15Z 학습 종료 후). P6v12-v18 escalate room 7회 1000 iter.

## B200 Inventory

- `logs/roarm_rl/p6v11_betagammadelta_combo_resumeP6v10/` — 23 ckpts (model_0/50/.../999) + nn/
- `logs/phase1Balpha/train_p6v11.{out,err}` — train log (out 1.4MB, err 2.2KB)
- `$ROARM_B200_ROOT/launch_p6v11.sh` — β+δ+γ combo + reset_std 1.5 launch script
- `roarm_stack_env.py` md5 `ec82ed18313cb377214773afa00b9696` (P6v11 β jackpot 5.0 patch) — local↔B200 일치
- `train_ppo.py` md5 `4fb9ff1cff92a50cb7c80338041f8263` (unchanged, P6v5에서 `--reset_actor_bias_idx` 이미 구현됨)

## 사용자 confirm 대기 (5/19 deadline 7일 잔여)

1. **P6v12 fix 선택**: (η) stage 2 near-cap + stage 3 transient +10 (강력 권장) / (ζ) stage 2 base 4→2 (보수) / (θ) weight reset (공격) / 다른
2. **(η) 선택 시 transient bonus 값**: 10.0 / 5.0 / 20.0
3. **(η) 선택 시 stage 2 cap 값**: 2.0 / 3.0 (place_progress=0)
4. **resume 위치**: P6v11 model_999 (transport 학습 보존) / P6v10 model_999 (P6v11 std 1.5 disrupt 제거) / from scratch (큰 disrupt)

답변 주시면 즉시 patch → sanity → launch → poll → result 진행.

⚠️ HARD RULE #18 준수: (η)/(ζ)/(θ) 모두 reward/weight 구조 변경이라 confirm 필수.
