# Phase 1.B-α P6v9 결과 — ManiSkill-strict 2-channel fix는 정확히 작동했지만 hover policy가 stage 2로 평형 이동 (FAIL 분기 C 확정, stage 2 reward farming 신규 진단) (5/16)

## TL;DR

- 🟢 **Patch 정확히 작동**: ManiSkill-strict 2-channel fix (① stage 3 = xy 30mm AND z 25mm 분리, ② ungrasp_signal `~is_grasped → 1.0` force-set) 그대로 작동 — `is_on_target_rate` 1000 iter 내내 ~0 (strict gate 99.99% fail = hover에서 strict 진입 차단 정확).
- 🔴 **FAIL 분기 (C) hover persist 확정**: stage4=0%, z_offset=67mm > 60mm threshold. iter 50에서 hover policy로 즉시 평형 → 950 iter 동안 정체 (Mean reward iter 50 804 → iter 999 811, +0.8% 마진).
- 🚨 **신규 root cause (P6v9-specific)**: stage 3 차단했더니 hover policy가 **stage 2 reward farming** (grasped + hover at d=120mm)으로 평형 이동. `stage2_grasp_frac` P6v8 0.22 → P6v9 **0.865 (4×)**. Stage 2 reward 4.48/step at d=120mm ≈ saturate (place_progress 0.45). d 줄일 incentive 약함.
- ⚠️ **디자인 doc 예측 -56pp 빗나감**: "hover → 240 reward (-75%)" 예측. 실제 811 (-19%). 산수 오류 = stage 2 frac을 P6v8의 0.22로 가정. 정정: P6v9 stage 3 차단 → policy hover behavior 유지하면서 stage 2로 흡수.
- ✅ **산수 정합 검증**: iter 999 reward 예측 813 vs 실제 811 (±0.2%) = stage 2 4.48/step × 0.865 × 199 step = 770 + stage 1 47 + action_penalty -5.5 = 813. 산수 모델 정확.
- ✅ **HARD RULE 준수**: #8 archive 1단계 (5/09 → MEMORY_archive_20260516.md), #11 /half-clone X 0회, #14 fail-fast guard, #15 cu128 sm_100 alive (학습 6:50 wall = 추가 검증), #17 state-only 28-dim, #18 사용자 명시 4 결정 보존 (γ transport shaping 제안은 confirm 필요), #19/#20 그대로, #26 5/19 deadline **3일 ahead**.

## Iter Trend Table (정량 진단)

| Metric | iter 0 | iter 50 | iter 100 | iter 200 | iter 500 | iter 999 | 해석 |
|---|---:|---:|---:|---:|---:|---:|---|
| Mean action std | 1.00 | 0.98 | 0.98 | 0.98 | 0.97 | 0.94 | reset_std 1.0 적용 ✓, gentle decay (P6v8 0.86 대비 마진 +9%) |
| **Mean reward** | 18.3 | 804.8 | 799.5 | 797.5 | 787.2 | **811.3** | iter 0→50 점프 후 flatline (학습 정체) |
| tcp_sponge_dist (m) | 0.180 | 0.026 | 0.026 | 0.025 | 0.026 | 0.024 | reach 학습 안정 ✓ |
| sponge_target_dist (m) | 0.181 | 0.121 | 0.122 | 0.124 | 0.129 | 0.117 | transport 정체 ~120mm (target 도달 못 함) |
| sponge_height (m) | 0.054 | 0.088 | 0.088 | 0.092 | 0.101 | 0.091 | hover ~90mm 유지 (target z=11mm와 80mm gap) |
| grasped_frac | 0.047 | 0.856 | 0.854 | 0.857 | 0.860 | **0.865** | grasp 학습 + 안정 ✓ |
| gripper_open_rate | 0.559 | 0.067 | 0.064 | 0.062 | 0.066 | **0.061** | 🔴 closed saturate (release 학습 0) |
| sponge_stable | 0.630 | 0.259 | 0.257 | 0.244 | 0.149 | 0.143 | ↓ sponge 점점 더 흔들림 |
| near_target_rate (3D 100mm) | 0.055 | 0.627 | 0.626 | 0.599 | 0.533 | 0.611 | 60% env가 100mm sphere 안 |
| is_success_zone (3D 50mm) | 0.001 | 0.010 | 0.009 | 0.004 | 0.0004 | **0.090** | ⚠️ iter 999만 200× 점프 (1 iter noise 의심) |
| **is_on_target (strict xy+z)** | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | **0.0001** | 🔴 1000 iter 내내 ~0 = strict gate 정확 작동 + 정책 통과 0 |
| **xy_offset_mean (m)** | 0.172 | 0.090 | 0.091 | 0.090 | 0.086 | **0.078** | gentle ↓ but 78mm >> 30mm threshold |
| **z_offset_mean (m)** | 0.030 | 0.065 | 0.065 | 0.069 | 0.078 | **0.067** | 🔴 hover 67mm 정체 (>>25mm threshold) |
| jackpot_fire | 0 | 0 | 0 | 0 | 0 | 0 | success_jackpot=0 disabled (P6v9 design) ✓ |
| stage1_reach_frac | 0.953 | 0.144 | 0.146 | 0.143 | 0.140 | 0.135 | grasp 후 ↓ (정상) |
| **stage2_grasp_frac** | 0.047 | 0.856 | 0.854 | 0.857 | 0.860 | **0.865** | 🚨 **87% hold dominant (P6v8 0.22 대비 4×)** |
| stage3_neartgt_frac (strict) | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0001 | 🔴 strict gate 0 |
| **stage4_success_frac** | 0 | 0 | 0 | 0 | 0 | **0** | 🔴 1000 iter 0 fire |
| ungrasp_signal | 0.980 | 0.190 | 0.190 | 0.187 | 0.185 | 0.183 | iter 0 high (random init) → 19% closed gripper |
| static_signal | 0.628 | 0.253 | 0.252 | 0.239 | 0.163 | 0.154 | ↓ stable ratio 감소 |
| action_penalty | -0.028 | -0.028 | -0.028 | -0.028 | -0.027 | -0.028 | flat |

## P6v8 vs P6v9 직접 비교

| Metric | P6v8 iter 999 | P6v9 iter 999 | Δ | 해석 |
|---|---:|---:|---:|---|
| Mean reward | 1005.5 | **811.3** | **-194 (-19%)** | hover policy의 reward source 일부만 차단 |
| action_std | 0.86 | 0.94 | +0.08 | reset_std 1.0 효과 (P6v8 0.86 → 1.0 → 학습으로 0.94 decay) |
| grasped_frac | 0.863 | 0.865 | +0.002 | grasp 동일 |
| **gripper_open_rate** | 0.061 | 0.061 | 0 | 🔴 release 학습 진행 0 |
| sponge_target_dist | 0.120 | 0.117 | -3mm | marginal |
| sponge_height | 0.088 | 0.091 | +3mm | hover slightly higher |
| **stage2_grasp_frac** | 0.220 | **0.865** | **+0.645 (4×)** | 🚨 stage 3 차단 → stage 2로 흡수 |
| stage3_neartgt_frac | 0.645 | 0.0001 | -0.645 | strict 정의 적용 (정확) |
| stage4_success_frac | 0 | 0 | 0 | 둘 다 0 |
| is_success_zone (50mm 3D) | 0.0088 | 0.090 | +0.081 | iter 999 1 iter noise 가능성 |

## 산수 cross-verify — Mean reward 813 ≈ 811 (P6v9 신규 진단 정합)

P6v9 iter 999에서:
- Stage 1 reach: 199 step × 0.135 frac × 2(1-tanh(5×0.024))=1.76 → **47 reward**
- **Stage 2 hold**: 199 step × 0.865 frac × (4+1-tanh(5×0.117))=4.48 → **770 reward** ← 🚨 **주범**
- Stage 3/4: 0
- Action penalty: -0.028 × 199 = -5.5
- **Total = 47 + 770 - 5.5 = 813 reward** ≈ 실제 **811** (±0.2% 오차)

→ 산수 모델 정확. P6v9 정책 = "잡고 d=120mm에서 hover"가 stage 2 reward farming으로 평형. ManiSkill stage 2 reward의 place_progress component (`1-tanh(5×0.117)`)가 d 120mm에서 0.45 → stage 2 reward 4.48/step (max 5/step의 90% saturate). **d 줄일 마진 = 4.48 → 4.85 (=+9%)에 불과**.

## 디자인 doc 예측 정정 — 산수 hidden flaw 분석

Design doc Channel 1 산수 (`240 reward at hover`)의 가정:
- "hover policy → stage 3 fire 0 → stage 1만 fire ~0.2/step × 199 ≈ 40 reward"
- 가정: hover 시 grasped=False → stage 1만 fire

**실제 P6v9**: hover 시 **grasped=True (87%) → stage 2가 fire**. Design doc는 P6v8 stage_2 frac (0.22)을 그대로 P6v9에 가정 → P6v9에서 stage 2 frac이 0.865로 4× 증가하는 걸 예측 못 함.

**Hidden flaw**: P6v8에서 stage_2 frac이 낮은 이유 = stage 3이 fire하면 stage 2 mask 됨 (in_stage2 = is_grasped & ~in_stage3 & ~in_stage4). P6v9에서 strict stage 3이 fire 안 함 → mask 해제 → stage 2 frac 0.865로 흡수.

→ **단일 channel cut으로 multi-stage reward landscape를 차단하기 어려움**. Stage 3 봉쇄가 stage 2 hover 평형으로 mass redistribution.

## 분기 (C) FAIL hover persist — 다음 fix 후보

| Fix | 설명 | 산수 영향 | 위험 | HARD RULE #18 |
|---|---|---|---|---|
| (α) reset_std 1.30 | exploration boost (design doc default fallback) | 약 — reward landscape 변화 없음 | low (P6v2 1.5 stable empirical) | OK |
| (γ) transport shaping ⭐⭐⭐ | stage 2/3 reward에 `+2*(1-tanh(5*d_sponge_target))` 추가 | **강 — stage 2 reward d=120mm 4.48 → 5.4, d=30mm 4.85 → 6.5 = 마진 +9% → +20% ↑** | low (reward 구조 변경) | ⚠️ **confirm 필요** (사용자 명시 fallback에 없음) |
| (β) stage 4 50mm continuous | stage 4 zone 안 continuous gradient | 중 | low | OK |
| (α') stage 2 reward weight 2× | `stage2_r = 4 + place_progress * 2` (place_progress 가중치 ↑) | 중 — d=120mm 4.9, d=30mm 5.7 | low | ⚠️ minor confirm |
| (δ) actor.6.bias[5]=0 + reset_std 1.5 | gripper bias reset (P5 패턴) + 강한 exploration | 중 (정책 partial reset) | medium (P5/P6v1 학습 일부 retreat) | OK |

## 권장 P6v10 plan (사용자 confirm 필요)

**제 권장**: γ transport shaping 단독 (또는 γ + reset_std 1.20 결합).

**근거**:
1. P6v9 진단 = "policy is safe in stage 2 hover, transport gradient too weak". reset_std boost는 root cause (reward landscape weakness)를 안 건드림.
2. γ는 stage 2/3 reward 안에 dense d_sponge_target gradient 형성 = transport 방향 학습 직접 유도. P6v9 stage 2 4.48/step at d=120mm가 transport (d→30mm) 시 5.4 → 6.5로 단조 증가 = PPO가 transport advantage 인식.
3. 단점: reward 구조 변경 = HARD RULE #18 사용자 명시 confirm 필요. P6v9까지의 ManiSkill-strict 원리는 유지하면서 dense gradient만 추가하는 형태로 보수적 변경.

**대안 (사용자 γ confirm 거부 시)**: α' stage 2 weight 2× (`stage2_r = 4 + 2*place_progress`) — 더 보수적 변경 (stage 2 max 5 → 6, gradient 2×).

## 다음 세션 즉시 명령 (사용자 γ confirm 시)

```bash
# 1. Local roarm_stack_env.py L584 + L578 패치
#    Stage 2: stage2_r = 4.0 + place_progress  →  stage2_r = 4.0 + place_progress + 2.0 * (1.0 - torch.tanh(5.0 * d_sponge_target))
#    Stage 3: stage3_r = 6.0 + 0.5 * ungrasp_signal + 0.5 * static_signal  →  + transport bonus
# 2. md5 verify + sanity 64env × 2 iter
# 3. B200 launch_p6v10.sh: resume p6v9 model_999, reset_std 1.0 (또는 1.20), entropy 0.001, episode 2.0s, experiment p6v10_gamma_transport_shaping
# 4. ~7min 후 polling, 분기 (A/B/C/D) 판정
```

## Falsifiability (P6v10 γ 적용 시 iter 999)

| 분기 | 조건 | 평가 기준 |
|---|---|---|
| **(A) ⭐⭐⭐⭐ SUCCESS** | stage4>5% AND on_target>10% AND z_off<30mm | transport + release 학습 |
| **(B) PARTIAL** | 5%>stage4>1% AND z_off<50mm AND on_target>3% | dense gradient 작동 but precision 부족 |
| **(C) FAIL hover persist** | stage4<1% AND z_off>60mm | γ가 hover penalize 못 함 → δ (bias reset) 고려 |
| **(D) FAIL slow** | stage4<1% AND z_off<50mm AND on_target>3% | progress 있지만 1000 iter 부족 → episode 400 또는 2000 iter |

## HARD RULES 준수

- **#8**: archive 1단계 (5/09 → MEMORY_archive_20260516.md), limit 5 violation 4개 잔존 (다음 archive 후보 5/11, 5/12).
- **#11**: /half-clone X 0회.
- **#14**: fail-fast guard 모든 ssh (`set -e; [[ -z "$ROARM_B200_ROOT" ]] && exit 1; [[ "$(whoami)" != "sogang_jhki" ]] && exit 1`).
- **#15**: cu128 sm_100 alive (P6v9 학습 wall 6:50 성공 = 추가 검증, NVML driver mismatch warning 무시 가능 확정).
- **#17**: state-only 28-dim only.
- **#18**: 사용자 명시 4 결정 (target / gravity / 28-dim / P4-P5-P6 phase) 보존. **γ transport shaping은 신규 reward 구조 변경 = 사용자 confirm 필요**.
- **#19/#20**: sponge edge-stand 47mm + tower geometry 그대로.
- **#26**: 5/19 deadline **3일 ahead** (1 iter ~7min 학습이라 1-2 시도 가능, P6v10 + P6v11 escalate room 있음).

## Reference URLs

- [ManiSkill stack_cube.py main](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/stack_cube.py) — Stage 2 `4 + place_reward` (place_reward = `1-tanh(5*d_cubeA_to_goal)`) 원본
- 본 P6v9 sanity test (5/15) + iter trend 추출 (5/16)

## B200 Inventory

- `logs/roarm_rl/p6v9_maniskill_strict_resumeP6v8/` — 22 ckpts (model_0, 50, 100, …, 999) + nn/
- `logs/phase1Balpha/train_p6v9.{out,err}` — 39K line train log
- `$ROARM_B200_ROOT/launch_p6v9.sh` — 적용된 launch script
- roarm_stack_env.py md5 `f43decac350acc534da1e3d5d26d2e09` (P6v9 patch) — local↔B200 일치
- train_ppo.py md5 `4fb9ff1cff92a50cb7c80338041f8263` — local↔B200 일치
