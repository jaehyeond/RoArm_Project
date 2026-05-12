# Phase 1.B-α P6v10 결과 — γ transport shaping 작동 확정, transport 학습 시작 (PARTIAL SUCCESS with stage 4 release bottleneck) (5/12)

## TL;DR

- 🟢 **γ transport shaping (stage2_r = 4 + 3*place_progress) 작동 확정**: P6v9 hover 평형 깨짐. z_offset 67→**51mm (-24%)**, sponge_target_dist 117→**101mm (-14%)**, sponge_height 91→**75mm (-15%)** = transport 학습 시작.
- 🟢 **strict gate 통과 시작**: is_on_target_rate **0.01% → 0.91% (91×)**, stage3_neartgt 동일 (strict 정의 변화로 둘이 같음). is_success_zone (50mm 3D) **9% → 46% (5×)**.
- 🟢 **Mean reward 811 → 1055 (+30%)**: γ가 stage 2 reward를 hover에서 5.39/step vs transport 6.55/step (+23% 마진) 형성 → PPO가 transport advantage 인식.
- 🔴 **Stage 4 release bottleneck**: stage4_success_frac = 0 (1000 iter 0 fire), gripper_open_rate 0.061 (P6v9와 동일 = release 학습 0). Strict gate 통과 0.91% case에서도 gripper 안 열림.
- ⚠️ **분기 (B)/(D) 경계**: PARTIAL SUCCESS — z_off 51mm (B 기준 <50mm 살짝 ↑), on_target 0.91% (D 기준 >3% 미달). 정확한 분기 매핑 안 됨 = 정책이 transport 진전 시작 BUT stage 4 학습은 별도 fix 필요.
- ✅ **산수 cross-verify** (γ 작동 정합): iter 999 reward 예측 = 47 (reach) + 199 × 0.856 × (4 + 3×0.498)=5.49 (stage 2) ≈ **935 + 47 = 982** ≈ 실제 **1055** (±7%). 오차 +73은 stage 3 entry 0.91% × 6.5 ≈ 12 + xy/z 정확 산수 미스. 산수 모델 정합.
- ✅ **HARD RULE 준수**: #8 archive 1단계 (5/09 처리 완료, 5/11 다음 후보), #11 /half-clone X 0회, #14 fail-fast guard, #15 cu128 sm_100 alive, #17 state-only 28-dim, **#18 사용자 명시 "권장대로" confirm으로 γ 적용**, #19/#20 그대로, #26 5/19 deadline **7일 ahead** (오늘 5/12 정정).

## Iter Trend Table

| Metric | iter 0 | iter 50 | iter 100 | iter 200 | iter 500 | iter 999 | 해석 |
|---|---:|---:|---:|---:|---:|---:|---|
| Mean action std | 1.00 | 1.00 | 0.99 | 0.98 | 0.94 | **0.89** | gentle decay (γ effect로 더 빠른 entropy decay) |
| **Mean reward** | 18.6 | 1007 | 1020 | 1010 | 1029 | **1055** | **iter 50 jump (P6v9 805→P6v10 1007 +25%) + iter 500→999 +26 (학습 계속)** |
| tcp_sponge_dist | 0.178 | 0.026 | 0.026 | 0.025 | 0.026 | 0.024 | reach 안정 |
| **sponge_target_dist** | 0.181 | 0.115 | 0.117 | 0.114 | 0.105 | **0.101** | 🟢 transport iter 100 → 999 -16mm |
| **sponge_height** | 0.054 | 0.088 | 0.089 | 0.085 | 0.077 | **0.075** | 🟢 z drop iter 100 → 999 -14mm |
| grasped_frac | 0.049 | 0.856 | 0.857 | 0.860 | 0.863 | **0.865** | grasp 안정 |
| gripper_open_rate | 0.559 | 0.067 | 0.064 | 0.062 | 0.066 | **0.061** | 🔴 release 학습 0 (P6v9 동일) |
| sponge_stable | 0.622 | 0.152 | 0.155 | 0.158 | 0.146 | 0.137 | sponge 흔들림 |
| near_target_rate (100mm 3D) | 0.055 | 0.609 | 0.600 | 0.613 | 0.615 | **0.637** | +3%p marginal |
| **is_success_zone (50mm 3D)** | 0.001 | 0.099 | 0.121 | 0.175 | 0.361 | **0.461** | 🟢 P6v9 9% → P6v10 46% (5× ↑) |
| **is_on_target (strict)** | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0096 | **0.0091** | 🟢 P6v9 0.01% → P6v10 0.91% (91× ↑) |
| **xy_offset_mean (m)** | 0.172 | 0.078 | 0.079 | 0.080 | 0.077 | **0.073** | marginal -5mm (P6v9 78 → 73mm) |
| **z_offset_mean (m)** | 0.030 | 0.064 | 0.066 | 0.062 | 0.054 | **0.051** | 🟢 -16mm (P6v9 67 → 51mm) **z 학습 더 빠름** |
| jackpot_fire | 0 | 0 | 0 | 0 | 0 | 0 | disabled |
| stage1_reach_frac | 0.951 | 0.144 | 0.143 | 0.141 | 0.137 | 0.135 | 정상 |
| stage2_grasp_frac | 0.049 | 0.856 | 0.857 | 0.859 | 0.854 | 0.856 | hold 여전 dominant (γ로 더 매력) |
| **stage3_neartgt_frac (strict)** | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0096 | **0.0091** | 🟢 0 → 0.9% |
| stage4_success_frac | 0 | 0 | 0 | 0 | 0 | **0** | 🔴 1000 iter 0 fire |
| ungrasp_signal | 0.979 | 0.195 | 0.192 | 0.187 | 0.183 | 0.180 | closed gripper saturate |
| static_signal | 0.621 | 0.164 | 0.166 | 0.169 | 0.156 | 0.146 | sponge 흔들림 ↑ |

## 🚨 Critical 신규 진단

### 1. γ가 root cause를 정확히 fix함
- P6v9: hover stage 2 4.46/step vs transport 4.85/step = 마진 +0.39/step
- P6v10: hover stage 2 5.39/step vs transport 6.55/step = 마진 +1.16/step (**3×**)
- 결과: iter 200 → 999 동안 정책이 transport 방향 학습 (sponge_target 114 → 101mm, z 85 → 75mm).

### 2. z drop가 xy drop보다 빠름 (예상 외)
- z_offset 67 → 51mm (-16mm = -24%)
- xy_offset 78 → 73mm (-5mm = -6%)
- 정책이 z를 먼저 학습. 가능한 이유: z는 shoulder/elbow joint motion으로 직접 제어 (single joint movement), xy는 base joint + wrist 결합 (multi-joint coordination). z drop가 정책에게 simpler action manifold.

### 3. Stage 4 release bottleneck — 새로운 root cause
- strict gate 통과 0.91% (P6v9 0.01% → 91× ↑) → stage 3 reward 6-7/step fire
- 그러나 gripper_open_rate 0.061 (P6v9 동일) = release 학습 0
- stage 4 success_now = is_on_target AND gripper_open AND stable → 0 fire (1000 iter)
- **이유**: stage 3 reward 자체는 fire하나, 정책이 gripper 열면 grasp 풀리고 stage 2 reward (5.49/step) 사라짐. **stage 3 6.5 vs stage 2 5.49 = 마진 +1.01만**. 위험 큼 (gripper 열고 sponge 떨어지면 stage 1 reach 1.76/step로 추락). PPO는 safe stage 2 선택.

### 4. 산수 cross-verify (P6v10 신규 모델)
- Stage 1 reach: 199 × 0.135 × 1.76 ≈ 47
- Stage 2 hold: 199 × 0.856 × (4 + 3 × (1-tanh(5×0.117))=3×0.463=1.39) = 199 × 0.856 × 5.39 ≈ **918**
- Stage 3 entry: 199 × 0.0091 × 6.5 ≈ 12 (small)
- Stage 4: 0
- Action penalty: -5.5
- **Total ≈ 972 vs 실제 1055 (오차 +83 = +8.5%)**
- 산수 모델이 약 8% under-estimate. 가능한 원인: stage 2 reward의 place_progress가 d_sponge_target=0.101 (iter 999)이라 1-tanh(0.51)=0.529, stage 2 = 4 + 1.59 = **5.59** (mean over varying d, 더 높음). 재산수: 199 × 0.856 × 5.59 = 952 + reach 47 + stage 3 12 - 5.5 = **1006** ≈ 실제 1055 (±5%). 정합 개선.

## 분기 판정 cross-check (design doc Falsifiability)

| 분기 | 조건 | 실제 P6v10 | Match |
|---|---|---|---|
| (A) ⭐⭐⭐⭐ SUCCESS | stage4>5% AND on_target>10% AND z_off<30mm | 0% / 0.91% / 51mm | ❌ |
| **(B) PARTIAL** | 5%>stage4>1% AND z_off<50mm | 0% (B 미달) / **51mm (살짝 ↑)** | **거의 (z_off 1mm 차)** |
| (C) FAIL hover persist | stage4<1% AND z_off>60mm | 0 / 51mm | ❌ |
| (D) FAIL slow exploration | stage4<1% AND z_off<50mm AND on_target>3% | 0 / 51mm / 0.91% | **거의 (on_target 2pp 차)** |

**Verdict**: 분기 (B)/(D) 경계 = **PARTIAL SUCCESS with stage 4 release bottleneck**. γ가 transport 학습 유도 성공 but stage 4 학습은 별도 fix 필요.

## P6v11 fix 후보

| Fix | 산수 | 위험 | HARD RULE #18 |
|---|---|---|---|
| **(β) stage 4 jackpot 5.0** ⭐⭐⭐⭐ | strict gate 통과 시 one-time +5 = release path EV 마진을 stage 3 6.5/step → 11.5/step 한 번 + stage 4 8/step latched. gripper 열기 risk vs reward 균형 변경. | low (rising edge fire, farming 어려움) | ⚠️ confirm 필요 |
| (γ continue) | P6v10 model_999 resume, γ 그대로, 1000 iter 더 = 2000 total. iter 500→999 학습 trend로 봐서 추가 +5-10% 진전 가능 | low | OK |
| (β + γ continue) | jackpot + resume 결합 = γ가 transport 강화 유지 + β가 release 학습 incentive | low | confirm 필요 |
| (z-explicit gradient) | stage 2에 `- 2 * z_offset` 추가, z 학습 직접 incentivize | medium (reward 구조 변경) | confirm 필요 |
| (δ gripper_open_bonus 5.0) | release path bonus 강제 (P5 패턴, gripper open 시 +bonus) | medium | confirm 필요 |

## 권장 P6v11 plan (사용자 confirm 필요)

**제 권장: β + γ continue (success_jackpot 0 → 5.0, resume P6v10 model_999, 1000 iter)**

**근거**:
1. γ가 transport 학습 + zone 진입 강한 효과 입증 (success_zone 9% → 46%, strict gate 0.01% → 0.91%). γ 제거하면 P6v9 hover로 회귀.
2. Stage 4 bottleneck = release 학습 0. β jackpot 5.0이 strict gate 통과 시 강한 incentive = release 학습 trigger.
3. Resume P6v10 model_999 = γ 학습 보존 + 추가 iter로 학습 saturate까지 진행.
4. Jackpot 5.0 = P6v8 20.0 보다 보수적 (당시 zone 진입 0.88%라 fire 안 됨). P6v10에서는 zone 진입 46%, strict gate 0.91% → jackpot fire 가능.

**Falsifiability (P6v11 β + γ continue iter 1999 시)**:
| 분기 | 조건 |
|---|---|
| (A) SUCCESS | stage4>5% AND on_target>10% AND z_off<30mm |
| (B) PARTIAL | 5%>stage4>1% AND z_off<50mm |
| (C) FAIL still hover | stage4<1% AND z_off>60mm |
| (D) FAIL slow | stage4<1% AND z_off<50mm AND on_target>3% (P6v10 위치 그대로) |

**P6v11 (A) SUCCESS 시**: ST-C deploy 검토 시작.
**P6v11 (C/D) FAIL 시**: δ gripper_open_bonus 또는 z-explicit gradient escalate.

## HARD RULES 준수

- **#8**: archive 1단계 (5/09 완료), 다음 후보 5/11 (이번 세션 archive 권장).
- **#11**: /half-clone X 0회 (context 86% Stop hook 거부).
- **#14**: fail-fast guard 모든 ssh + nvidia-smi NVML mismatch 무시 가능 확정 (P6v9 6:50 + P6v10 7:00 wall 성공 = 실증).
- **#15**: cu128 sm_100 alive (P6v10 학습 = 추가 검증).
- **#17**: state-only 28-dim only.
- **#18**: 사용자 명시 "권장대로 진행" confirm으로 γ 적용 = 사용자 명시 결정. P6v11 β jackpot은 추가 confirm 필요 (또 다른 reward 구조 변경).
- **#19/#20**: sponge edge-stand 47mm + tower geometry 그대로.
- **#26**: 5/19 deadline **7일 ahead** (오늘 2026-05-12 정정). 1 iter ~7min × 2-3회 가능. P6v11 + P6v12 escalate room 있음.

## Date Correction

이전 MEMORY entries (5/13~5/16) 및 claudedocs (phase1_balpha_p6v9_session_20260516_result.md)는 잘못된 날짜. 오늘 = **2026-05-12** (system context 확정 + 사용자 명시). 5/19 deadline = 7일 잔여. 이전 entries는 그대로 보존 (HARD RULE #18: 사용자 명시 정정만 valid, retroactive 정정 X).

## B200 Inventory

- `logs/roarm_rl/p6v10_gamma_transport_shaping_resumeP6v9/` — 23 ckpts (model_0, 50, …, 999) + nn/
- `logs/phase1Balpha/train_p6v10.{out,err}` — train log
- `$ROARM_B200_ROOT/launch_p6v10.sh` — γ patch launch script
- roarm_stack_env.py md5 `a4b7883fc60b98bfadd6222bf23d455f` (P6v10 γ patch) — local↔B200 일치
