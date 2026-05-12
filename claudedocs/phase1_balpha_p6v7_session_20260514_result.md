# Phase 1.B-α P6v7 결과 polling — ε ungrasp_signal sign fix 단독은 부족, 분기 (C) FAIL 확정 (5/14 session)

## TL;DR

- ✅ **ε fix 코드 레벨 정확 작동 verified**: ungrasp_signal_mean **0.9409 → 0.0616** (P6v6 → P6v7), 정확한 sign flip (1-0.94=0.06 within rounding). `(high-q)/(high-low)` 적용 정상.
- 🔴 **그러나 macro-result 거의 변화 없음**: stage4_success_frac **0.0148 → 0.0153** (+0.05%p only), gripper_open_rate **0.0272 → 0.0272** (정확히 동일).
- 🔴 **분기 (C) FAIL 확정**: 사용자 명시 ablation criteria — A=fail (stage4 < 5%), B=fail (open < 5%), C=match (stage4 ≤ 2% AND P6v6 차이 미미), D=부분 (의외 회귀 아님).
- 🔍 **Root cause 재진단**: ε fix는 stage 3 안 micro-incentive만 정정 (stage 3 reward 6.59→6.16/step, -7%). Release path 자체의 macro-cliff (grasped=False가 stage 2/3 entirety OFF → -2.7/step net loss)는 변함 없음 → PPO release 회피.
- ✅ **권장 다음 step**: α (episode 400→200) + Fix A (is_success_zone=50mm 분리, stage 3 → stage 4 진입 시 jackpot reward) **결합** P6v8. 사용자 confirm 대기.
- ✅ B200 학습 6:55 wall @ 240-244K steps/s, 22 ckpts.

## B200 학습 종료 (PID 2144634)

| 항목 | 값 |
|---|---:|
| Wall time | **6:55** |
| Steps/s | 240-244K |
| Total timesteps | 98.3M |
| Iters | 1000 |
| Envs | 4096 |
| Ckpts | 22 (model_0~999, 50 iter step) |
| experiment_name | `p6v7_ungrasp_sign_fix_resumeP6v6` |
| Resume from | P6v6 model_999 (no reset_std, no reset_actor_bias_idx) |
| Final std | **0.96** (1.09 → 0.96, P6v6 1.30→1.09 대비 더 빠른 entropy collapse) |

## P6v6 vs P6v7 iter 999 비교 (결정적 evidence)

| 지표 | P6v6 (iter 999) | P6v7 (iter 999) | Δ | 해석 |
|---|---:|---:|---:|---|
| Mean reward | 2360.16 | **2221.06** | -139 (-6%) | ungrasp signal 0.94→0.06 작아진 만큼 (-0.5×0.88×400×0.76 ≈ -134) ✅ 산수 일치 |
| action_std | 1.09 | **0.96** | -0.13 | entropy collapse 가속 (정책 더 결정적) |
| tcp_sponge_dist (m) | 0.0131 | 0.0135 | +0.4mm | 동일 (excellent reach) |
| sponge_target_dist (m) | 0.1107 | 0.1103 | -0.4mm | 동일 (110mm hover) |
| sponge_height (m) | 0.0978 | 0.0916 | -6.2mm | 살짝 ↓ but 여전히 ~9cm hover |
| grasped_frac | 0.9365 | 0.9333 | -0.3%p | 동일 (94% grasping) |
| was_grasped_rate | 0.9365 | 0.9336 | -0.3%p | latch 정상 작동 |
| **gripper_open_rate** | **0.0272** | **0.0272** | **0.0%p** | **🔴 정확히 동일! 0% 변화** |
| sponge_grounded_rate | 0.0032 | 0.0038 | +0.06%p | noise 수준 |
| sponge_stable_rate | 0.2365 | 0.2614 | +2.5%p | 약간 ↑ |
| near_target_rate | 0.7620 | 0.7771 | +1.5%p | 동일 (78% near) |
| stage1_reach_frac | 0.0625 | 0.0654 | 동일 | 6% reach 단계 |
| stage2_grasp_frac | 0.1745 | 0.1559 | -1.9%p | 동일 |
| **stage3_neartgt_frac** | **0.7482** | **0.7634** | **+1.5%p** | **🔴 dominant 그대로 (76%)** |
| **stage4_success_frac** | **0.0148** | **0.0153** | **+0.05%p** | **🔴 변화 미미** |
| place_success_rate | 0.0148 | 0.0153 | +0.05%p | stage 4와 동일 |
| **ungrasp_signal_mean** | **0.9409** | **0.0616** | **-0.879** | **✅ sign flip 정확! 1-0.94=0.06** |
| static_signal_mean | 0.2365 | 0.2471 | +1.1%p | noise |

## ε fix verification — 코드 레벨 정확 작동

- `roarm_rl/roarm_stack_env.py:519-528` patch: `ungrasp_signal = (high - q) / (high - low)` (sign inverted from `(q-low)/(high-low)`).
- iter 999 ungrasp_signal_mean **0.0616 ≈ 1 - 0.9409 = 0.0591** (rounding 차이 0.003) → **flip 수학적으로 정확**.
- gripper joint convention: q LOW=OPEN, q HIGH=CLOSED (HARD RULE #19/#20 + 5/13 evening E1/E2/E3 cross-check).
- ✅ ε fix는 의도대로 적용됨.

## Root cause 재진단 — ε fix 단독이 부족한 이유

**가설 (5/13 evening)**: ungrasp_signal sign이 정확하면 stage 3 안에서 release 인센티브가 정확히 작동 → release rate 상승 → stage 4 진입.

**실제 결과**: stage 3 reward만 micro 변화 (6.59→6.16/step), gripper_open_rate 정확히 0% 변화. release 시도 fire 안 함.

**왜 fail?**

| State | grasped | reward 구성 | reward (P6v7 iter 999) |
|---|---:|---|---:|
| HOLD (잡고 hover near) | True | stage 3: `6 + 0.5×ungrasp(0.06) + 0.5×static(0.25)` | **+6.16/step** |
| RELEASE (놓는 시도) | False (1 step 후) | stage 1 only: `2×(1-tanh(5×d_tcp_sponge)) ≈ 2×(1-tanh(0.07)) ≈ 1.86` | **+1.86/step** |
| Net loss for release | | | **-4.30/step** |

**즉**: PPO 입장 release 시도 = -4.30/step 보상 절벽 → ε fix가 stage 3 안 incentive를 0.5 → 0.03으로 낮춰도 cliff(-4.30)에 비해 무시 가능. **release path 자체의 macro-cliff (grasped=False가 stage 2/3 entirety OFF)** 가 핵심 issue.

**P6v6 Stage-3 trap (5/13 entry 분기 F) 진단이 P6v7에도 그대로 적용**: hold-path globally optimal, ε fix 단독 무효.

## 분기 매핑 (사용자 명시 ε ablation criteria)

| 분기 | 조건 | 실제 P6v7 | 매핑 |
|---|---|---|---|
| **(A) SUCCESS** | stage4 > 5% AND open > 10% | stage4=1.53%, open=2.72% | **FAIL** (둘 다 미달) |
| **(B) PARTIAL** | stage4 2-5% AND open 5-10% | stage4 < 2% AND open < 5% | **FAIL** |
| **🔴 (C) FAIL** | stage4 ≤ 2% 또는 P6v6과 차이 미미 | stage4=1.53% ≤ 2% AND Δ=+0.05%p | **MATCH** |
| **(D) 의외 회귀** | stage4 < 1% 또는 gripper_open=0 | stage4=1.53% > 1% AND open=2.72% > 0 | FAIL (회귀 아님) |

→ **분기 (C) FAIL 확정**. 사용자 명시 권장: **α=200 + Fix A (success_zone=50mm 분리) 결합 P6v8**.

## 다음 fix 후보 (분기 (C) 매핑 + 5/13 evening Track C SOTA 종합)

| 옵션 | 변경 | 효과 | 위험 | 권장도 |
|---|---|---|---|---|
| **(α) episode length 400→200** | `--episode_length 200` | ManiSkill 50 + DrS "horizon = 3-5× minimum completion" 절충 (우리 task min ~150 → 권장 450-750 BUT stage 3 hover bias 완화 위해 200) — stage 3 누적 1976 → 988 (50% ↓) | reach 단계 시간 부족 가능 | ⭐⭐ |
| **(Fix A) success_zone=50mm 분리** | `is_near_target=100mm` (stage 3) vs `is_success_zone=50mm` (stage 4) — stage 3 → stage 4 진입 시 별도 jackpot (예: +20 fire) | release path가 hold path 압도 (cliff -4.30 > jackpot +20 ≫ -4.30) | reward farming 위험 (jackpot 너무 크면 fire 반복) | ⭐⭐⭐ |
| **🟢 (α + Fix A 결합)** | 위 둘 동시 | stage 3 누적 50% ↓ + stage 4 진입 jackpot — release path 압도 + horizon scaling | tuning 변수 2개 | **⭐⭐⭐⭐ 최우선** |
| (β) jackpot 단독 | stage 4 reward 8 → 50 | release 인센티브 6.25× | stage 3 누적 1976 여전 dominant → 무효 가능 | ⭐ |
| (γ) actuator/init mismatch fix | stiffness/damping/effort/init q 정합 (5/13 evening 발견) | sim2real transfer 정확도 ↑ but RL 학습 가능성에 직접 영향 X | scope creep | 5/19 후 |

**권장 우선순위 (사용자 confirm 필요)**:
1. **(α + Fix A 결합)** — P6v8 launch (resume P6v7 model_999, 1000 iter, ETA ~7min)
2. (Fix A 단독, jackpot=20) — α 위험 회피용 fallback
3. (β 단독) — α 거부 시 alternative

**Falsifiability** (P6v8 success criteria iter 999):
- `stage4_success_frac > 0.05` (현재 0.015 3× ↑ minimum)
- `gripper_open_rate > 0.05` (현재 0.027 1.8× ↑)
- `stage3_neartgt_frac < 0.50` (현재 0.76 dominant 완화)
- `place_success_rate > 0.05`

## 부수 발견 — Track B + Track C (5/13 evening 적용)

5/13 evening session의 Track B (IK 미사용 확인) + Track C (SOTA research) 결과는 본 polling에서 추가 검증 안 함 (별도 issue):

- **Track B**: action 6-dim = 6 joint q_target **delta** (incremental, action_scale=0.1). PPO joint space 직접 학습 (no EE pose, no IK). ✅ 본 학습에 영향 없음.
- **Track C — Actuator/Init mismatch (별도 fix 후보 γ)**: `isaac_roarm_m3/.../roarm_m3.py` reference vs 우리 `roarm_stack_env.py:152-168` — stiffness 80 vs 200/170/120/80/50/grip 60 (mass-scaled), damping 4 vs 80/65/45/30/20/grip 20 (**5-20× 부족**), effort 2.5 vs 1.9 (1.3× 강함), init q [0,0,90°,0,0,0] vs dataset mean (v6 deploy 정합) [0.047,0.704,0.228,1.095,-0.046,0.168]. **학습 가능성 OK but sim2real transfer 정확도 ↓**. 5/19 deadline 우선순위 ↓ (P6v8 결과 후 별도 fix).

## HARD RULES 준수

- **#8 archive**: 5/08 morning (Phase 1.B-α Stack 1→6, 5 BUGs, P4-P5-P6 진행) → `MEMORY_archive_20260514.md` 본문 그대로 이동 + MEMORY.md 한 줄 pointer (6→5 정리). ⚠️ 이전 5/13 evening 노트의 "6 full bodies" 카운트 vs 실제 grep 카운트 (8 full bodies) 불일치 — 5/14 prepend +1 archive -1 → 8 그대로 (limit 5 violation 3개 잔존). 다음 세션 추가 archive 필요 (5/08 evening, 5/09, 5/11 후보).
- **#11**: /half-clone X 0회.
- **#14**: 모든 ssh `set -e + source env.sh + [[ -z $ROARM_B200_ROOT ]] && exit 1 + [[ $(whoami) != sogang_jhki ]] && exit 1` 패턴 준수. Non-login 셸 `$ROARM_B200_ROOT` 미정의 trap 가드 적용.
- **#15**: cu128 sm_100 alive (학습 6:55 완료 = 추가 검증).
- **#17**: state-only 28-dim only (visual RL X).
- **#18**: 사용자 명시 4 결정 보존 (target / gravity / 28-dim / P4-P5-P6 phase 구조). P6 안의 reward 디자인 + episode length는 implementation detail.
- **#19/#20**: sponge edge-stand 47mm + tower geometry 그대로.
- **#26**: 5/19 deadline **5일 ahead** (B200 RL training PoC 진행).

## B200 inventory

```
$ROARM_B200_ROOT/logs/roarm_rl/p6v7_ungrasp_sign_fix_resumeP6v6/
  ├── 22 ckpts (model_0 ~ model_999, 50 iter step) — model_999.pt = next P6v8 resume 후보
  ├── git/  (학습 시 코드 snapshot)
  └── events.out.tfevents.1778568753.JHPark-container.2144634.0
$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v7.{out,err}
$ROARM_B200_ROOT/launch_p6v7.sh
$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py  md5=143bd74b... (5/13 evening 적용 ε fix 포함)
```

## 다음 세션 즉시 명령 (P6v8 launch 시)

```bash
# 1) α + Fix A 패치 적용 (사용자 confirm 후)
# roarm_rl/roarm_stack_env.py:
#   - is_success_zone = sponge_target_dist < 0.05  (stage 4 진입 zone, 50mm)
#   - is_near_target = sponge_target_dist < 0.10   (stage 3 진입 zone, 100mm 그대로)
#   - stage 4 reward에 jackpot +20 fire (success_zone 진입 시 1회만)
# roarm_rl/train_ppo.py:
#   - --episode_length 200 (default 400 → 200, ManiSkill 50 + DrS 절충)

# 2) md5 verify + sanity test PASS
md5sum roarm_rl/roarm_stack_env.py
ssh JHPark "md5sum $ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py"
python -m roarm_rl.test_sanity_stack  # 16env × 30step

# 3) B200 launch
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh;
  [[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1;
  nohup \$ROARM_B200_ROOT/launch_p6v8.sh > \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v8.out 2>\$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v8.err &
  sleep 2; ps -p \$! -o pid,etime,stat"'

# 4) 결과 polling (~10min 후)
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh;
  [[ -z \"\$ROARM_B200_ROOT\" ]] && exit 1;
  ps -p <P6v8_PID> -o pid,etime,stat;
  tail -100 \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v8.out;
  ls \$ROARM_B200_ROOT/logs/roarm_rl/p6v8_*/"'
```
