# Phase 1.B-α P6v6 결과 polling — Stage-3 TRAP 신규 분기 (F) 확정 (5/13 session)

## TL;DR

- 🔴 **결과 FAIL**: stage4_success_frac iter 0=**0.0422** → iter 999=**0.0148** (65% **↓**). PPO 1000 iter 학습이 stage 4 성공률을 **떨어뜨림**.
- 🔴 **신규 분기 (F) STAGE-3 TRAP**: 정책이 stage 3 (near + grasped) hover 학습 → stage 4 (success) 진입 incentive 거의 없음. 사용자 명시 분기 A/B/C/D/E 어디에도 정확히 매핑 안 됨 (E와 부분 매치).
- 🔍 **Reward 산수 진단**: stage 3 누적 ≈ 4.94/step × 400 step = **1976** 보장 / stage 4 누적 ≈ 0.12/step × 400 = **48**. 정책이 stage 3 hover 우세 → ManiSkill convention (max_ep=50) 미스매치.
- ⚠️ **잠재 추가 bug**: `ungrasp_signal = (q-low)/(high-low)` 정의가 RoArm gripper 컨벤션과 반대일 가능성 — iter 999 ungrasp_signal=0.94 (max) + gripper_open_rate=0.027 + grasped=0.937 동시 → release 인센티브 sign이 거꾸로일 가능성.
- ✅ B200 학습 7:00 wall @ 254K steps/s, 22 ckpts, 사용자 분기 결정 대기.

## B200 학습 종료 (PID 2054743)

| 항목 | 값 |
|---|---:|
| Wall time | **7:00** |
| Steps/s | 254K |
| Total timesteps | 98.3M |
| Iters | 1000 |
| Envs | 4096 |
| Ckpts | 22 (model_0 ~ model_999, 50 iter 간격) |
| Final std | 1.09 (1.30 → 1.09 천천히 collapse) |

## Iter trajectory (결정적 evidence)

| iter | action_std | reward | tgt_dist (mm) | height (mm) | grasped | open | grounded | stable | near | stage1 | stage2 | stage3 | **stage4** | ungrasp | static |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.30 | 20 | 182 | 55 | **0.048** | **0.542** | **0.135** | **0.616** | **0.056** | 0.900 | 0.043 | 0.020 | **0.0372** | 0.239 | 0.616 |
| 50 | 1.29 | 1690 | 133 | 122 | 0.928 | 0.032 | 0.003 | 0.120 | 0.099 | 0.071 | 0.818 | 0.097 | **0.0135** | 0.933 | 0.143 |
| 100 | 1.28 | 2079 | 127 | 115 | 0.928 | 0.032 | 0.003 | 0.125 | 0.531 | 0.071 | 0.391 | 0.521 | **0.0172** | 0.934 | 0.148 |
| 200 | 1.25 | 2291 | 123 | 109 | 0.927 | 0.031 | 0.003 | 0.135 | 0.678 | 0.071 | 0.247 | 0.666 | **0.0156** | 0.934 | 0.156 |
| 500 | 1.17 | 2283 | 122 | 104 | 0.933 | 0.028 | 0.002 | 0.197 | 0.705 | 0.066 | 0.227 | 0.697 | **0.0103** | 0.939 | 0.194 |
| 800 | 1.13 | 2299 | 111 | 94 | 0.938 | 0.029 | 0.003 | 0.233 | 0.758 | 0.061 | 0.180 | **0.7475** | **0.0119** | 0.941 | 0.222 |
| 999 | 1.09 | 2360 | **111** | **98** | **0.937** | **0.027** | **0.003** | 0.253 | **0.762** | 0.063 | 0.175 | **0.7482** | **0.0148** | **0.941** | 0.237 |

**핵심 관찰**:
1. **iter 0**: P6v5 정책 + sigma 1.30 noise → grasped 4.8% (학습 X) + gripper_open 54% + sponge_grounded 13.5% + sponge_stable 62% → 우연히 일부 env에서 (open + near + stable) 충족 → **latched stage4 4.22%**.
2. **iter 50 (1 step만에 대전환)**: 정책 즉시 grasp+hover 학습 → grasped 4.8→92.8% + open 54→3.2% + grounded 13.5→0.3% + stage4 4.22→1.35% (65% **↓** 즉시 발생).
3. **iter 100~999**: near_target rate 10→76% monotonic ↑ (target 도달은 학습) BUT stage4 정체 1.0~1.7%.
4. **stage3_neartgt** 10→**74.8%** dominant — 정책이 stage 3 hover에 머묾.
5. **gripper_open_rate** 0.027 + **ungrasp_signal_mean** 0.94 동시 (정의 충돌 의심).

## 분기 매핑 (사용자 명시 vs 실제)

| 분기 | 조건 | 실제 결과 | 매핑 |
|---|---|---|---|
| **(A) FULL SUCCESS** | stage4>0.3 AND open>0.15 | stage4=0.015, open=0.027 | **FAIL** |
| **(B) PARTIAL** | stage4 0.1-0.3 AND open>0.10 | 둘 다 미달 | **FAIL** |
| **(C) iter 0 plateau** | stage4 ≈ 0.04 변화 없음 | iter 0=0.042 → iter 999=0.015 (**↓**) | **FAIL** (오히려 ↓) |
| **(D) Stage 3 stuck** | stage2 그대로 + stage3<0.05 | stage2 0.175 (변화), stage3=0.748 (반대) | **FAIL** |
| **(E) FAIL 종합** | stage4<0.05 AND 모든 key 변화 없음 | stage4<0.05 ✓ but 큰 변화 多 | **부분 매치** |
| **🔴 (F) NEW: STAGE-3 TRAP** | stage3 dominant (>0.5) AND stage4 정체 + ↓ | stage3=0.748 ✓ + stage4 4.22%→1.48% ✓ | **CONFIRMED** |

## Root Cause 분석 — Reward 산수

ManiSkill StackCube convention (max_ep=50):
- stage 3 누적 최대: 7/step × 50 = **350**
- stage 4 누적 최대: 8/step × 50 = **400**
- → stage 4가 우세, success transition incentive 명확

**우리 P6v6 (max_ep=400)**:
- stage 3 누적 (iter 999): 6.59/step × 75% fire = **4.94/step avg** × 400 = **1976**
- stage 4 누적 (iter 999): 8.0/step × 1.5% fire = **0.12/step avg** × 400 = **48**
- → stage 3 hover가 stage 4 transition보다 **41× 우세**
- PPO 입장: stage 4 진입 시도 vs stage 3 유지 = expected gain +0.12 vs guaranteed +4.94 → **stage 4 회피가 합리적**

**결정적 misalignment**: ManiSkill의 reward 구조는 max_ep=50 horizon에 tuned. 우리 max_ep=400 long-horizon에서 동일 reward 적용 → stage 3 hover가 globally optimal.

## ⚠️ 잠재 추가 bug — gripper convention sign

| 변수 | 정의 (코드) | iter 999 값 | 의미 |
|---|---|---:|---|
| `ungrasp_signal` | `(q-low)/(high-low)` | **0.941** | q가 high 쪽 saturate |
| `gripper_open` (binary) | `q < 0.4 rad` | **0.027** | 3%만 q<0.4 |
| `grasped` | physics attach state | **0.937** | 94% 그리퍼 잡음 |

**모순 진단**: grasped=0.94 (잡고 있음, 즉 closed) + q high 쪽 saturate + binary "open"=3% (즉 97%는 q>0.4=closed)

이 조합은 일관성 있음 (모두 closed) BUT `ungrasp_signal=0.94`는 **"release 90%"** 의미인데 실제로는 **closed**. 즉 ungrasp_signal definition이 의도와 반대.

**가능한 sim joint convention**:
- RoArm URDF에서 joint 5 range가 close=high-q, open=low-q (HARD RULE #19/#20 사용자 컨벤션 cmd 5°=closed, +60°=open와 비교 검증 필요)
- 코드는 `(q-low)/(high-low)` = high-q일수록 ungrasp_signal high → **closed일 때 ungrasp_signal high = release 인센티브 정의가 거꾸로**

**확정 필요**: roarm_stack_env.py의 gripper joint URDF/USD definition + low/high 변수 출처 검증. 만약 거꾸로면 `ungrasp_signal = 1 - (q-low)/(high-low)` 또는 `(high-q)/(high-low)` 수정.

## 5 fix 후보 (사용자 confirm 필요)

| 옵션 | 변경 | 효과 | 위험 |
|---|---|---|---|
| **(α) episode length 400→50** | `max_ep=400 → 50` | ManiSkill convention matching, stage 3 hover incentive ↓ | Reach 단계 시간 부족 가능 → reach 자체 실패 |
| **(β) stage 4 reward 8→50** | success 시 jackpot | stage 4 진입 인센티브 11×↑ | reward farming, value func 불안 |
| **(γ) stage 3 reward 6→4** | gap (stage4-stage3) 1→4 ↑ | transition gradient sharp | stage 3 incentive 약 → 정책 stage 2로 retreat |
| **(δ) stage 4 latch step-wise +50/step** | success 후 누적 보너스 | place+maintain 학습 | 한 ep 1 success로 episode 종료 위주 학습 가능 |
| **(ε) gripper convention bug fix** | ungrasp_signal sign 확인 + 필요 시 invert | release 인센티브 정확히 작동 | bug 없으면 영향 X (검증 작업) |

**권장 우선순위**:
1. **ε 먼저** (코드 검증 30분, bug면 무료 fix)
2. **α + ε 결합** (ManiSkill matching) — horizon mismatch 핵심
3. ε 작동 시 β/γ 보류, ε 무효 시 β (jackpot) 또는 α (horizon)

**Falsifiability** (다음 학습 success criteria iter 999):
- `stage4_success_frac > 0.20` (현재 0.015 13× ↑)
- `place_success_rate > 0.20`
- `gripper_open_rate > 0.10` (현재 0.027 4× ↑)
- `near_target_rate ≥ 0.50` (현재 0.762 유지)

## HARD RULES 준수

- #8 archive: 5/08 late → `MEMORY_archive_20260513.md` 완료. **⚠️ 6 full bodies 잔존 (limit 5 위반 1개) — 다음 세션 추가 archive (5/08 morning P4-P5-P6 후보)**.
- #11 /half-clone X 0회 (Stop hook ~86% 거부 예상, claudedocs + MEMORY로 처리).
- #14 fail-fast guard: 모든 ssh `set -e + source env.sh + [[ -z $ROARM_B200_ROOT ]] && exit 1`. ssh non-login 셸 `$ROARM_B200_ROOT` 미정의 trap 1회 발견 후 즉시 가드 추가.
- #15 cu128 sm_100 alive (학습 종료 확인 = 추가 검증).
- #17 state-only 28-dim only (visual RL 시도 X).
- #18 사용자 명시 4 결정 보존 (target / gravity / 28-dim / P4-P5-P6 phase 구조 — P6 안의 reward 디자인은 implementation detail).
- #19/#20 sponge edge-stand 47mm + tower geometry 그대로.
- #26 5/19 deadline **6일 ahead** (B200 RL training PoC 진행 중).

## B200 inventory

```
$ROARM_B200_ROOT/logs/roarm_rl/p6v6_maniskill_replace_tower_resumeP6v5/
  ├── 22 ckpts (model_0 ~ model_999, 50 iter step) — model_999.pt next resume 후보
  ├── git/  (학습 시 코드 snapshot)
  └── events.out.tfevents.* (TensorBoard log)
$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v6.{out,err}
$ROARM_B200_ROOT/launch_p6v6.sh
$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py  md5=9b0bccb5...
```

## 다음 세션 즉시 명령

```bash
# 옵션 ε (gripper convention 검증) 먼저
grep -n "ungrasp_signal\|gripper_q\|gripper_open\|joint.*5\|low\|high" roarm_rl/roarm_stack_env.py | head -30

# RoArm URDF/USD joint 5 range 확인 (사용자 컨벤션 vs 코드 매핑)
# CLAUDE.md HARD RULE #19/#20: gripper cmd 5°=closed, +60°=open

# B200 ckpt 직접 inspection (gripper actor bias 확인)
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh;
  python -c \"
import torch
ckpt = torch.load('"'"'$ROARM_B200_ROOT/logs/roarm_rl/p6v6_maniskill_replace_tower_resumeP6v5/model_999.pt'"'"', map_location='"'"'cpu'"'"')
sd = ckpt['"'"'model_state_dict'"'"']
for k in sd:
    if '"'"'actor'"'"' in k.lower() and ('"'"'.bias'"'"' in k or '"'"'.weight'"'"' in k):
        v = sd[k]
        if v.numel() <= 10:
            print(k, v.tolist())
\""'

# 사용자 결정 받고 (α/β/γ/δ/ε 선택) 패치 → resume model_999 + 1000 iter
```
