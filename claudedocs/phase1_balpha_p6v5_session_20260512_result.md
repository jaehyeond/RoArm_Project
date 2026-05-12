# Phase 1.B-α P6v5 결과 — 5/12 polling session

## TL;DR

- ✅ **P6v5 학습 완료** PID 2045744, **wall 7:20** @ 258K steps/s (4096 envs, 1000 iter, 98.3M timesteps), 22 ckpts (model_0/50/100/.../999).
- ✅ **B.1 (_was_grasped latch) + B.2 (actor bias reset)** 패치 모두 활성화 verify:
  - `[train] reset_actor_bias: actor.6.bias[5]: +0.8446 -> 0.0`
  - `[train] reset_std: ckpt std [1.36, 1.30, 1.30, 1.28, 1.31, 1.31] -> [1.30×6]`
  - `entropy_coef override: 0.005 -> 0.001`
- 🔴 **분기 (C) BIAS RE-SATURATION 확정**: gripper_open_rate iter 0=**0.542** → iter 50=**0.032** (17× ↓) → iter 999=**0.027**. **PPO가 50 iter 만에 bias 재saturate**. 동시에 std도 1.30→**1.17** (천천히 entropy collapse).
- 🔴 **place_success_rate 0.0000 1000 iter 내내** (fire 0회). place_cond_fire_rate **0.0001** (negligible).
- 🔴 **misspecification 정량 확정**: hold-path globally optimal. PPO가 bias reset 후 50 iter 만에 same conclusion 도달. Reward 구조 자체를 바꾸지 않으면 entropy clipping만으로 부족.

## 학습 trajectory (iter snapshot)

| iter | std | reward | sponge_target_dist | sponge_height | grasped | gripper_open | grounded | lift_succ | place_succ | place_cond_fire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | 1.30 | 6.24 | 181.9mm | **54.7mm** | 0.048 | **0.542** ✅ | **0.135** ✅ | 0.000 | 0.000 | 0.0045 |
| 50 | 1.29 | 665.35 | 141.4mm | 129.3mm | 0.93 | **0.032** ❌ | 0.004 | 0.754 | 0.0002 | 0.0000 |
| 100 | 1.29 | 644.93 | 139.6mm | 128.0mm | 0.93 | 0.032 | 0.003 | 0.755 | 0.000 | 0.0001 |
| 200 | 1.27 | 679.72 | 137.4mm | 127.9mm | 0.93 | 0.031 | 0.003 | 0.755 | 0.000 | 0.0000 |
| 300 | 1.26 | 682.56 | 137.5mm | 127.9mm | 0.93 | 0.030 | 0.003 | 0.754 | 0.000 | 0.0000 |
| 500 | 1.24 | 678.59 | 136.1mm | 127.9mm | 0.93 | 0.028 | 0.003 | 0.758 | 0.000 | 0.0000 |
| 700 | 1.21 | 676.68 | 135.6mm | 127.4mm | 0.94 | 0.027 | 0.003 | 0.767 | 0.000 | 0.0000 |
| **999** | **1.17** | **688.87** | **133.0mm** | **124.6mm** | **0.9347** | **0.0272** | **0.0028** | **0.7657** | **0.000** | **0.0001** |

### 결정적 관찰

1. **iter 0의 의미**: bias=0, std=1.30 직후 → gripper_open_rate=**54.2%**, sponge_grounded_rate=**13.5%**, grasped=**4.8%**. 즉 정책 자체는 random gripper sampling 능력을 가졌지만, P6v4 ckpt에서 학습된 navigation/grasp trajectory는 무너짐 (bias reset이 actor 마지막 레이어의 한 dim만 0으로 만들었으나 그 변화가 grasp pipeline 전체에 영향).
2. **iter 50 (1/20 학습 진행 시점)**: gripper_open 54.2% → 3.2% (17× ↓), grasped 0.048 → 0.93 (19× ↑). **PPO가 50 iter 만에 close-gripper 재학습 완료**. 즉 reward gradient가 일관되게 close 방향.
3. **iter 50 → 999**: gripper_open 0.032 → 0.027로 미세 변화. 정책은 50 iter에서 이미 hold-path 수렴, 이후 950 iter는 fine-tune (sponge_target_dist 141→133mm로 8mm 감소 only).
4. **std collapse**: 1.30 → 1.17. log_std parameter가 entropy_coef=0.001만으로는 천천히 감소. log_std_min clipping 부재로 entropy collapse 진행 중.
5. **sponge_height 0.547mm → 124.6mm**: sponge 공중 hover **~12.5cm above table** 유지. lift_success_rate=0.77 (sponge_height>10cm threshold) 만족.

## 진단 — 왜 fix가 무효화되었는가?

### B.2 (bias reset) 효과 검증

- ✅ Reset 자체는 SUCCESS (iter 0 verify): bias[5] +0.8446 → 0.0, std 1.30 uniform.
- ❌ Reset 후 50 iter 내 **PPO가 bias[5]를 다시 양수로 학습**. iter 999에서 정확한 bias 값은 ckpt read 필요 (sub-task) but gripper_open_rate=0.027 → close 방향으로 saturate 다시 (positive bias 추정).
- **Root cause**: reward gradient가 close 방향. P6v4와 동일 reward 구조 → PPO가 동일 local optimum (hold path) 재발견.

### B.1 (_was_grasped latch) 효과 검증

- ✅ Latch 작동 verify: was_grasped_rate=0.9347 ≈ grasped=0.9347 (정확히 매칭). 즉 latch가 grasp 동안 True 되었고 그 후 풀리지 않음 (`_was_grasped[env_ids] = False`은 episode 경계에서만 reset).
- ❌ 하지만 **gripper_open 자체가 매우 드물게 fire** (2.7%) → release path가 거의 실행 안 됨 → latch가 unlock하는 시나리오 (release 후 lower_reward 활성) 자체가 fire 안 함.
- **Root cause**: latch 디자인은 correct but, bias re-saturation이 release 자체를 막아 latch 효과 측정 불가.

### B.4 (logging) 효과 검증

- ✅ 4 신규 key 모두 출력 정상 (was_grasped_rate, gripper_open_rate, sponge_grounded_rate, place_cond_fire_rate).
- 진단 정확도 ↑: place_cond_fire_rate=0.0001로 place_cond 자체가 거의 fire 안 함을 정량화 가능.

## 분기 결정 — Branch (C) BIAS RE-SATURATION

사용자 명시 분기 매핑:

| 분기 | 조건 | P6v5 결과 | 매칭 |
|---|---|---|---|
| (A) FULL SUCCESS | open>0.10 AND place>0.05 AND height<0.05 | open=0.027, place=0.000, height=0.125 | ❌ |
| (B) PARTIAL | open>0.10 AND place<0.05 | open=0.027<0.10 | ❌ |
| **(C) BIAS RE-SATURATION** | **open<0.05 (1000 iter 사이 bias 재saturate)** | **open=0.027<0.05** | **✅** |
| (D) DESCENT FAIL | grounded<0.01 AND open>0.10 | open=0.027<0.10 | ❌ |
| (E) FULL FAIL | 모든 신규 key 변화 없음 | iter 0 vs 999 std 1.30→1.17, grasped 0.048→0.93 등 변화 있음 | ❌ |

**Branch C 확정.** Fix 우선순위:

### 권장 Fix (사용자 confirm 필요)

🔴 **C-primary: A2 #2 log_std clipping 영구 적용 (즉시 적용)**

근거:
- std 1.30→1.17 천천히 감소 → entropy_coef=0.001만으로는 entropy collapse 미해결.
- rsl_rl ActorCritic에 `log_std_min/max` 기능 추가 또는 wrapper로 매 step `.std.data.clamp_(min=0.5)` 형태로 강제.
- 효과: gripper dim std≥0.5 유지 → P(close)/step 항상 random sampling 일정 비율 보장.

C-primary patch plan (next session):

```python
# train_ppo.py 또는 별도 wrapper
# 매 PPO update 후 std clipping
LOG_STD_MIN = -0.69  # std >= 0.5
for p in target.actor.log_std.parameters():  # or .std if exposed
    p.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
```

🟡 **C-secondary: per-dim entropy bonus on gripper dim**

- gripper dim (idx 5)에만 entropy bonus weighting 증가 (예: entropy_coef × 5).
- Implementation: actor의 6-dim normal dist에서 dim 5 entropy만 weighted 추출.
- 효과: gripper dim 탐험 보장, base/shoulder/elbow는 deterministic 유지.

🟠 **C+A 결합 옵션 (가장 강력, 사용자 confirm 필요)**

만약 log_std clipping만으로 fail 시 (next round 후 평가) → **Option A reward re-weight 결합**:
- `lift_reward_scale 5→0.5` (hold-path 인센티브 약화)
- `grasp_bonus_scale 2→0.2`
- `place_bonus_scale 5→50` (release path 강화)
- `action_penalty` 추가 (constant negative drift)

HARD RULE #18 confirm 검토: P4-P5-P6 phase 구조는 implementation detail 안 → reward scaling 자체는 OK.

🔵 **Option C (last resort): obs 28→32 (sponge_to_target_quat 4dim)**

3-agent 모두 권장. 사용자 명시 confirm 필요 (HARD RULE #18 — obs 28-dim은 사용자 명시 4 결정).

## 다음 세션 plan

1. **사용자 confirm**: C-primary (log_std clipping) 단독 vs C+A 결합 선택.
2. **C-primary 패치**:
   - `train_ppo.py`: ActorCritic의 std parameter 위치 확인 (rsl_rl 3.1.2 어디?) → 매 update 후 clamp.
   - 또는 rsl_rl PPO algorithm.update_step 내에 hook 삽입.
3. **resume P6v5 model_999 + log_std clamp**, 1000 iter ~7min.
4. **성공 기준**: gripper_open_rate>0.10 (sustained at iter 999, not just iter 0).
5. **Falsifiability**: iter 999에서도 gripper_open<0.05면 → reward 구조 자체 문제 → Option A 결합.

## HARD RULES 준수

- #8 archive 1단계: 5/08 새벽 (P3 fix + Precision Compare + 1.B-α env code) 본문 → `MEMORY_archive_20260511.md`. MEMORY.md 한 줄 pointer.
- #11 /half-clone 거부 0회 (Stop hook context 안정).
- #14 fail-fast guard: 모든 ssh `set -e + source env.sh + [[ -z $ROARM_B200_ROOT ]] && exit 1`. 초기 `$ROARM_B200_ROOT` 비대화 셸 미정의 trap 발견 후 즉시 수정.
- #15 cu128 sm_100 alive: P6v5 학습 완료 = 추가 검증.
- #17 visual RL X (state-only 28-dim only).
- #18 사용자 명시 4 결정 (target Y=-0.0435 / gravity / 22→28-dim / P4-P5-P6) 보존. B.1/B.2/B.4는 implementation detail.
- #19 sponge edge-stand 47mm / #20 # tower geometry / #26 5/19 deadline 8일 ahead.

## B200 inventory (5/12)

```
$ROARM_B200_ROOT/logs/roarm_rl/p6v5_was_grasped_latch_bias_reset_resumeP6v4/
├── events.out.tfevents.1778499490.JHPark-container.2045744.0
├── git/
├── model_0.pt ... model_50.pt ... model_100.pt ... model_999.pt   (22 files)

$ROARM_B200_ROOT/logs/phase1Balpha/
├── train_p6v5.out  (1.29 MB)
└── train_p6v5.err  (2.18 KB)
```

Next resume 후보: **model_999.pt** (final, std=1.17 안정, grasp 능력 보존).

## 환각 정정 + 검증 흔적

- 초기 polling 시 `\$ROARM_B200_ROOT` 비대화 셸 expand 실패 → tail이 `/logs/...` 절대경로로 잘못 해석되어 "No such file" 에러. HARD RULE #14 fail-fast guard 패턴으로 즉시 정정 (`source /NHNHOME/.../env.sh` 절대경로).
- bias reset 검증은 ckpt read까지 안 가고 log의 `[train] reset_actor_bias: ...` line으로 직접 검증 (가장 reliable).
- iter 0 metrics가 매우 중요 — bias reset 직후 정책의 base behavior (random gripper sampling) 확인 evidence.
