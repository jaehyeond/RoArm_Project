# Phase 1.B-α P6 v3 (A2 #1 chicken-and-egg fix) — 5/08 evening session

## TL;DR
- ✅ A2 #1 (CRITICAL) 패치 3개 적용: cfg `gripper_open_bonus_scale=2.0` + `_place_condition`에서 gripper_open 제거 후 `sponge_grounded`(z<table+30mm) 추가 + `_get_rewards` P6에 `gripper_open_when_near` separate bonus
- ✅ md5 일치 확인 (89fc3f92938d47613a0da614994a1777, local↔B200), sanity 16env×30step PASS
- 🚀 P6v3 학습 launch (B200 PID 1681274): resume P6v2 model_999 + reset_std 1.5 + entropy_coef 0.001, 1000 iter (~7min ETA)
- ❌ **P6v3 plateau (iter 503 시점)**: place_success_rate=0.000 그대로. action_std 1.48 안정 ✓, lift_success 0.18→0.32 ↑, **sponge_height=0.1006m** (sponge 공중 hover) → sponge_grounded NEVER fire. **chicken-and-egg #2 발견**: gripper_open 제거해도 sponge_grounded가 새 bottleneck.

## 패치 (`roarm_rl/roarm_stack_env.py`)
| # | 변경 | 위치 | 효과 |
|---|---|---|---|
| 1 | `gripper_open_bonus_scale: float = 2.0` 추가 | cfg L221-223 | separate gripper-open incentive |
| 2 | `_place_condition`: gripper_open 제거 → `sponge_near & sponge_stable & sponge_grounded` | env L530-548 | place_cond fire 가능 (gripper saturate close에 안 막힘) |
| 3 | `_get_rewards` P6: `(gripper_open & sponge_near).float() * gripper_open_bonus_scale` | env L398-414 | gripper open 학습 신호 보존 |

## chicken-and-egg root cause (5/08 late 진단)
P5v2 actor.6.bias[5]=+0.798 (close 방향) + std=1.5 + action_scale=0.1 + clamp(-1,1) → gripper_q saturate at max (close) → `gripper_open` (gripper_q < 0.4 rad) NEVER fires → place_cond NEVER fires (P6v2 1000 iter 내내 fire 0회).

해결: place_cond의 AND-condition에서 gripper_open 분리 → sponge_grounded로 대체 (sponge가 table 근처에 있고 stable이면 ok). gripper open은 `gripper_open_when_near` 별도 bonus로 incentivize.

## 학습 설정 (P6v3)
```bash
exec python -u -m roarm_rl.train_ppo \
    --task stack \
    --num_envs 4096 \
    --max_iterations 1000 \
    --reward_phase 6 \
    --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v2_thresh100mm_resetstd1p5_entropy0p001_resumeP5v2/model_999.pt \
    --reset_std 1.5 \
    --entropy_coef 0.001 \
    --experiment_name p6v3_no_gripper_in_placecond_grounded_resumeP6v2
```

- log: `$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v3.{out,err}`
- ckpts: `$ROARM_B200_ROOT/logs/roarm_rl/p6v3_no_gripper_in_placecond_grounded_resumeP6v2/`
- ETA: ~7 min @ 244K steps/s

## P6v2 vs P6v3 비교 (목표)
| 메트릭 | P6 v2 (이전) | P6 v3 (target) |
|---|---|---|
| action_std | 1.50→1.46 (안정) | 안정 유지 |
| sponge_target_dist | 103mm | <60mm 목표 |
| grasped_frac | 0.94 | 유지 (~0.9) |
| lift_success_rate | 0.18 | 유지~상승 |
| **place_success_rate** | **0.000** | **>0** ← 핵심 |

## P6v3 plateau 진단 (chicken-and-egg #2, iter 503)

### 데이터 (iter 1000 final)
| 메트릭 | P6 v2 (이전 끝) | P6 v3 (iter 1000) | 변화 |
|---|---|---|---|
| action_std | 1.46 | 1.48 | 안정 ✓ |
| sponge_target_dist | 0.103m | 0.108m | 거의 동일 plateau |
| **sponge_height** | (미관측) | **0.104m** | sponge 공중 hover ~10cm |
| lift_success_rate | 0.18 | 0.34 | ↑ |
| **place_success_rate** | **0.000** | **0.000** | ❌ 1000 iter 내내 fire 0회 |
| Total timesteps | 24.6M | 98.3M | 1000 iter |

### 진단 (chicken-and-egg #2)
정책은 sponge를 target (x,y) 위에 ~10cm 높이로 들고 있음. `sponge_grounded` (z<TABLE_Z+0.030 = +0.018m) 조건은 sponge_z=+0.088m이라 NEVER fires.

### Root cause: 보상 균형 (reward arithmetic)
정책이 grasp→hold 자세를 유지하면 매 step:
- `grasp_bonus` = +2/step (grasped 동안)
- `lift_reward` = +5/step (sponge 들고 있는 동안)
- `nav_reward` = -d × 5 = -0.5/step (d=0.1m, 작음)
- 총합 ≈ +6.5/step

gripper open + 내려놓기 시:
- 위 3개 다 0 (`post_place_gate=~place_success_flag`이지만 grasp/lift는 grasped 끊기면 자동 0)
- `gripper_open_when_near` = +2/step
- `place_bonus_scale` = +5/step (place_cond fire하면, 즉 grounded되면)
- 총합 = +7/step **after** sponge가 grounded

**문제**: sponge가 grounded되기 위해서는 일단 release 해야 하고, release하는 동안 sponge가 떨어지는 ~10-20 step은 reward 손실. 그 다음 grounded되어 place_bonus 받기까지 credit assignment ~30 step gap. PPO가 이 path 학습 어려움.

## P6v4 fix plan (다음 세션 즉시 적용)

### 핵심 아이디어: Release path 보상 강화 + lift/grasp gate 변경

**Patch #1: `lower_when_near` reward 추가** (sponge가 target 근처에 있을 때 height 줄이기 reward)
```python
# In _get_rewards P6 (after gripper_open_when_near)
if self.cfg.reward_phase >= 6:
    # Lower sponge while near target (encourages descent before release)
    sponge_height = self._sponge_pos_w[:, 2] - TABLE_Z
    sponge_near = d_sponge_target < self.cfg.place_dist_thresh
    lower_reward = -sponge_height * self.cfg.lower_reward_scale  # negative when high
    rewards = rewards + lower_reward * sponge_near.float() * self._grasped.float()
```
- cfg: `lower_reward_scale: float = 5.0` (sponge_height 10cm × 5 = 0.5/step penalty)

**Patch #2: gripper_open_bonus_scale 2.0 → 10.0**
release path가 hold path보다 우월해지도록 강화.

**Patch #3: lift_reward, grasp_bonus를 sponge_near gate** (target 근처 도달하면 lift/grasp reward 끄기)
```python
# In _get_rewards P5 area: gate by ~sponge_near so they fade out near target
near_gate = ~(d_sponge_target < self.cfg.place_dist_thresh)
rewards = rewards + self.cfg.lift_reward_scale * lift * post_place_gate * near_gate.float()
rewards = rewards + grasp_cond.float() * self.cfg.grasp_bonus_scale * post_place_gate * near_gate.float()
```
이렇게 하면 sponge가 target 근처 도달 시 hold path = 0/step, release path = +7/step → 압도적 우위.

### 학습 명령 (P6v4)
```bash
exec python -u -m roarm_rl.train_ppo \
    --task stack \
    --num_envs 4096 \
    --max_iterations 1000 \
    --reward_phase 6 \
    --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v3_no_gripper_in_placecond_grounded_resumeP6v2/model_999.pt \
    --reset_std 1.5 \
    --entropy_coef 0.001 \
    --experiment_name p6v4_release_path_reshape_resumeP6v3
```

### Success criterion
- `place_success_rate` > 0.05 (5% env에서 place 성공)
- `sponge_height` < 0.05m (sponge 내려놓기 학습 신호)

### Fallback (P6v4 또 실패 시)
- A2 #4 — P6a warm-start reset: 30% env에서 sponge를 target 근처 + gripper closed로 spawn. place 학습 가속.
- A2 #2 — rsl_rl `log_std_min/max` clipping (영구 std 제어).
- 보상 magnitude 재튜닝.

## HARD RULES 준수
- #11 /half-clone X
- #14 fail-fast guard 모든 ssh
- #15 cu128 sm_100 alive verify
- #17 visual RL X (state-only 28-dim)
- #18 사용자 명시 4 결정 (target/gravity/22→28/P4-P5-P6) 보존, P6 reward design 변경은 implementation detail
- #19 sponge edge-stand 47mm
- #20 # tower geometry
- #26 5/19 deadline 11일 ahead

## MEMORY HARD RULE #8 archive (다음 세션 진입 시)
현재 10+ entries (5 limit 위반). 가장 오래된 5 entries (5/05 night, 5/06, 5/07, 5/07 evening, 5/07 night 후보)를 `MEMORY_archive_20260508.md`로 본문 그대로 이동 + MEMORY.md엔 한 줄 pointer만 유지.

## 다음 세션 entry 명령 (P6v4 진행)
```
다음 세션 entry: claudedocs/phase1_balpha_p6v3_session_20260508_evening.md 읽고 P6v4 패치 3개 적용:
1) cfg에 lower_reward_scale: float = 5.0 추가 + gripper_open_bonus_scale 2.0→10.0
2) _get_rewards P6 마지막에:
   sponge_height_above = self._sponge_pos_w[:, 2] - TABLE_Z
   sponge_near = d_sponge_target < self.cfg.place_dist_thresh
   lower_reward = -sponge_height_above * self.cfg.lower_reward_scale
   rewards = rewards + lower_reward * sponge_near.float() * self._grasped.float()
3) lift_reward + grasp_bonus를 ~sponge_near gate (P5 reward 위치):
   near_gate = (~sponge_near).float()
   rewards = rewards + self.cfg.lift_reward_scale * lift * post_place_gate * near_gate
   rewards = rewards + grasp_cond.float() * self.cfg.grasp_bonus_scale * post_place_gate * near_gate

→ md5 transfer + sanity 16env×30step P6 → 학습:
ssh JHPark "set -e; source $ROARM_B200_ROOT/env.sh; ..." \
  python -m roarm_rl.train_ppo --task stack --num_envs 4096 --max_iterations 1000 \
  --reward_phase 6 --seed 0 \
  --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v3_no_gripper_in_placecond_grounded_resumeP6v2/model_999.pt \
  --reset_std 1.5 --entropy_coef 0.001 \
  --experiment_name p6v4_release_path_reshape_resumeP6v3

ETA ~7min. Success criterion = place_success_rate>0.05 AND sponge_height<0.05m.

또한 MEMORY.md HARD RULE #8 archive 정리: 가장 오래된 5 entries를 MEMORY_archive_20260508.md로 이동 (본문 그대로).
```
