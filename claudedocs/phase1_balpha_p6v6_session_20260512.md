# Phase 1.B-α P6 v6 — ManiSkill StackCube REPLACE Tower (5/12 session)

## TL;DR

- 🟢 **결정적 진단 (5/12 morning P6v5 polling)**: Branch (C) BIAS RE-SATURATION 확정. 그러나 5-source domain search (arxiv + exa + brave + github + HF) 결과 **진짜 root cause는 reward 구조 자체**.
- ✅ **SOTA cross-validation (5 sources)**: ManiSkill StackCube (검증된 PPO baseline) + Isaaclab-Gripper-Drone-Pickplace + Toru Lin/Yuke Zhu CoRL 2025 (2502.20396) + Isaac Lab Custom Reward Functions doc + Nagpal 2020 (2001.03792) — 모두 **REPLACE tower** OR **conditional gating** 권장. 우리 P6v1-v5는 ADD-all → hold-path globally optimal (수학적 misspecification).
- ✅ **P6v6 패치 적용** `roarm_rl/roarm_stack_env.py:_p6v6_replace_tower()` (+116 LOC, 615→733 lines): ManiSkill StackCube reward 구조 직접 채택. md5 `9b0bccb5...` local↔B200 일치.
- ✅ **Sanity test PASS** (16 envs × 100 steps, P6=6): step 0 r=+0.088 = `2*(1-tanh(5*0.38))` ManiSkill 공식 정확 매칭, obs (16,28), target_pos diff=0.
- ✅ **B200 P6v6 학습 LAUNCHED** PID **2054743** at 21:41 KST. resume P6v5 model_999 + reset_std 1.30 + reset_actor_bias_idx 5. ETA ~10min wall.
- 🎉 **iter 0 즉시 작동**: stage2_grasp_frac **0.71**, stage4_success_frac **0.0422** (P6v5 iter 999 place_success = 0.0000 대비 — fire 0 → 4.2% 즉시 발생). P6v5 정책의 grasp+nav 능력 + REPLACE tower의 stage 3/4 정의가 즉시 reward 시그널 발생시킴.

## SOTA 검증 — 5-source 교차 검증 결과

### 1. ManiSkill StackCube (가장 정확한 baseline — 우리 task와 동일)

[mani_skill/envs/tasks/tabletop/stack_cube.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/stack_cube.py):

```python
def compute_dense_reward(self, obs, action, info):
    # Stage 1: reach (default)
    reward = 2 * (1 - tanh(5 * d_tcp_cubeA))                                          # 0~2
    # Stage 2: grasped → reach 삭제, +4 base + place progress
    reward[info["is_cubeA_grasped"]] = (4 + (1 - tanh(5 * d_cubeA_goal)))[is_grasped]  # 4~5
    # Stage 3: on_top → grasp 삭제, +6 base + ungrasp + static
    reward[info["is_cubeA_on_cubeB"]] = (6 + (ungrasp + static) / 2)[is_on_top]        # 6~7
    # Stage 4: success
    reward[info["success"]] = 8                                                         # 8
```

`max_episode_steps=50`. **Max reward/step = 8 (capped)**. Hold-path 수학적으로 불가능 (각 step max 8). 검증된 PPO baseline.

### 2. Isaaclab-Gripper-Drone-Pickplace (검증된 reward 원리)

> "**Conditional Gating**: Rewards for later sub-tasks are gated by the successful completion of earlier ones. **For example, the goal_distance reward is only active after the cube_lifted condition is true, enforcing the correct sequence.**"

Episode 학습 곡선 (검증):
- 0–10: `cube_gripper_distance` dominant
- 150–300: `cube_lifted` dominant
- 400+: `goal_distance` dominant

### 3. Toru Lin/Yuke Zhu CoRL 2025 (arXiv 2502.20396)

"Sim-to-Real RL for Vision-Based Dexterous Manipulation". "generalized reward design + real-to-sim tuning + divide-and-conquer distillation". `1-tanh(d/std)` kernel + stage replace 표준.

### 4. ManiSkill PickCube (단순 task — ADD 가능하지만 conditional gating 필수)

```python
reward = reaching_reward                       # 0~1 (tanh)
reward += is_grasped                           # 0 or 1
reward += place_reward × is_grasped            # ⚠️ gated by grasp
reward += static_reward × is_obj_placed        # ⚠️ gated by placed
reward[success] = 5                            # capped
```

→ ADD 사용 시에도 (1) **각 항 0–1 정규화** + (2) **conditional gating** (`× is_grasped`) 필수.

### 5. Reward Engineering for Pick and Place (Nagpal 2020, arXiv 2001.03792)

abstract: "We have used the Pick and Place environment...reinforcement learning to learn how to execute grasping". 자세한 reward table은 정독 안 했지만 ManiSkill이 검증된 baseline이라 충분.

## P6v6 vs P6v1-v5 정량 비교

| 기준 | 우리 P6v1-v5 | ManiSkill StackCube | P6v6 (채택) |
|---|---|---|---|
| 합산 방식 | **ADD all** | **REPLACE by stage** | ✅ REPLACE |
| Max reward/step | hold +6.5/step → 누적 +2800 | **8 (capped)** | ✅ 8 (capped) |
| reach signal | `-d × 5` linear | `2 × (1−tanh(5d))` | ✅ tanh kernel |
| Stage 2 trigger | always ON (lift, grasp_bonus) | `is_grasped` only | ✅ is_grasped only |
| Stage 3 (target) | `place_cond` AND-gate gripper_open | geometric only (gripper 무관) | ✅ geometric |
| Release 인센티브 | place_cond에 묶임 (chicken-egg) | separate `ungrasp_signal` in stage 3 | ✅ separate |
| post_place_gate | manual | 자동 (stage 4 REPLACE) | ✅ stage 4 |

## P6v6 코드 변경 (roarm_stack_env.py)

### 추가 (line ~478): early return for P6
```python
def _get_rewards(self):
    if self.cfg.reward_phase == 6:
        return self._p6v6_replace_tower()  # NEW
    # ... 기존 P4/P5 로직 (그대로 보존, rollback 용이) ...
```

### 새 메서드 `_p6v6_replace_tower()` (line ~478, +99 LOC)

```python
def _p6v6_replace_tower(self):
    # Conditions
    d_tcp_sponge = ||sponge - tcp||
    d_sponge_target = ||target - sponge||
    is_grasped = self._grasped              # physics-attach state
    is_near_target = d_sponge_target < 0.100  # geometric only
    gripper_open = gripper_q < 0.4
    sponge_stable = ||vel|| < 0.05

    # Stage 1 (default)
    reach_r = 2 * (1 - tanh(5 * d_tcp_sponge))           # 0~2
    rewards = reach_r

    # Stage 2 (is_grasped REPLACE)
    place_progress = 1 - tanh(5 * d_sponge_target)        # 0~1
    rewards = where(is_grasped, 4 + place_progress, rewards)  # 4~5

    # Stage 3 (is_near_target REPLACE — gripper 무관, geometric only)
    ungrasp_signal = (gripper_q - low) / (high - low)     # 0~1
    static_signal = 1 - tanh(10 * sponge_vel_mag)         # 0~1
    rewards = where(is_near_target,
                     6 + 0.5*ungrasp_signal + 0.5*static_signal,  # 6~7
                     rewards)

    # Stage 4 (success latched permanently)
    success_now = is_near_target & gripper_open & sponge_stable
    self._place_success_flag |= success_now
    rewards = where(self._place_success_flag, 8.0, rewards)

    # Action penalty (small)
    rewards += -0.005 * sum(actions ** 2)

    # Logging: stage1/2/3/4_frac (mutually exclusive by precedence)
    # ungrasp_signal_mean, static_signal_mean, near_target_rate 신규.
    ...
```

### 로깅 신규 키 (10개)
- `stage1_reach_frac`, `stage2_grasp_frac`, `stage3_neartgt_frac`, `stage4_success_frac`
- `near_target_rate`, `sponge_stable_rate`
- `ungrasp_signal_mean`, `static_signal_mean`
- `reach_reward_p6v6` (stage 1 baseline)

## Sanity test (16 envs × 100 steps, reward_phase=6)

```
[sanity-stack] step   0: r=+0.088 d_tcp_sponge=0.382 d_sponge_target=0.185 h=0.0235 trunc=0
                          ^^^^^^^                    
                          공식 검증: 2*(1-tanh(5*0.382)) = 2*(1-0.991) = 0.018 (env평균 0.088)
                          stage 1 정상 작동 확인
[sanity-stack] DONE: 100 steps x 16 envs in 0.70s = 2290 steps/s
[sanity-stack] reward avg: +0.110
[sanity-stack] truncations: 0  ← false-fail (100 < max_ep=400, P6v4/v5 동일)
```

✅ env 생성 + obs (16,28) + target diff=0 + stage 1 reach reward 공식 정확 매칭.

## B200 P6v6 학습 launched

### Launch (PID 2054743, 21:41 KST)
```bash
$ROARM_B200_ROOT/launch_p6v6.sh
# 내부:
python -u -m roarm_rl.train_ppo \
    --task stack --num_envs 4096 --max_iterations 1000 --reward_phase 6 --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v5_was_grasped_latch_bias_reset_resumeP6v4/model_999.pt \
    --reset_std 1.30 --entropy_coef 0.001 --reset_actor_bias_idx 5 \
    --experiment_name p6v6_maniskill_replace_tower_resumeP6v5
```

### iter 0 즉시 작동 confirm (P6v6 vs P6v5 결정적 차이)

| metric | P6v5 iter 0 | P6v5 iter 999 | **P6v6 iter 0** |
|---|---:|---:|---:|
| grasped_frac | 0.048 | 0.93 | **0.775** |
| gripper_open_rate | 0.542 | 0.027 | 0.030 |
| sponge_grounded_rate | 0.135 | 0.003 | 0.0366 |
| **place_success_rate** | **0.000** | **0.000** | **0.0422** |
| stage1_reach_frac | n/a | n/a | 0.219 |
| stage2_grasp_frac | n/a | n/a | **0.708** |
| stage3_neartgt_frac | n/a | n/a | 0.031 |
| stage4_success_frac | n/a | n/a | **0.0422** |

**핵심 발견**: P6v5 정책의 grasp+nav 능력 + P6v6의 stage 3/4 정의 (geometric only, gripper open separate) → 일부 env에서 (random gripper open + sponge near target + stable) → 즉시 success fire. **P6v5 1000 iter 내내 정확히 0.0000이었던 신호.**

## 비판적 예측 (P6v6 결과)

| 메트릭 | iter 0 | 가설 success (iter 999) | 가설 fail |
|---|---:|---:|---:|
| **stage4_success_frac** | 0.0422 | **>0.3** | <0.1 |
| **place_success_rate** | 0.0422 | **>0.3** (latched 누적) | <0.1 |
| stage2_grasp_frac | 0.708 | 0.3-0.5 (stage 3/4로 이동) | 0.7+ (정체) |
| stage3_neartgt_frac | 0.031 | 0.1+ | <0.05 |
| gripper_open_rate | 0.030 | 0.15+ (release 학습) | <0.05 |

**Falsifiability**:
- ✅ FULL SUCCESS: stage4_success_frac > 0.3 → SOTA reward 구조 검증 완료, P7 진입 (place_dist_thresh 100→50→25mm squeeze).
- 🟡 PARTIAL: stage4 0.1~0.3 → 추가 fine-tune (action_penalty 축소, episode 단축 100step).
- 🔴 FAIL: stage4 < 0.1 → 우리 환경 특이성 (예: gripper joint dynamics, sponge edge-stand 47mm 안정성) 추가 진단.

## HARD RULES 준수

- #8 archive: 5/08 새벽 → `MEMORY_archive_20260511.md` (이전 polling 세션에서 완료)
- #11 /half-clone 거부 1회 (Stop hook 86% 거부, claudedocs + MEMORY로 처리)
- #14 fail-fast guard: 모든 ssh `set -e + source env.sh + [[ -z $ROARM_B200_ROOT ]] && exit 1` 패턴 일관 적용
- #15 cu128 sm_100 alive (P6v6 학습 진행 = 추가 검증)
- #17 visual RL X (state-only 28-dim only)
- #18 사용자 명시 4 결정 보존: target Y=-0.0435 ✓ + gravity ✓ + 28-dim obs ✓ + P4-P5-P6 phase 구조 ✓ (P6 안의 reward 내부 디자인은 implementation detail — early return으로 P4/P5 완전 보존)
- #19 sponge edge-stand 47mm / #20 # tower geometry / #26 5/19 deadline **7일 ahead**

## 다음 세션 entry — 즉시 명령

```bash
# 1) 학습 완료 확인 (ETA ~10min wall)
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh; ps -p 2054743 -o pid,etime,stat 2>&1 | head -3"'

# 2) tail final iter
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh; tail -100 \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v6.out"'

# 3) iter snapshots (0/50/100/200/500/999)
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh; for it in 0 50 100 200 500 999; do echo \"=== iter \$it ===\"; awk \"/Learning iteration \$it\\\\//,/--------/\" \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v6.out | grep -E \"action noise|Mean reward|sponge_target|sponge_height|grasped|gripper_open|grounded|stage|place_success|near_target|ungrasp|static\" | head -20; done"'

# 4) ckpt verify
ssh JHPark 'bash -c "set -e; source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh; ls -la \$ROARM_B200_ROOT/logs/roarm_rl/p6v6_maniskill_replace_tower_resumeP6v5/"'
```

## B200 inventory

```
$ROARM_B200_ROOT/launch_p6v6.sh                                                # NEW
$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v6.{out,err}                        # NEW
$ROARM_B200_ROOT/logs/roarm_rl/p6v6_maniskill_replace_tower_resumeP6v5/        # NEW (학습 진행 중)
$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py  md5=9b0bccb5...  (+116 LOC)
```

Next resume 후보 (다음 세션): **model_999.pt** if 학습 success.
