# Phase 1.B-α P6 v4 (release-path reshape) — 5/09 session

## TL;DR
- ✅ MEMORY HARD RULE #8 archive: 5 oldest entries (5/05 night, 5/06, 5/07, 5/07 evening, 5/07 night) → `MEMORY_archive_20260508.md` (12→7 full-bodies; user-specified 5 moved verbatim).
- ✅ P6v4 patch applied (3 changes in `roarm_rl/roarm_stack_env.py`).
- ✅ B200 transfer + md5 verify PASS (`2121ded14e479c9311db8c545f992bec`, local↔B200).
- ✅ Sanity 16env × 30step PASS.
- ✅ **P6v4 training COMPLETE** — PID 1775376, wall 6:19, 21 ckpts (model_0~999), 98.3M timesteps @ ~258K steps/s.
- ❌ **FAIL (branch C)**: `place_success_rate=0.0000` 1000 iter 내내. **sponge_target_dist 105→143mm** (멀어짐), **sponge_height 0.10→0.13m** (더 높이 hover). lift_success_rate 0.34→**0.76** (2× 상승, but 잘못된 방향).
- 🔴 **NEW DIAGNOSIS — chicken-and-egg #3 (cliff effect)**: `near_gate` 패치가 hold-path를 near zone에서 너무 강하게 끔 → 정책이 cliff를 인식하고 far zone에 머묾. 즉 **near zone 진입 자체가 불이익** → 탐험 멈춤.
- ➡️ **다음 세션**: A2 #4 P6a warm-start reset (30% env target 근처 spawn) — cliff 우회. 또는 near_gate 완화 (grasp 유지 + lift만 gate / smooth ramp).

## Patch summary (`roarm_rl/roarm_stack_env.py`)

| # | 변경 | 위치 | 효과 |
|---|---|---|---|
| 1 | `gripper_open_bonus_scale: 2.0 → 10.0` | cfg L221-224 | release path bonus 5× boost |
| 2 | `lower_reward_scale: float = 5.0` 신규 | cfg L225-228 | sponge_height penalty cfg |
| 3 | `_get_rewards`: `d_sponge_target` / `sponge_near` / `near_gate` up-front (L379-383) | env | gate 구조 정렬 |
| 4 | `lift_reward * post_place_gate * near_gate` (L394) | env | target 도달 시 lift bonus 0 |
| 5 | `grasp_bonus * post_place_gate * near_gate` (L397) | env | target 도달 시 grasp bonus 0 |
| 6 | P6: `lower_reward = -sponge_height_above × 5.0` gated by (sponge_near AND grasped) (L419-425) | env | descent 인센티브 |

목표 (보상 산수):
- target 도달 전 (`sponge_near=False`): hold path = lift(+5) + grasp(+2) + nav(small) ≈ +6.5/step
- target 도달 시 (`sponge_near=True`): hold path = nav(small) only ≈ -0.5/step → release path = gripper_open_when_near(+10) + place_bonus(+5 when grounded) + lower(-h×5) → +14/step (grounded) / +10/step (still high) — release strictly dominant.

## md5 / paths
- Local: `/home/cgxr/Documents/Robotics/RoArm_Project/roarm_rl/roarm_stack_env.py` (md5 `2121ded14e479c9311db8c545f992bec`)
- B200: `$ROARM_B200_ROOT/code/roarm_rl/roarm_stack_env.py` (md5 일치)
- Resume ckpt: `$ROARM_B200_ROOT/logs/roarm_rl/p6v3_no_gripper_in_placecond_grounded_resumeP6v2/model_999.pt` (1.2MB, May 8 16:21)
- Train logs: `$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v4.{out,err}`
- Output ckpts (target): `$ROARM_B200_ROOT/logs/roarm_rl/p6v4_release_path_reshape_resumeP6v3/`
- Launch script: `$ROARM_B200_ROOT/launch_p6v4.sh` (chmod +x, /tmp는 noexec라 ROARM_B200_ROOT 안에서 실행)

## 학습 명령 (실제 launch)
```bash
ssh JHPark
nohup $ROARM_B200_ROOT/launch_p6v4.sh \
  > $ROARM_B200_ROOT/logs/phase1Balpha/train_p6v4.out \
  2> $ROARM_B200_ROOT/logs/phase1Balpha/train_p6v4.err < /dev/null &
# PID 1775376 — 12:01:38 KST 5/09
```

내부 명령:
```bash
python -u -m roarm_rl.train_ppo \
    --task stack --num_envs 4096 --max_iterations 1000 --reward_phase 6 --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v3_no_gripper_in_placecond_grounded_resumeP6v2/model_999.pt \
    --reset_std 1.5 --entropy_coef 0.001 \
    --experiment_name p6v4_release_path_reshape_resumeP6v3
```

## Iter snapshots (5/09 12:00~12:08 KST, B200 P6v4 wall 6:19)

| iter | action_std | sponge_target_dist | sponge_height | grasped | lift_succ | place_succ | mean_rwd |
|---|---|---|---|---|---|---|---|
| 0 (resume) | 1.49 | 0.1056 m | 0.1008 m | 0.929 | 0.260 | 0.0000 | (resume) |
| 37 | 1.49 | 0.1054 m | 0.1008 m | 0.930 | 0.265 | 0.0000 | -158 |
| 184 (18%) | 1.46 | 0.1065 m | 0.1012 m | 0.930 | 0.273 | 0.0000 | -174 |
| **999 FINAL** | **1.31** | **0.1436 m** | **0.1321 m** | **0.934** | **0.764** | **0.0000** | **+664.58** |

**P6v3 final 비교**:
| 메트릭 | P6v3 final | P6v4 final | 변화 |
|---|---|---|---|
| sponge_target_dist | 0.108 m | **0.144 m** | ❌ +33% (멀어짐) |
| sponge_height | 0.104 m | **0.132 m** | ❌ +27% (더 높이) |
| lift_success_rate | 0.34 | **0.76** | 2.2× ↑ (잘못된 방향) |
| place_success_rate | 0.000 | **0.000** | 동일 (fire 0회) |

## 🔴 Diagnosis — chicken-and-egg #3 (cliff effect)

P6v4의 `near_gate` 패치가 hold-path를 near zone (d<100mm)에서 0으로 강제 cutoff → cliff 형성 → 정책이 cliff 인식하고 far zone에 머묾.

### Reward arithmetic (iter 999 평균 sponge_target=0.144m, sponge_height=0.132m, grasped=0.93)

far zone (d=0.144m > 0.10 thresh, near_gate=1):
- reach: -d_tcp_sponge × 1.0 ≈ -0.014/step
- lift: +5 × clamp(0.132, max=0.10) = +5 × 0.10 = **+0.50/step** (saturation)
- grasp: +2 (when grasped, 93%) = **+1.86/step**
- nav: -0.144 × 5 × grasped = **-0.67/step**
- action_penalty: -0.028/step
- **합 = +1.65/step × 399 = +658 ≈ +664 mean_reward 매칭** ✓

if 정책이 sponge를 near zone (예: d=0.05m, h=0.05m)으로 옮긴다면 (gripper closed):
- reach: -0.014/step
- lift: 0 (near_gate=0 cutoff)
- grasp: 0 (near_gate=0 cutoff)
- nav: -0.05 × 5 × 0.93 = **-0.23/step**
- gripper_open_when_near: 0 (closed)
- lower_reward: -0.05 × 5 × 0.93 (grasped) = **-0.23/step** (추가 penalty)
- place_bonus: 0 (not grounded yet)
- action_penalty: -0.028/step
- **합 = -0.50/step** (페널티!)

→ 멀리 있을 때 +1.65/step 받는 정책이 가까이 가면 -0.50/step. **near zone 진입은 negative reward**. PPO advantage가 음수 → 탐험 X. 정책은 lift_reward saturation을 maximize하고 nav_reward는 grasp_bonus 우위에 묻혀 영향 미미. 결과적으로 sponge를 더 높이 들고 더 멀어지는 방향 학습.

### Why P6v3 plateau hovered at 100mm vs P6v4 143mm?

P6v3는 hold path = +6.5/step everywhere (lift+grasp+nav 모두 active). nav가 d>0.10 이상에서 작동하면서 약하게 끌어당김. 결과 d≈100mm 근처 균형 (nav -d×5 == lift saturation 5 + grasp 2 = 7, near nav 0.5 균형).

P6v4는 near zone에서 hold path = 0 (cliff). far zone에서 +1.65/step이 plateau가 아니라 **계속 올라가는 reward** (lift saturation까지). 정책이 더 멀어져도 reward 손실 작음 (-0.05m × 5 = -0.25만 추가) but lift는 계속 +0.5/step 유지. 결과 d 조금씩 멀어지면서 lift는 saturation 유지가 stable.

## Hypothesis (다음 세션 사용자 confirm 필요)

**root cause**: near_gate가 너무 sharp cutoff (binary 1/0). PPO가 cliff edge에서 안정 fixed point 찾음.

**fix 후보**:

(α) **A2 #4 — P6a warm-start reset** ⭐ 권장: 30% env에서 episode reset 시 sponge를 target 근처 (d<100mm) + gripper closed + sponge_z=33mm로 spawn → 정책이 cliff edge 너머에서 출발하므로 cliff 영향 X. release+lower 학습 직접 가능. PPO가 그 success path를 generalize하면 cliff edge 넘어 진입.
- 구현: `_reset_idx`에서 30% env에 대해 sponge initial pose = target_pos_w + small_jitter, _grasped = True, gripper joint = closed.
- 위험: warm-start env가 너무 쉬워서 실제 grasp 학습 잊을 수 있음 → curriculum (initial 50% warm-start → 점차 0%).

(β) **near_gate 완화 — smooth ramp**: `near_gate = sigmoid((d - 0.05) × 50)` → cliff 대신 부드러운 transition. d=0.10에서 ~0.99 (거의 켜짐), d=0.05에서 0.5, d=0에서 ~0.01.
- 장점: 코드 변경 1줄.
- 단점: cliff는 완화되지만 release path 학습 신호는 여전히 약함.

(γ) **grasp_bonus는 게이트 X (lift만 게이트)**: P6v3 처럼 grasp_bonus 항상 +2. 이러면 near zone 진입해도 +2/step 보존되어 cliff 약함.
- 보상: near zone hold = grasp(+2) + nav(-0.25) = +1.75 vs far zone +1.65 → near zone 약간 더 좋음. 정책이 가까이 갈 인센티브 발생.
- 단점: P6v3에서 같은 hold_path가 release를 압도했었음. 같은 issue 재발 가능.

(δ) **gripper_open_bonus_scale 추가 강화 (10 → 30)**: release 보상을 hold 압도하도록 강화.
- near zone open = +30 - 0.5 (nav+lower) = +29.5/step (vs far zone +1.65). 강한 incentive.
- 단점: gripper가 closed에서 open으로 transition하는 매 step에 보상이 +30 → 정책이 sponge release 한 후 open 유지하기만 해도 reward farming 가능. _place_bonus_paid처럼 latch 필요.

**추천 조합 (권장 순)**: α (warm-start) > δ (gripper_open 30) + γ (grasp 항상) > β (smooth ramp). α가 cliff exploration 문제를 근본 해결.

## 다음 세션 entry — 즉시 명령
```
다음 세션 진입 시 P6v4 학습 종료 결과 확인:

1) ssh JHPark "ps -p 1775376 -o pid,etime 2>&1 | head -2"
   (없으면 종료 완료)

2) ssh JHPark "tail -80 /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/logs/phase1Balpha/train_p6v4.out"
   - 최종 iter ~1000 metrics 확인:
     - place_success_rate (target >0.05)
     - sponge_height_m (target <0.05m)
     - sponge_target_dist_m (target <60mm)
     - action_std (target stable ~1.5, divergence X)

3) ssh JHPark "ls -la \$ROARM_B200_ROOT/logs/roarm_rl/p6v4_release_path_reshape_resumeP6v3/" 
   - 21개 ckpt 확인 (model_0~999)

4) bash run_in_background ID b2402qtj6 출력도 확인:
   - /tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/edcca341-4052-4167-b6f2-002dbbfc8911/tasks/b2402qtj6.output

분기:

(A) SUCCESS: place_success_rate>0.05 AND sponge_height<0.05m
    → A2 #2 (rsl_rl log_std_min/max clipping 영구 std 제어) 적용 + place_dist_thresh
       100→50→25mm curriculum squeeze. 다음 세션 P7 (squeeze).

(B) PARTIAL: place_success_rate>0.01 OR sponge_height<0.08m (감소 시작)
    → P6v4 ckpt에서 1000 iter 추가 학습 (resume p6v4 model_999, 동일 설정).

(C) FAIL: place_success_rate=0 AND sponge_height>0.08m (P6v3와 동일 plateau)
    → A2 #4 P6a warm-start reset 도입:
       roarm_stack_env.py:_reset_idx에서 30% env에 대해 sponge를 target 근처
       (place_dist_thresh 안) + gripper closed로 spawn → place 학습 가속.
       또는 보상 magnitude 추가 재튜닝 (gripper_open_bonus_scale 10→20,
       lower_reward_scale 5→10).

(D) DIVERGE: action_std >2.5 또는 reward 발산
    → A2 #2 log_std_min/max clipping 즉시 적용 (먼저).
```

## HARD RULES 준수
- #8 archive 1단계로 처리 (5 entries → MEMORY_archive_20260508.md, 본문 그대로)
- #11 /half-clone 거부 1회 (Stop hook 101% 거부, continuation prompt + claudedocs로 처리)
- #14 fail-fast guard 모든 ssh 적용 (`set -e; source env.sh; [[ -z $ROARM_B200_ROOT ]] && exit 1; [[ $(whoami) != sogang_jhki ]] && exit 1`)
- #15 cu128 sm_100 alive (이전 세션 verified, 본 세션 학습 정상 진행 = 검증)
- #17 visual RL X (state-only 28-dim only)
- #18 사용자 명시 4 결정 (target Y=-0.0435 / gravity / 22→28-dim / P4-P5-P6) 그대로 유지. P6v4 reward 재설계는 implementation detail
- #19 sponge edge-stand 47mm
- #20 # tower geometry
- #26 5/19 deadline 10일 ahead

## 트랩 발견
- `/tmp` mount = `noexec` (B200 컨테이너). nohup으로 `/tmp/launch_*.sh` 실행 시 "Permission denied". `$ROARM_B200_ROOT/launch_*.sh`로 옮긴 후 정상 실행. 다음 세션 launch에도 동일 패턴 적용.
