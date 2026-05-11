# Phase 1.B-α P6 v2 (옵션 B + std reset + entropy ↓) — 5/08 late session

## TL;DR
- ✅ **std 발산 해결**: P6 v1 5.28→7.38 (발산) vs P6 v2 1.50→1.46 (안정). entropy_coef 0.005→0.001 + std force reset 1.5 동시 적용.
- ❌ **place_success_rate = 0.000 그대로** (1000 iter 동안 fire 0회). 옵션 B (place_dist_thresh 25→100mm) + std fix만으로는 부족.
- 🎯 **진짜 root cause 진단**: place_cond의 `gripper_open` AND condition. P5v2 학습된 policy의 action[5] bias=+0.798 (close), action_scale=0.1×N(0.798, 1.5)→gripper joint saturate close → `gripper_open` 조건 fire 0회.
- ⚡ **다음 단계 (A2 sim2real agent #1 추천)**: place_cond에서 gripper_open 제거 + `sponge_grounded` (z<table+30mm) 조건 추가 + gripper_open_when_near 별도 bonus reward 분리.

## 적용 패치 (5/08 late)

| 변경 | 파일 | 위치 | 효과 |
|---|---|---|---|
| place_dist_thresh 0.025→0.100 | roarm_stack_env.py | L219 | sponge_near 조건 통과 가능 (d=103mm > 100mm 살짝 초과) |
| `--reset_std` flag 추가 | train_ppo.py | L24-30 | resume 시 ckpt['std'] force overwrite |
| `--entropy_coef` flag 추가 | train_ppo.py | L31-33 | PPO algorithm.entropy_coef override |

## 학습 설정 (P6v2)
```bash
python -m roarm_rl.train_ppo \
    --task stack \
    --num_envs 4096 \
    --max_iterations 1000 \
    --reward_phase 6 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/roarm_stack_p5v2_1500iter_seed0_resumeP4_rewardfix/model_1499.pt \
    --reset_std 1.5 \
    --entropy_coef 0.001 \
    --experiment_name p6v2_thresh100mm_resetstd1p5_entropy0p001_resumeP5v2
```

- ckpts: `$ROARM_B200_ROOT/logs/roarm_rl/p6v2_thresh100mm_resetstd1p5_entropy0p001_resumeP5v2/` (model_0~999, 21 ckpts)
- log: `$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v2.{out,err}`
- wall: 6:40, throughput 244K steps/s

## 결과 (1000 iter 끝)

| 메트릭 | P6 v1 (이전, 2000 iter) | P6 v2 (방금, 1000 iter) | 진단 |
|---|---|---|---|
| action_std | 5.28→7.38 (발산) | 1.50→1.46 (안정) | ✅ std 발산 해결 |
| sponge_target_dist | 91mm | 103mm | 비슷 |
| grasped_frac | 0.93 | 0.94 | 비슷 |
| lift_success_rate | 0.11 | 0.18 | 약간 향상 |
| **place_success_rate** | **0.000** | **0.000** | ❌ fire 0회 |

## std 발산 trajectory (P3→P6 v1→P6 v2)

| Phase | iter | std first | std last |
|---|---|---|---|
| P3 (Pick fixed) | 500 | 2.68 | 3.19 |
| P4 (stack warmstart) | 500 | 3.19 | 3.92 |
| P5 v1 (nav stalled) | 500 | 3.92 | 4.55 |
| P5 v2 (reward fix) | 1500 | 3.92 | 5.28 |
| P6 v1 (place orig) | 2000 | 5.28 | **7.38** |
| **P6 v2 (this session)** | 1000 | **1.50** (reset) | **1.46** (entropy 0.001) |

P5v2 model_1499 per-joint std: `[3.46, 3.61, 4.46, 4.89, 7.64, 7.61]`
P6v2 reset 후: `[1.5, 1.5, 1.5, 1.5, 1.5, 1.5]`

## chicken-and-egg 진단 (5/08 late 확정)

```
P5v2 학습된 grasp policy:
  actor.6.bias[5] = +0.798  (gripper close 방향 push)

P6v2 stuck mechanism:
  매 step: gripper_joint += action_scale × action[5]
                         = 0.1 × clamp(N(0.798, 1.5), -1, 1)
                         ≈ 99% 시간 +값 (close 방향)
  → gripper_q saturate at max (close)
  → gripper_open (gripper_q < 0.4 rad) condition NEVER fires
  → place_cond = sponge_near AND gripper_open AND stable = NEVER True
  → place_bonus signal = 0
  → place 학습 진행 불가
```

## A2 sim2real research agent 핵심 SOTA findings (5/08)

### Isaac Lab 2025-2026 SOTA paper top-10
| arxiv | paper | 활용 |
|---|---|---|
| 2511.04831 | Isaac Lab GPU paper (NVIDIA, 2025-11-06) | `state_dependent_std` 옵션 reference |
| 2502.20396 | Sim-to-Real Vision-Based Dexterous Manipulation (Toru Lin/Yuke Zhu, 2025-02) | real-to-sim tuning + divide-and-conquer distillation |
| 2310.02743 | RLPD (Demo-augmented SAC) | Cold-start 해소, v6 50ep 활용 |
| 1709.10463 | DAPG (PPO + demo log-prob imitation) | PPO 유지하면서 warm-start |
| 2405.14523 | TRANSIC (RSS 2024) | Stacking 72% real, interactive correction |
| 2310.17688 | MimicGen (CoRL 2024) | 50 demo → 1000 sim demo, subtask 분리 |
| 2310.12931 | Eureka (ICLR 2025) | LLM-auto reward design |
| 2410.00425 | ManiSkill3 (2025) | DAPG/SAC manipulation baselines |
| 2602.16863 | SimToolReal (2026-02) | Procedural simulation + universal RL |
| 2504.12609 | Human2Sim2Robot (2025-04) | One human demo → sim-to-real RL |

### 우리 파이프라인 Gap 우선순위 (CRITICAL → MEDIUM)
| # | 패치 | Gap 심각도 | 적용 시간 | 다음 세션 우선 |
|---|---|---|---|---|
| **#1** | **place_cond에서 gripper_open 제거 + sponge_grounded 추가 + separate gripper_open_when_near bonus** | **CRITICAL** | 30min | 🥇 **즉시 적용** |
| #2 | rsl_rl `log_std_min/max` clipping (영구 std 제어) | CRITICAL | 30min | 🥈 #1 후 |
| #3 | nav_reward에 `lift_success_flag` 조건 추가 | HIGH | 15min | 같이 |
| #4 | P6a warm-start reset (30% sponge target 근처 spawn) | HIGH | 1-2h | curriculum |
| #5 | DAPG-style demo augmented loss (v6 50 real ep) | HIGH | 1-2일 | v6→sim mapping 후 |
| #6 | AsymPPO (privileged critic) | MEDIUM | 1-2일 | |
| #7 | Action delay DR | MEDIUM | 2h | sim-to-real 후 |
| #8 | entropy_coef linear decay 0.01→0.0001 | MEDIUM | 1h | |
| #9 | Eureka GPT-4o reward auto-design | MEDIUM | 도구 활용 | |

## 다음 세션 진입 시 즉시 할 일 (A2 #1 패치)

### 코드 변경 (`roarm_rl/roarm_stack_env.py`)

**`_place_condition` (L515-525)** — gripper_open 제거 + sponge_grounded 추가:
```python
def _place_condition(self, d_sponge_target: torch.Tensor) -> torch.Tensor:
    """Place success per-step condition.

    sponge near target (≤100mm) ∧ sponge grounded (z<table+30mm) ∧ stable (vel<5cm/s).
    NOTE: gripper_open 조건 제거 (Phase 1.B-α P6 v3 chicken-and-egg fix, 5/08 late).
    Reason: gripper_open이 close-fit policy의 action saturation에 의해 fire 안 됨.
    Separate gripper_open_when_near bonus로 gripper open 유도.
    """
    sponge_lin_vel = self._sponge.data.root_lin_vel_w
    sponge_stable = torch.norm(sponge_lin_vel, p=2, dim=-1) < 0.05
    sponge_near = d_sponge_target < self.cfg.place_dist_thresh
    sponge_grounded = self._sponge_pos_w[:, 2] < (TABLE_Z + 0.030)  # 30mm above table
    return sponge_near & sponge_stable & sponge_grounded
```

**`_get_rewards` P6 (L395-402)** — gripper_open_when_near 별도 bonus 추가:
```python
if self.cfg.reward_phase >= 6:
    place_cond = self._place_condition(d_sponge_target)
    rewards = rewards + place_cond.float() * self.cfg.place_bonus_scale

    # NEW: gripper_open_when_near separate bonus (encourages release)
    gripper_q = self._robot.data.joint_pos[:, self.gripper_joint_idx]
    gripper_open = gripper_q < self.cfg.grasp_gripper_thresh
    sponge_near = d_sponge_target < self.cfg.place_dist_thresh
    rewards = rewards + (gripper_open & sponge_near).float() * self.cfg.gripper_open_bonus_scale

    should_pay_place = self._place_success_flag & ~self._place_bonus_paid
    rewards = rewards + should_pay_place.float() * self.cfg.place_success_bonus
    self._place_bonus_paid = self._place_bonus_paid | should_pay_place
```

**Cfg (L215-225)** — gripper_open_bonus_scale 추가:
```python
# P6 place reward (P6 v3, 5/08 late)
place_dist_thresh: float = 0.100
place_bonus_scale: float = 5.0
gripper_open_bonus_scale: float = 2.0  # NEW: separate gripper open incentive when near target
place_success_bonus: float = 50.0
place_success_steps: int = 50
```

### 학습 명령 (P6v3)
```bash
ssh JHPark "set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z \$ROARM_B200_ROOT ]] && exit 1
[[ \$(whoami) != sogang_jhki ]] && exit 1
nohup bash /tmp/launch_p6v3.sh > \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v3.out 2> \$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v3.err < /dev/null &
"

# /tmp/launch_p6v3.sh 내용:
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
export OMNI_KIT_ACCEPT_EULA=YES
cd $ROARM_B200_ROOT/code
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

ETA: ~7 min, place_success_rate > 0이면 success.

## B200 환경 sanity (HARD RULE #14/#15)

- B200 GPU 0 UUID `c553ca20-377c-49dd-c30b-f5c530b3ff69` (Lenovo)
- ROARM_B200_ROOT = `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200`
- micromamba env `isaacsim_5_1` (cu128 nightly + sm_100)
- nvidia-smi NVML mismatch (driver 580.95.05 vs lib 580.159) — monitoring 영향, **컴퓨트는 정상** (P6v2 244K steps/s)
- 학습 background launch 패턴: `/tmp/launch_*.sh` script + `nohup bash /tmp/launch_*.sh ... < /dev/null &`

## HARD RULES 준수
- #11 /half-clone 거부 (96% context 경고 떴지만 continuation prompt + claudedocs + MEMORY로 세션 넘김)
- #14 fail-fast guard 모든 ssh
- #15 cu128 sm_100 alive verify
- #17 visual RL X (state-only 28-dim)
- #18 사용자 명시 4 결정 (target/gravity/22→28/P4-P5-P6) 보존
- #19 sponge edge-stand 47mm
- #20 # tower geometry
- #26 5/19 deadline 11일 ahead
