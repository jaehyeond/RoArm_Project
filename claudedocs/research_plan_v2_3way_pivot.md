# ResearchPlan v2 — 3-way Comparison Pivot

**작성**: 2026-05-05 night
**연구 방향 lock-in**: HARD RULE #21 (3-way: Pure VLA / Pure Isaac Lab RL / Real-to-sim Hybrid)
**B200 데드라인**: 2026-05-19 (~2주 잔여)
**사용자 명시 우선순위**: B200 활용 극대화 + OpenVLA/π0 백업 자산화 + external reference repo 활용

---

## Phase 0: Research Direction Lock-In (DONE)

### 3-way 비교

| Approach | 목적 | Status |
|---|---|---|
| **#1 Pure VLA (BC)** | BC baseline, 1-grasp only | v6_base READY |
| **#2 Pure Isaac Lab RL** | RL from scratch, 데모 X | TODO Day 1-9 |
| **#3 Real-to-sim Hybrid** | BC warmstart + RL finetune (DAPG/AWAC/SimpleVLA-RL) | TODO Day 4-9 |

### 평가 차원

- **Sim eval**: success rate (sponge correctly stacked), episode length, action smoothness, reward curve
- **Real eval (deploy)**: same metrics + sim2real gap
- **Statistical**: 3 seeds × 3 approaches = 9 conditions
- **추가 차원 (시간 남으면)**: SmolVLA / OpenVLA / π0 모델 비교 on best approach

---

## Phase 1: Isaac Lab/ManiSkill3 RL Infrastructure (Day 1-3, 4090 dev + B200 dry-run)

### Day 1 — ManiSkill3 install + RoArm URDF 통합

**4090 dev** (Lenovo, cgxr@cgxr-Legion-Pro-7-16IRX9H):

```bash
# new conda env (avoid roarm conflict)
conda create -n roarm_rl python=3.10 -y
conda activate roarm_rl

# ManiSkill3 install (PPO/SAC + StackCube task)
pip install --upgrade mani_skill
# verify: 30k FPS on 4090 expected
python -m mani_skill.examples.demo_random_action -e StackCube-v1
```

**Tasks**:
1. ManiSkill3 install + verify (PPO baseline run)
2. RoArm M3 URDF 위치 확인 (`lerobot_backup/` 또는 `sim_scripts/` 참조)
3. ManiSkill3에 RoArm URDF import (BaseEnv subclass)
4. Joint mapping (6-DOF: base/shoulder/elbow/wrist_p/wrist_r/gripper) verify
5. Action space: 6-DoF Δjoint (continuous) — ManiSkill StackCube와 동일 패턴

**Verify**:
- 4090 random-action episode 1개 정상 종료
- Episode reset → HOME [0,0,90,0,0,5]
- Action range = JOINT_LIMITS (CLAUDE.md 참조)

### Day 2 — RoArmStackTask scaffolding

**Goal**: ManiSkill3 StackCube 패턴으로 sponge stacking task 정의

**파일 구조**:
```
sim_scripts/
└── roarm_rl/
    ├── __init__.py
    ├── roarm_robot.py        # ManiSkill3 BaseAgent subclass + RoArm URDF
    ├── roarm_stack_task.py   # ManiSkill3 BaseEnv subclass — # tower task
    ├── reward_shaping.py     # sparse outcome + dense distance/grasp
    └── configs/
        ├── ppo_config.yaml
        └── sac_config.yaml
```

**Task 정의 (HARD RULE #19/#20 준수)**:
- Sponges: 4 sources (edge-stand 47mm tall, 22mm width, 125mm long), Layout S1/S2/S3/S4 random spawn
- Goal: # tower at center (Layer 1: X-axis c2c=87mm, Layer 2: Y-axis c2c=67mm)
- Episode max: 200 steps
- Success: 4 sponges placed correctly (z + xy 모두 within ±15mm tolerance)

**Reward shaping (DEMO3 + ManiSkill3 StackCube 패턴)**:
- Sparse outcome (+1 per sponge correctly placed, +5 final)
- Dense distance (TCP→source / TCP→target, normalized)
- Grasp shaping (gripper close at source proximity → reward)
- Safety penalty (joint limit violation → -1)

### Day 3 — PPO/SAC dry-run (4090 + B200)

**4090 dry-run** (1 hour, single seed):
```bash
python -m mani_skill.examples.baselines.ppo --env-id RoArmStack-v0 \
  --num-envs 8 --total_timesteps 100_000 --update-epochs 4
```

**B200 dry-run** (sogang_jhki@JHPark-container, fail-fast guard):
- Set up roarm_rl env on B200 (HARD RULE #15 nightly cu128 verify)
- Single seed PPO 1h smoke test
- Reproducibility check: 4090 vs B200 first 10K loss curve byte-comparable?

**Decision point Day 3 EOD**:
- ✅ PPO/SAC running stable → proceed to Day 4 real teleop
- ❌ Issues → Day 4 reserved for debugging, push timeline

---

## Phase 2: Real Stacking Teleop Data (Day 4-5)

### Day 4 — L-F teleop 30-50ep

**Pre-flight**:
- USB0=Leader (gripper clamp 팔 #1) + USB1=Follower (palm 팔 #3) verify
- Azure Kinect calib (4/15 RMSE 10.13mm) 확인
- 4 sponges edge-stand 47mm tall (HARD RULE #19)
- Recording: collect_data_manual.py L-F mode

**수집 시나리오** (다양화 권장):
- 10ep: S1→L1.sp1, S2→L1.sp2, S3→L2.sp1, S4→L2.sp2 (default order)
- 10ep: 다양한 source 위치 + first-grasp 골고루 (S1/S2/S3/S4 균일)
- 10ep: failure recovery (sponge 떨어뜨리면 재집기)
- 10ep (옵션): 3-sponge subset (early stop, partial completion)

**Target**: 30-50 episodes, 6-10K total frames
**예상 시간**: 4-5h (1 ep ~5-7min including reset)

### Day 5 — convert_to_lerobot_v3 + dataset build

```bash
python convert_to_lerobot_v3.py \
  --input collected_data/real_stacking \
  --task "Stack four pink sponges into a # pattern" \
  --output lerobot_dataset_real_stacking_v1
```

**Verification**:
- 30-50ep × ~150fr = 4500-7500 frames
- Task index 0
- AV1 video, observation.images.top (single Kinect)
- HARD RULE #16: train_config.json source-of-truth check

**B200 rsync**: ~50-80MB → JHPark-container (HARD RULE #14 fail-fast guard)

---

## Phase 3: 3-way Training (Day 6-9, B200 main)

### B200 GPU 예산 추정 (Day 6-9, 80h total, 5 days × ~16h/day)

| Approach | Algorithm | Steps | Seeds | Time/seed | Total B200 hours |
|---|---|---|---|---|---|
| #1 Pure VLA | (재학습 X) | — | — | — | 0h |
| #2 Pure Isaac RL | PPO (ManiSkill3) | 5M-10M | 3 | ~6-8h | ~24h |
| #3 Hybrid | DAPG (PPO + BC loss) | 5M | 3 | ~8-12h | ~36h |
| Backup | SimpleVLA-RL OpenVLA SFT→RL | — | 1 | ~10-15h | ~15h |
| **Total** | | | | | **~75h** |

### Day 6 — #2 Pure Isaac RL training (3 seeds)

```bash
# B200 (sogang_jhki@JHPark)
set -e
source ~/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && exit 1
[[ "$(whoami)" != "sogang_jhki" ]] && exit 1
[[ "$(hostname)" != "JHPark"* ]] && exit 1

cd $ROARM_B200_ROOT
for seed in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 python -m mani_skill.examples.baselines.ppo \
    --env-id RoArmStack-v0 --seed $seed \
    --num-envs 64 --total_timesteps 5_000_000 \
    --output-dir outputs/pure_rl_seed${seed} &
done
```

**Monitor**: Day 6 evening — first seed converged?

### Day 7-8 — #3 Hybrid training

DAPG patch (~30-50 lines) on ManiSkill3 PPO:
```python
# pseudo-code
loss_pg = ppo_clipped_loss(...)
loss_bc = -log_prob(demo_actions, demo_states)  # demos from v6+real_stacking
loss_total = loss_pg + lambda_bc * loss_bc  # lambda_bc decay
```

**3 seeds × 5M steps × ~10h**:
```bash
for seed in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 python -m roarm_rl.dapg \
    --env-id RoArmStack-v0 --seed $seed \
    --demo-dataset lerobot_dataset_v6_real_stacking_merged \
    --num-envs 64 --total_timesteps 5_000_000 \
    --output-dir outputs/hybrid_seed${seed} &
done
```

### Day 9 — Backup VLA+RL (시간 허용 시)

옵션 1: **SimpleVLA-RL** with SmolVLA wrapper
- Clone PRIME-RL/SimpleVLA-RL
- Wrap SmolVLA action expert in their RL pipeline
- SFT (v6+real_stacking) → RL (~10-15h B200)

옵션 2: **OpenVLA-OFT or π0 finetune**
- LeRobot-side: lerobot-train --policy.type=π0 (지원 시) on merged data
- 단순 BC 추가 (RL X) → 모델 자산화

**시간 부족 시**: Day 9는 monitoring + Day 6-8 학습 결과 spot-check + 다음 phase 준비.

---

## Phase 4: 3-way Evaluation (Day 10-12)

### Day 10 — Sim evaluation

각 seed × approach (9 conditions) sim eval (ManiSkill3 환경):
- Success rate over 100 random init layouts
- Mean episode length
- Action smoothness (joint velocity std)
- Reward curve (training)

```bash
for approach in pure_vla pure_rl hybrid; do
  for seed in 1 2 3; do
    python -m roarm_rl.eval --ckpt outputs/${approach}_seed${seed} \
      --num-eps 100 --output-csv eval_${approach}_seed${seed}.csv
  done
done
```

### Day 11 — Real deployment

각 approach 1 best seed × 5 episodes deploy on real RoArm M3:
- USB1 Follower + Azure Kinect
- 4 sponges edge-stand 다양 layout
- Closed-loop deploy (n_action_steps=1)
- Per-step CSV log + frames MP4

**Conditions**:
- #1 Pure VLA: v6_base, task="Stack four pink sponges into a # pattern"
- #2 Pure RL: best seed sim ckpt → real deploy (sim2real gap 측정)
- #3 Hybrid: best seed sim ckpt → real deploy

### Day 12 — Statistical analysis + 정량표

ANOVA / paired t-test across 9 conditions:
- Success rate (sim) — main effect of approach
- Sim2real gap (sim success - real success) — interaction effect
- Action smoothness — quality dimension

Final figure: 3 bars × 2 environments (sim/real) + error bars.

---

## Phase 5: Backup VLA Models + Paper Writing (Day 13-14)

### 시간 남으면

옵션 A: **OpenVLA / π0 hybrid 학습** (가장 가치 高)
- Best approach (예상: #3 hybrid) 에 OpenVLA backbone 적용
- B200 ~10-15h (한 모델만)

옵션 B: **추가 seeds for statistical power**
- 각 approach 5 seeds로 확장

옵션 C: **Long-horizon ablation**
- 3-step subset (3 sponges) 추가 학습 + 평가

### Paper section drafting (parallel)

- Abstract + Introduction
- Method: 3-way pipeline
- Experiments: 9 conditions table
- Results + Discussion
- Limitations: 2-week sprint, single robot platform, sponge-specific

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ManiSkill3 RoArm URDF integration 1 day 초과 | MED | timeline +2-3일 | Day 0 pre-check (URDF 위치, joint count) |
| PPO 5M steps stacking task 부족 | MED | success rate <50% | Reward shaping 강화, num_envs ↑ |
| DAPG demo loss balance 어려움 | HIGH | hybrid #3 fails | AWAC 또는 SimpleVLA-RL 백업 알고리즘 |
| Sim2real gap 너무 큼 (#2/#3) | HIGH | real deploy 0% | Domain randomization 추가, real fine-tune step |
| B200 잔여 정확히 14일 X | LOW-MED | 학습 중단 | Day 0 잔여 시간 정확 확인, plan 압축 |
| Real teleop 30-50ep 시간 부족 | LOW | demo data 적음 | Day 4 evening checkpoint, 30ep만으로 진행 |

---

## Critical Decision Points

### Day 3 EOD: Infrastructure ready?
- ✅ PPO/SAC running stable + RoArm URDF integrated → Day 4 진입
- ❌ ManiSkill3 RoArm bug → 1 day debug + push timeline

### Day 5 EOD: Real teleop done?
- ✅ 30+ ep collected → Day 6 Hybrid training 시작
- ❌ Hardware issue → Day 6 reserved for debugging

### Day 9 EOD: Training results review
- ✅ All 3 approaches converged → Day 10 evaluation
- ❌ Some approach still training → Day 10-12 phase rebalance

### Day 12 EOD: Results
- ✅ Clear 3-way comparison → Day 13-14 backup VLA + paper draft
- ❌ Some approach failed → analyze why, adjust paper narrative

---

## Reusable Assets (Day 1 시작 전 확인)

| Asset | 위치 | 용도 |
|---|---|---|
| v6 BC model | `outputs/smolvla_v6_b200/checkpoints/last/pretrained_model` | #1 Pure VLA + #3 BC warmstart |
| v3 BC ckpts (5K~20K) | `outputs/smolvla_v6_stacking_v3_b200/checkpoints/{005000,010000,015000,020000}` | Paper "BC fail" baseline |
| v6 dataset (real teleop) | `lerobot_dataset_v6/` | #3 demo source |
| Sim env (Isaac Sim) | `sim_scripts/{stacking_scene,render_stacking_demos}_v3.py` | ManiSkill3 통합 reference |
| Kinect calib | `sim_scripts/kinect_calib.yaml` | Real eval setup |
| Table plane | `sim_scripts/table_plane.json` | Sim env z=0 anchor |
| Sponge poses | `sim_scripts/sponge_poses.json` | Sim spawn 다양화 |
| RoArm URDF | TBD (Day 1 확인 필요) | ManiSkill3 BaseAgent |
| γ diagnostic | `vision_conditioning_diagnostic.py` + `capture_real_layouts_for_gamma.py` | All approach 평가 |
| 15 real layouts | `data/real_layouts_20260505_170501/` | γ re-test |

---

## 다음 step (사용자 confirm 후)

1. **Day 1 시작**: ManiSkill3 install + RoArm URDF 위치 확인 + integration scaffolding
2. **Reference repo clone** (병렬):
   - `git clone https://github.com/haosulab/ManiSkill`
   - `git clone https://github.com/PRIME-RL/SimpleVLA-RL` (Day 9 backup용)
3. **B200 잔여 확인**: 사용자 → Sogang/NHN 임대 잔여 정확 일수 확인
4. **Day 1 Critical task list**:
   - [ ] roarm_rl conda env 생성
   - [ ] ManiSkill3 install + StackCube random-action verify
   - [ ] RoArm M3 URDF locate + ManiSkill3 BaseAgent integration scaffolding
   - [ ] Action space + observation space 정의
   - [ ] 4090 첫 PPO smoke test (10K steps)

## Plan v2 vs Plan v1 차이

| 항목 | Plan v1 (5/05 evening) | Plan v2 (5/05 night) |
|---|---|---|
| 핵심 학습법 | BC only (SmolVLA finetune) | BC + RL hybrid (3-way 비교) |
| Sim 역할 | 데이터 생성기 | RL 환경 + 데이터 생성기 |
| B200 활용 | 1.5h finetune | ~75h training |
| 외부 reference | 없음 | ManiSkill3 + SimpleVLA-RL |
| Paper 메시지 | "BC stacking 가능" | "Real-to-sim hybrid가 BC vs RL 대비 어디 강/약" |
| 데드라인 의식 | 약함 | 2주 lock-in |

---

## Cross-validation rules (Plan 변경 시)

- 사용자 명시 정정 우선 (HARD RULE #18)
- Phase별 EOD checkpoint에서 사용자 confirm
- 변경 시 trace 추가 + 본 문서 history 보존
- HARD RULE #21 lock-in: 추가 BC variant 학습 금지 (sim diversity v4, real_stacking BC only 등)
