# Session 2026-05-14 — Path A REJECTED + Path B RPL Launch + 5-Path Strategy

## TL;DR
- **Round 1 BC+DAPG pivot에서 5-path 병렬 전략으로 재설계** (Path A/B/D/E/F sequential).
- **Path A (P6v15 = P6v14c + reset_actor_bias_idx 5) REJECTED**: P6v15 metrics ≈ P6v14c bit-identical (iter 0/1/10/499 모두 매칭). Gripper close-bias 가설 거짓.
- **Path B (P6v16 = RPL frozen BC + residual) 진행 중**: iter ~305 시점 stage4_success 3× 개선 (0.017 vs P6v15 0.006). 결정적 판단은 iter 0/1/10 비교 필요. 종료 polling 중.

## Round 1+2 Research 종합 (6 agents)
- **NVIDIA Isaac Lab source check (Agent 5 ground truth)**: 155 RL configs vs 13 BC configs. BC+PPO hybrid = **0 official examples**. Round 1 "BC가 트렌드"는 confirmation bias.
- **2026 trend (Agent 4 verified)**: PPO from scratch 9/14 community Isaac Lab manipulation repos. DAPG (2017) supersede됨. SOTA = off-policy + frozen BC (IBRL 2024, ResFiT 2025).
- **robomimic 0.5에 DiffusionPolicyUNet 포함** (Agent 6). B200에 robomimic 0.3.0 install — DiffusionPolicy 지원 0.5 필요 (재install or LeRobot diffusion).
- **NVIDIA SkillGen (cuRobo + Mimic)**: Pick-Place-Milk 95% benchmark.

## Path A (P6v15) FAIL 분석
launch_p6v15.sh = launch_p6v14c.sh + `--reset_actor_bias_idx 5` 1줄 추가.
| iter | stage4_v15 | stage4_v14c | gripper_open v15→14c | grasped |
|------|-----------|-------------|----------------------|---------|
| 0 | 0.3649 | 0.3653 | 0.7840 / 0.7835 | 0.0797 / 0.0807 |
| 1 | 0.3425 | 0.3446 | 0.2068 / 0.2100 | 0.6729 / 0.6696 |
| 10 | 0.0033 | 0.0037 | 0.1124 / 0.1119 | 0.8491 / 0.8496 |
| 499 | 0.0093 | 0.0105 | 0.0681 / 0.0692 | 0.9128 / 0.9132 |

**진단 재구성**:
- "PPO 1-iter forget" 진단은 **부분적 잘못**. iter 0→1 stage4 6% 감소만 (release path 미세 남음).
- 진짜 collapse는 iter 5 (0.145, 60% drop) → iter 10 (0.003, 98% drop). **5-10 iter cascade**.
- Reward shape, bias reset, kl=0.005, adaptive LR 모두 active했음에도 cascade 차단 못함.
- **Algorithmic limit**: PPO advantage가 rare-event jackpot < dense grasp basin. Structural fix 필요.

## Path B (P6v16) RPL 구현
**Files added/modified**:
- [roarm_rl/policies/residual_actor.py](roarm_rl/policies/residual_actor.py) (151 LOC, new)
  - `ResidualMLPWrapper(bc_mlp, residual_mlp, alpha=0.3)`: forward = bc(no_grad) + α × res
  - `build_residual_mlp(28, 6, [64,32])`: zero-init final layer (start at BC)
  - `install_residual_actor(actor_critic, bc_state_dict, alpha, hidden)`: in-place patch
- [roarm_rl/policies/__init__.py](roarm_rl/policies/__init__.py) (empty marker)
- [roarm_rl/train_ppo.py](roarm_rl/train_ppo.py) — 4 new CLI flags:
  - `--residual_mode` (mutually exclusive with `--resume`)
  - `--residual_bc_ckpt` (path to BC actor .pt)
  - `--residual_alpha` (default 0.3)
  - `--residual_hidden` (default "64,32")
  - Post-runner-init patch: load BC → install wrapper → rebuild optimizer over trainable only

**md5 (new sync state, 5/14)**:
- roarm_stack_env.py = `ff31c5a32e85cc61bec39302a9b739c0` (unchanged)
- train_ppo.py = `a5fa7482d98fd66b13e4b272454237df` (residual_mode CLI)
- residual_actor.py = `f47d16d7dc43339a5cc3d5a549138ea6` (new)

**Launch**: [launch_p6v16_pathB.sh](launch_p6v16_pathB.sh) — `--residual_mode --residual_bc_ckpt P6v14a/model_499.pt --residual_alpha 0.3 --residual_hidden 64,32`. tmux session `p6v16` on B200.

**중간 결과 (iter ~305 시점)**:
| metric | P6v16 iter 305 | P6v15 iter 300 |
|--------|---------------|----------------|
| stage4_success | 0.0171 | 0.0058 |
| stage2_grasp | 0.8125 | 0.89 |
| gripper_open | 0.0725 | 0.078 |
| jackpot_fire | 0.0001 | 0.0 |

미미한 개선. 결정적 판단은 iter 0/1/10 비교 필요 (학습 종료 후 metrics extract).

## Path D Design (완료)
[claudedocs/path_d_design_20260514.md](claudedocs/path_d_design_20260514.md) — Task decomposition (P6v14a pick + release-only BC + state machine handoff). 3-phase pipeline:
- Phase D.1: P6v14a rollout sweep → successful episode 70개 → release-only trajectory (gen_release_demos_from_rollout.py ~120 LOC)
- Phase D.2: BC MLP 28→64→6 train (~3min)
- Phase D.3: State machine eval (P6v14a + release BC) 500 episodes

## Path E (Diffusion) 준비
- robomimic 0.3.0 installed on B200 (BC/BC-RNN/BC-Transformer만 지원, **DiffusionPolicyUNet 0.5에서 추가** — 재install 필요 OR LeRobot diffusion_policy 사용)
- generate_pick_place_demos.py 113 LOC skeleton (constants + waypoint_sequence 정의됨, main placeholder)
- 200 LOC complete 필요 (Isaac Sim launch + 7 waypoint IK + 200 step linear interp + 28-dim obs recording + torch save)

## Path F (SkillGen) 미진행
NVIDIA SkillGen + cuRobo. Isaac Lab 2.3 ship. ~1 day impl.

## 사용자 명시 정정 (HARD RULE #18)
- 5/14 evening: "B/D/E/F 다 해보자, B200 학습 전용. step-by-step. 왜 학습 안 됐는지 확인"
- 5/14 evening: "Path A (PPO + safeguards) 추가 = 5 path"
- 5/14 evening: "Sequential, cheap → expensive"
- 5/14 evening: "이전 launch_*.sh 패턴 따라가" (HARD RULE 강조)

## HARD RULE 준수
- #4 (10+ search × 2 source): Round 1 + Round 2 = 6 agents 검증
- #8 archive: 6번째 entry로 prepend → 5번째를 archive로 이동 필요 (next session)
- #11 /half-clone 절대 안 함 (context 141% 시점에도 continuation prompt 방식)
- #13 B200 path = /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200 (실제 path 검증 5/14)
- #14 fail-fast guard 모든 ssh + bash pipe-to-source 금지
- #15 cu128 sm_100 alive (torch 2.12.0.dev20260407+cu128, B200 191.5GB)
- #17 state-only 28-dim 유지
- #18 사용자 명시 4회 follow

## 다음 세션 즉시 할 일
1. **p6v16 학습 종료 polling** 결과 받기 (현재 background `bf2jdkc5q`)
2. **claudedocs/p6v16_data/ 생성 + extract_metrics.py 작성** (p6v15 패턴 재활용)
3. **P6v16 vs P6v15 vs P6v14c iter 0/1/5/10/50/100/200/499 비교 표**
4. **RPL 가설 판정**:
   - PASS (iter 10 stage4 ≥ 0.20): RPL 정답 → Path B fine-tune (alpha 0.1/0.2/0.5 sweep, or longer train 1000 iter)
   - FAIL (iter 10 stage4 < 0.05): Path D 진입 → Phase D.1 demo gen script 작성
5. 결과에 따라 Path D/E/F 진행

## Inventory (5/14 변화)
- B200 ckpts:
  - `logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt` (Path B BC base, deploy OK)
  - `logs/roarm_rl/p6v14c_phase0a_prime_hover_resumeP6v14a/model_*.pt` (11 ckpts, deploy 금지)
  - `logs/roarm_rl/p6v15_pathA_bias_reset_resumeP6v14a/model_*.pt` (11 ckpts, deploy 금지 — Path A fail)
  - `logs/roarm_rl/p6v16_pathB_RPL_alpha03/` (학습 진행 중)
- 로컬:
  - launch_p6v15.sh, launch_p6v16_pathB.sh (new)
  - roarm_rl/policies/__init__.py, residual_actor.py (new)
  - roarm_rl/train_ppo.py (modified, md5 a5fa748...)
  - claudedocs/p6v15_data/{train_p6v15.out, p6v15_metrics.csv, extract_metrics.py}
  - claudedocs/path_d_design_20260514.md
