---
name: Stacking RL Open-source Survey (2026-05-13)
description: Isaac Lab/ManiSkill stacking RL baseline 조사, pre-grasp curriculum 문헌, Phase 0b 설정 권고, 5/19 6일 deadline feasibility 분석
type: project
---

## Key Findings

### Isaac Lab Official Stack = BC Only (no PPO)
- /home/cgxr/Documents/DK/DTR/soarm_stack/third_party/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/
- agents/robomimic/bc_rnn_low_dim.json + bc_rnn_image_200.json ONLY
- rewards=None, curriculum=None in StackEnvCfg
- 업계가 stacking을 "RL 어렵다"고 인식 → BC+Mimic 선택한 것

### ManiSkill StackCube PPO 정확한 수치
- n_steps=50, num_envs=512-1024, batch=512, n_epochs=10, lr=3e-4
- entropy_coef=0.0, gamma=0.8, gae_lambda=0.9
- Sample budget: 50-100M steps → 90%+ success (sim only, Panda 7-DOF)
- Episode 50 step (우리 200 step) — key difference

### 우리 PPO 설정 문제
- gamma=0.99 + 200 step: stage4 현재가치 = 8 × 0.99^200 = 1.07 vs stage2 hover 5.0
- stage4가 현재가치 4.7배 낮음 → hover policy incentivized
- gamma=0.95 권장 ablation (별도 실험, Phase 1 이후)

### Pre-grasp Curriculum 문헌 기반
- Reverse Curriculum (Florensa 2017): 목표 state→점점 멀리 = 우리 Phase 0a→0b→1→2
- CASHER (ICLR 2024): 성공 state pool 저장 재사용 = farming 재발 최후 수단
- OpenAI Dactyl 3단계 초기화: 동일 원리 대규모 실증

### Phase 0a → 0b Catastrophic Forgetting 위험
- 위험도: LOW-MEDIUM
- grasped_frac=0.39 생존 → grasp skill 남아있음
- Mitigation: reset_std=1.5 유지 + entropy_coef=0.003 + sanity gate was_grasped>0.30

### 5/19 6일 Feasibility
- Phase 0b (full pick-place, curriculum): HIGH (~1h wall)
- Phase 0c→1→2 (full workspace): MEDIUM (3-5h total, 각 단계 1h)
- 4-sponge cross stacking Pure RL: INFEASIBLE (200-400M+ steps 필요)
- 현실적 목표: 1-sponge pick-place 75%+ sim → 논문 "RL baseline"

### Phase 0b Recommended Config
entropy_coef: 0.001→0.003 (longer exploration)
sanity gate 추가: was_grasped_rate>0.30 + sponge_target_dist<0.15 at iter 50

**Why:** Phase 0a에서 release 77.8% 학습됨 (P6v14a 확인). Phase 0b는 grasp-first가 필요.
**How to apply:** Phase 0b 시작 전 이 메모리 참조. 4-sponge는 Phase 2 이후 HRL 또는 BC+RL 경로.
