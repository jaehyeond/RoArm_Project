# Session 2026-05-13 — Hierarchical Chain Skills Pivot

## Context

사용자 결정 (HARD RULE #18 사용자 명시): PATH D BC FAIL (5/17 CLEAN 9.38%) 후 **Option C 진행 (Pure RL infrastructure)**. 그러나 검증 결과 **Phase 0 V1 이미 PASS (5/07 night)** — P6v1~v14c + PATH D 9회 시도가 그 결과물. 진짜 다음 단계 = **abstraction level 변경**.

## 선택된 path: (B) Hierarchical skill primitive + scripted prior

사용자 분석 Tier ✅ ("Hierarchical: high-level planner + 4 skill primitives") 일치 +
P6v14a evidence (stage4 77.8% narrow scaffold만 작동 = primitive 단위 PPO 학습 가능).

### Chain 구조
```
Skill 0: HOME → hover above sponge   (scripted, IK)
Skill 1: descend + gripper close      (scripted, IK + gripper cmd)
Skill 2: lift + transport hover       (scripted, IK)
Skill 3: descend + release            (LEARNED, P6v14a/model_499.pt)
```

### Geometry (HARD RULE #19/#20 측정값 활용)
- TABLE_Z = -12.117mm, SPONGE_CENTER_Z = +11.4mm
- TCP_GRASP_Z = +33mm, transport_offset_z = +30mm (P6v14a entry +61.6mm 매치)
- GRIPPER_CLOSE_DEG = 45.84° (P6v14a init q=0.8 rad 매치, NOT 5° mech-close)
- Action_scale = 0.1 rad/step

## 작성 파일

- [roarm_rl/sanity_chain_ik.py](roarm_rl/sanity_chain_ik.py) — 24 waypoint IK sanity (23/24 PASS, R2_inner_front FAIL elbow boundary)
- [roarm_rl/chain_skills.py](roarm_rl/chain_skills.py) — TrajectoryPlanner + ChainRunner
- [sim_scripts/roarm_kinematics.py](sim_scripts/roarm_kinematics.py) — pandas lazy import (isaacsim_5_1 env 호환)

## B200 Chain Run Results (7 iterations)

| # | Fix | Skill 0 | Skill 1 grasp | Skill 3 outcome | Verdict |
|---|---|---|---|---|---|
| 1 | initial | env.step ValueError 5-tuple | — | — | TypeError |
| 2 | pandas lazy | 동일 | — | — | re-test ok |
| 3 | 4-tuple unpack | actor IndexError dict obs | — | — | fix |
| 4 | obs_t TensorDict | 60 step max_err 6.95°, tcp_err 35.2mm | False (d_tcp=120mm) | TIMEOUT | open-loop overshoot |
| 5 | closed-loop tol 1° | 200 step max_err 7°, **steady state** | False (d_tcp=53.8mm) | SUCCESS step 1 fp | PD limit cycle 확인 |
| 6 | + diagnostic log | elbow s=40 +128.8° (target 121° overshoot +7.7°) | sponge_after1=(+270,-84,+11) | SUCCESS fp | **Limit cycle root cause** |
| 7 | **force-set robot_dof_targets** | **21 step max_err 1.56° ✓** | False (d_tcp=31.1mm, descent fail) | SUCCESS step 1 fp | Skill 0 fix, Skill 1 new issue |

### 결정적 발견
1. **PD limit cycle**: action_scale=0.1 + saturated ±1 closed-loop → `robot_dof_targets` 누적 → joint limit saturate → robot이 limit으로 끌려감. **PPO 학습 시는 stochastic action이라 안 생김.**
2. **Force-set fix**: scripted skill에서 `base_env.robot_dof_targets[:] = target_t` + null action → PD limit cycle 우회. **Skill 0 200→21 step 단축**.
3. **Skill 1 descent fail (force-set 후도)**: Robot이 sponge 위 약간 옆에 도달 (tcp_err 10mm) → descent 중 sponge 측면 충돌 → sponge 옆/위로 밀림 (+12mm 들림, -44mm Y) → grasp 영영 불가.
4. **Skill 3 SUCCESS step 1 = false positive**: sponge 안 잡혔는데 sponge 위치가 우연히 d_xy<30mm, d_z<25mm thresh 만족.

## 진짜 root cause

**Sim physics의 sponge 충돌 모델**: edge-stand sponge (22mm wide × 47mm tall) 옆에 robot이 약간 비껴 도달하면 측면 충돌 → 옆으로 밀림. IK tcp_err 10mm = sponge 22mm wide 안에 거의 들어가지만 마진 작음.

이건 P6v14a evidence (narrow scaffold만 작동)와 일치 — RoArm M3 + edge-stand sponge에서 grasp 자체가 sim에서 어려운 task.

## Hierarchical 가설 부분 검증 status

| Component | Status |
|---|---|
| Skill 0 (reach) scripted IK + force-set | ✅ 작동 (21 step) |
| Skill 1 (descend + grasp) | ❌ sponge 충돌로 fail |
| Skill 2 (transport) | (Skill 1 fail로 검증 안 됨) |
| Skill 3 (P6v14a release) | OOD 상태로 step 1 false positive |

## 다음 step 옵션 (사용자 결정 대기)

| 옵션 | 작업 | 가치 |
|---|---|---|
| **(α) Skill 0/1 우회 + Skill 3 단독 검증** | env init = Skill 2 끝 state force-set (sponge in hand, `_grasped=True` latch, TCP at place hover). Skill 3 inference. **진짜 hierarchical 핵심 가설 1:1 검증** | 가장 cheap |
| (β) Sim physics 디버그 | Sponge friction/restitution/mass + IK accuracy mm 단위 | 시간 많이 듦 |
| (γ) Skill 0/1을 PPO로 학습 | 4 specialist library (DT-HRL/MAPLE 학술 SOTA) | 1-2주, paper-quality |

**권장: (α)** — Skill 3가 우리 chain end state에서 작동 증명되면 (β)/(γ) 진행 의미 명확. Skill 3도 fail이면 hierarchical 가설 자체 falsified.

## HARD RULE 준수

- #4 외부 citation 본 세션 없음 (사용자 분석에서 가져온 DT-HRL/MAPLE 등은 재검증 안 함)
- #11 /half-clone 거부 (context 152%에도 본 prompt + MEMORY로 세션 넘김)
- #14 fail-fast guard + no `2>&1` + no pipe-to-source 모든 ssh 적용
- #17 state-only RL physics-only로 narrow inline (Skill 3 = state obs 28-dim only)
- #18 사용자 명시 정정 우선 ((B) Hierarchical 선택, 5/19 deadline 무시)
- #19 edge-stand 47mm 유지
- #26 5/07 night Phase 0 V1 PASS 사실 확인 → C 작업 이미 한참 진행됨 인식

## 파일 inventory

### 변경 (Lenovo local)
- [roarm_rl/chain_skills.py](roarm_rl/chain_skills.py) — 489 LOC, force-set + closed-loop + diag log
- [roarm_rl/sanity_chain_ik.py](roarm_rl/sanity_chain_ik.py) — 145 LOC
- [sim_scripts/roarm_kinematics.py](sim_scripts/roarm_kinematics.py) — pandas lazy import

### B200 sync 완료 (md5 verified)
- `code/roarm_rl/chain_skills.py`
- `code/sim_scripts/roarm_kinematics.py`

### 로그
- `b200_chain[2-8].out/err` (Lenovo local, run #1-7 stdout/stderr)
- B200 `logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt` (변경 없음, Skill 3 source)

## 다음 세션 첫 작업

사용자 (α)/(β)/(γ) 결정 → 결정에 따라:
- (α) 선택 시: chain_skills.py에 `--skip_scripted` flag 추가 → env init을 Skill 2 끝 state로 force-set + Skill 3만 inference. 단일 episode metric (SUCCESS_step, sponge_z trajectory, gripper_q trajectory).
- (β) 선택 시: roarm_stack_env.py sponge `RigidObjectCfg` mass/friction/restitution 확인 + 조정.
- (γ) 선택 시: 새 env (RoArmReachEnv, RoArmGraspEnv 등) 4개 + 4 PPO training launch (~1주 B200).
