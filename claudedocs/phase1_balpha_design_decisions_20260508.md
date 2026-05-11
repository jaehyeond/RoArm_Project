# Phase 1.B-α 설계 결정사항 (실험 실패 시 의심 리스트)

> **확정일**: 2026-05-08
> **Phase**: Phase 1.B-α (1 sponge → L1.spot1 stacking)
> **목적**: 1.B-α 실험 결과가 기대치 미달일 때, 어떤 결정이 원인이었는지 역추적 가능하도록 기록.

---

## 결정 1 — Target 좌표 (확정)

**값**: L1.spot1 sponge center = `(x = +0.280m, y = −0.0435m, z = +0.011383m)` world coord

**근거**:
- HARD RULE #19: edge-stand 47mm tall × 22mm wide × 125mm long
- HARD RULE #20: L1 Y center-to-center = 87mm, inner gap = 65mm, L1 = 2 sponge X-axis 평행
- HARD RULE #21: A layout center = (+0.280, 0)
- spot1 Y = −0.087/2 = −0.0435m
- z = TABLE_Z + SPONGE_HEIGHT_EDGE/2 = −0.012117 + 0.0235 = +0.011383m

**자체 검증**:
- 87mm c2c − 22mm width = 65mm inner gap → HARD RULE #20 명시값과 일치 ✓
- Sponge spawn 영역 R1-R4 모두 target과 75mm+ 떨어짐 → 학습 task 의미 있음 ✓
- z 값은 sponge **center** (TCP place z=+33mm world와는 다름; place z는 TCP 도달점)

**1.B-α에서 의심해야 할 경우**:
- Place success rate 매우 낮음 (e.g. <30%) + sponge가 target 근처 도달은 함 → target z 정의가 sponge center vs TCP 헷갈렸을 가능성. Place success criterion이 sponge center 기준으로 정의됐는지 재확인.
- Real deploy 시 target 좌표가 base→world 변환에서 어긋남 → calibration 재확인.

---

## 결정 2 — Place release 모드 = **gravity 모드** (kinematic-pin 아님)

**값**: gripper open 후 sponge에 자유낙하 (kinematic 강제 고정 없음)

**근거**:
- 1.B-α는 단일 sponge → L1 위에 다른 layer 없으므로 무너질 stack 없음
- 현실 재현이 자연스러움 (sim-to-real gap 줄임)

**1.B-α에서 의심해야 할 경우**:
- Place success rate 낮음 + sponge가 떨어진 후 굴러 옆으로 빠짐 → friction coefficient (현재 static=1.5, dynamic=1.2) 낮을 가능성. table material이 너무 미끄러운지 확인.
- Sponge가 target 위에 도달했는데 gripper open 직후 튀어 오름 → release 시 gripper 갑작스런 open 속도 또는 gripper 위치가 너무 높아서 sponge가 자유낙하 시 회전. release 직전 sponge_z를 측정해서 너무 높으면 reward 페널티 추가.
- **β 단계로 넘어갈 때 재고려 필요**: L1에 이미 1개 placed sponge가 있는 상태에서 spot2에 두 번째 놓을 때, gravity 모드는 첫 sponge가 흔들릴 위험. β 단계에서 kinematic-pin 모드 도입 옵션 검토.

---

## 결정 3 — Observation 차원 = 28 (22 → 28 확장)

**값**: 28-dim observation
- joint_pos (6) + joint_vel (6) + sponge_pos_local (3) + sponge_quat (4) + tcp_to_sponge (3) = 22 (Phase 1.A 그대로)
- + target_pos_local (3) + sponge_to_target (3) = 28

**근거**:
- Place는 본질적으로 "target 위치 도달" task → obs에 target 정보 없으면 implicit 학습 비효율
- β/γ 단계에서 target 위치 변경 (multi-spot) 일반화에 자연스러움

**Warm-start 필요**: Phase 1.A best checkpoint (new_1497, 22-dim 입력)에서 첫 layer weight 22-dim 부분 복사, 새 6-dim 입력 부분은 zero 또는 small random init.

**1.B-α에서 의심해야 할 경우**:
- Warm-start 직후 sanity test가 Phase 1.A 수준 (reach + lift) 아님 → first layer weight 복사가 잘못됐음. expand 스크립트 검증 필요.
- 학습 초반 너무 늦게 수렴 (200+ iter 필요) → 새 6-dim input weight 초기화 방식 부적절. zero init보다 small random (std=0.01) 시도.
- target_pos_local의 좌표계가 헷갈림 (env-local vs world) → Phase 1.A의 sponge_pos_local이 env-origin 빼서 만든 것과 동일 좌표계인지 확인.

---

## 결정 4 — Reward Curriculum: P4 → P5 → P6 점진 방식

**값**:
| Phase | Stage | Reward 구성 |
|---|---|---|
| 1.B-α-P4 | stabilize | reach + lift + grasp + success (Phase 1.A P3 그대로, target 무시) |
| 1.B-α-P5 | navigate | P4 + nav_reward (grasped 상태에서만 −\|sponge − target\|) |
| 1.B-α-P6 | place | P5 + place_bonus (sponge near target ∧ gripper open ∧ stable) + place_success_bonus (single-shot) |

**근거**: Phase 1.A의 new_1100 transient dip (96%) 교훈 — reward 형상 한 번에 크게 바꾸면 100 iter 정도 transient 발생.

**1.B-α에서 의심해야 할 경우**:
- P4 학습 후 success rate가 Phase 1.A 수준 안 나옴 → warm-start 또는 28-dim 추가 input의 영향. obs 첫 22-dim 부분만 사용하는 ablation 시도.
- P5 → P6 전환 시 dip 큼 (예: 80% → 50%) → place_bonus 가중치 너무 강함 (현재 5.0). 1.0으로 줄여서 시작 → 점진 증가.
- P6 학습 끝나도 release timing 학습 안 됨 (gripper close 유지) → grasp_bonus가 너무 강해서 release할 incentive 없음. P6에서 grasp_bonus 줄이거나, sponge near target 시 grasp_bonus 무효화.
- P5에서 sponge가 grasp 안 된 상태에서 target 근처 가기만 함 → nav_reward의 "grasped일 때만" 조건이 코드에서 빠졌을 가능성.

---

## Termination 정책 (Phase 1.A 교훈)

**값**: termination = False (success 시에도 episode 끝까지 진행). success_bonus는 single-shot.

**근거**: Phase 1.A에서 termination=True가 collapse 원인이었음 (Step E 26.87% collapse). Phase 1.B-α에서 같은 mistake 안 하기 위해.

**의심 리스트**:
- Episode 끝까지 진행 시 policy가 release한 sponge를 다시 집어드는 행동 → place_success_flag latched 후의 reward 구조 확인. release 후 reach_reward가 다시 fire되면 그 incentive 발생. flag latched 후 reach_reward 무효화 옵션.

---

## Action / Hyperparameter (Phase 1.A에서 그대로 가져옴)

| 항목 | 값 | 근거 |
|---|---|---|
| action_scale | 0.1 rad/step delta | Phase 1.A 그대로 |
| dof_velocity_scale | 0.1 | Phase 1.A 그대로 |
| episode_length_s | 4.0 (200 step) | Phase 1.A 그대로 |
| desired_kl | 0.005 | Phase 1.A fix 그대로 |
| init_noise_std | 0.8 | Phase 1.A 그대로 |
| num_envs | 4096 | Phase 1.A 그대로 |
| max_iterations per phase | 500 (P4 적게, P5/P6는 500) | 추정. 결과 보고 조정 |

---

## 측정 메트릭 (Report 2 기준)

1. **Place success rate** (primary): sponge가 target 25mm 이내, gripper open, 50 consecutive step 안정
2. **Final position error** (secondary): episode 종료 시 sponge ↔ target 거리 (mm)
3. **Release timing**: gripper open step과 sponge 안정 step의 차이
4. **Trajectory length**: success까지의 step 수

수용 기준: Place success ≥ 70% (4096 trials × 4 seeds, SE ≤ 0.36%), final error mean ≤ 30mm.

---

## 결정사항 변경 이력

- 2026-05-08 작성. 변경 시 이 섹션에 record.
