---
name: Joint Safety & OOD Drift Analysis
description: RoArm M3 관절 안전 제약, Wrist_R 폭주 사례, closed-loop drift 메커니즘, JOINT_LIMITS 존재 이유.
type: project
---

## 알려진 위험 사례

- Wrist_R: -3° → -92° (4σ OOD drift, Run 1, 2026-02-11)
- Elbow: 13° → 36° (DEEP 에피소드 부족으로 위로만 이동)
- Closed-loop n=1: 매 스텝 재추론 → mean regression → 한 방향 drift
- **2026-03-31 배포 실패**: `time.sleep(3)` 부족 → 로봇 미도달 → OOD 시작
  - shoulder 8.5° (목표 44°, 2.2σ OOD), wrist_pitch 9.3° (목표 67°, 2.0σ OOD)
  - elbow 86.9° (목표 41°, 1.4σ OOD) → 전체 에피소드 mean-action 붕괴
  - init→dataset_mean 이동에 필요한 시간: elbow 50° 여행 ≈ 5-7초, sleep(3)은 부족
  - 분석 파일: `trajectory_deploy_20260331_analysis.py`

## OOD Drift 메커니즘

```
초기 위치: dataset_mean (정상)
step 1: 작은 예측 오차 δ
step 2: 오차 누적 → OOD state
step 3: OOD state에서 모델 출력 = mean action (평균으로 회귀)
step 4: mean action이 현재 OOD state에서 더 큰 오차 → 발산
```

## JOINT_LIMITS 하드코딩 이유

절대 제거 금지. 하드웨어 보호 최후 방어선.
3DGS, 증강, 새 학습 방식 도입 시에도 JOINT_LIMITS는 유지.

## 안전 관련 설계 원칙

- n_action_steps=50 (공식 기본값) — closed-loop n=1 drift 방지
- dataset_mean 시작 위치 필수 — [0,0,0,0,0,0] OOD 방지
- 배포 중 joint limit proximity 모니터링 권장
- **start-pos 이동 후 sleep(3)은 부족** — sleep(7) 이상 또는 position-verified wait 필수
  - init→dataset_mean: elbow 91→41=50°, shoulder 1.7→44=42.3° → 5-7초 소요
  - 검증 기준: inference 전 shoulder ±5°, elbow ±5°, wrist_pitch ±5° 확인

**Why:** 2026-02-11 및 2026-03-31 배포 실패에서 직접 학습. OOD 시작 상태는 mean-action 붕괴로 이어짐.
**How to apply:** 새 배포 스크립트 검토 시 n_action_steps, 시작 위치, JOINT_LIMITS, 이동 완료 대기 시간 4가지 반드시 확인.

## Visual Grounding Failure (2026-03-31, v5 120K deployment)

- **증상**: base 관절이 항상 ~10° (dataset mean)에 고착 — sponge 위치 무시
- **Shoulder/Gripper는 정상**: pick sequence (descend→open→close→lift) 올바르게 실행
- **eval이 PASS한 이유**: 5가지 구조적 blindspot (상세: `trajectory_visual_grounding_eval_critique.py`)

### eval blindspot 요약 (5개)

1. **CRITICAL — 이미지 조건부 메트릭 없음**: 이미지를 셔플해도 L2가 안 변하면 image grounding 없음
2. **CRITICAL — 6D L2 averaging**: base 오차 15°는 25° norm에 ~5° 기여 → 마스킹됨
3. **HIGH — base_std가 proprioceptive echo**: offline eval에서 state input이 varying → base pred varying. deployment에서 state 고정 → base 고정
4. **HIGH — 존 샘플 크기 불충분**: n=7 LEFT, 80% power에 n≥98 필요. 현재 power ≈ 16%
5. **MODERATE — zone L2 = 6D error, base-only 아님**

### 추가해야 할 eval 메트릭 (우선순위 순)

```python
# (1) Pearson correlation — base joint grounding의 핵심 테스트
r = np.corrcoef(all_actions_arr[:,0], all_gt_arr[:,0])[0,1]
# PASS: r ≥ 0.50 / FAIL: r < 0.30

# (2) Image-permutation ablation — 이미지 dependency 직접 검증
# shuffled images로 재추론 → base prediction이 얼마나 바뀌는지
# PASS: image_sensitivity ≥ 5° / FAIL: < 2°

# (3) Zone directional accuracy — LEFT/RIGHT 방향 맞히는지
dir_acc = mean(sign(pred_base) == sign(gt_base))
# n=15이면 충분 (sign test)

# (4) Constant-predictor baseline comparison — 항상 mean 예측과 비교
# model_base_L2_LEFT < 0.70 × baseline_base_L2_LEFT 이어야 PASS
```

### 데이터 문제 (근본 원인)

80.1% CENTER 에피소드 → 모델이 "base≈10°는 항상 안전"을 학습
LEFT 2ep, RIGHT 25ep → LEFT zone에서 base 학습 사실상 불가능
FIX: 존별 균등 수집 (최소 30ep/zone) 또는 loss weighting
