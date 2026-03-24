---
name: Joint Safety & OOD Drift Analysis
description: RoArm M3 관절 안전 제약, Wrist_R 폭주 사례, closed-loop drift 메커니즘, JOINT_LIMITS 존재 이유.
type: project
---

## 알려진 위험 사례

- Wrist_R: -3° → -92° (4σ OOD drift, Run 1, 2026-02-11)
- Elbow: 13° → 36° (DEEP 에피소드 부족으로 위로만 이동)
- Closed-loop n=1: 매 스텝 재추론 → mean regression → 한 방향 drift

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

**Why:** 2026-02-11 배포 실패에서 직접 학습. Wrist_R 폭주는 단순 OOD에서 시작.
**How to apply:** 새 배포 스크립트 검토 시 n_action_steps, 시작 위치, JOINT_LIMITS 3가지 반드시 확인.
