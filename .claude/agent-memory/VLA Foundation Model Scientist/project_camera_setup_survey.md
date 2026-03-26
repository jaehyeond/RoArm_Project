---
name: VLA 논문 카메라 설정 전수조사 (2026-03-26)
description: SmolVLA/pi0/OpenVLA/RT-2/ACT/Octo/DROID 카메라 고정성·뷰 다양성·텔레옵·손가림 처리 전수조사
type: project
---

## 핵심 결론

1. **카메라 고정이 모든 VLA 논문의 표준.** 예외 없음.
2. **카메라 뷰 invariance를 달성한 VLA = 없음.** 알려진 open problem.
3. **텔레옵 방식이 손 가림 여부 결정.** leader-follower/SpaceMouse/VR = 손 가림 없음. 토크 OFF hand-guiding = 손 가림 있음.
4. **DROID 유일하게 연구실간 의도적 뷰 다양성** — 단 단일 세션 내 카메라 이동 아님.

**Why:** Stage 1 데이터 수집 전략 및 "카메라 위치 변경 = 전체 무효" 판단 근거 확보.

**How to apply:** "카메라 위치를 바꿔도 된다" 주장 나오면 이 조사로 즉시 반증 가능. "뷰 다양성 연구 갭"은 Stage 2+ 이후 탐색.

## 논문별 카메라 설정 요약

| 논문 | 카메라 수 | 고정? | 뷰 다양성 | 텔레옵 방식 |
|------|---------|-------|----------|-----------|
| SmolVLA | 3 | 고정 | 없음 | Leader-follower |
| pi0 | 3 | 고정 | 없음 | SpaceMouse/ALOHA |
| OpenVLA/OFT | 1-2 | 고정 | 없음 | SpaceMouse/Leader-follower |
| RT-2 | 1-2 | 이동 로봇 탑재 | 이동 중 변화 | VR/kinesthetic |
| ACT/ALOHA | 4 | 고정 | 없음 | Leader-follower |
| Octo | 1-2 | 고정 | 데이터셋간 | 혼합 |
| DROID | 3 | 연구실별 고정 | 연구실간 다양성 | SpaceMouse/VR |
| GR00T N1 | 2+ | Robot 탑재 | DR (외관만) | Motion capture |
| GraspVLA | 1 | 고정 | Synthetic 각도 | Kinesthetic |

## 카메라 뷰 관련 연구 갭

- "카메라 위치 변화에 robust한 VLA 파인튜닝" = 현재 갭
- SpatialVLA (2501.15830): extrinsics calibration으로 partial 해결
- GraspVLA synthetic diversity: 합성 다양성 → 제한적 robustness
- 완전한 뷰 invariance: 0편 (2025년 8월 기준, MEDIUM confidence)

## 손 가림 처리 방법

1. 구조적 회피 (가장 일반적): leader-follower → operator 카메라 밖
2. 암묵적 허용: hand-guiding 포함해 학습 (커뮤니티 관행)
3. Wrist camera 보완: ALOHA/pi0 → gripper 시점으로 외부 가림 보완
4. 명시적 필터링: 보고 논문 없음

## DROID 카메라 ablation (검증된 수치)

- Wrist image 제거: -8% 성능하락 (Table 3)
- 2번째 exterior camera 제거: -2% (무의미)
- → wrist camera는 중요, 2번째 외부 카메라는 불필요

## 파일 위치

- 상세 분석: `/home/cgxr/Documents/Robotics/RoArm_Project/model_camera_setup_survey.md`
