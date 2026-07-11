---
name: feedback_sensor_evaluation
description: 센서 추가 평가 방침 및 하드웨어 분석 접근법
type: feedback
---

# 센서 평가 방침

## 규칙
카메라 위치 변경 시 항상 정량적 검증 수행 — 시각적 판단 금지.

**Why:** SmolVLA는 raw 픽셀 직접 학습. 사람 눈에 "거의 같아 보여도" 모델에는 OOD일 수 있음.

**How to apply:** hw_camera_remount_verify.py --mode check 실행 후 ORB shift 수치 확인. 10px(224px 기준) 임계값 적용.

## 하드웨어 분석 원칙
- 저비용 플랫폼($130 팔) 기준으로 실용적 임계값 사용 — 산업용 정밀도 요구하지 않음
- 검증 방법은 항상 "실행 가능한 코드"로 제공
- 수치 추정은 항상 가정(작업거리, focal length 등)을 명시
