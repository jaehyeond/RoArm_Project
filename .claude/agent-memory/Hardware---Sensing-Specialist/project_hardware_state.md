---
name: project_hardware_state
description: RoArm-M3 하드웨어 현황, 센서 구성, 카메라 재장착 위험도 분석
type: project
---

# RoArm-M3 하드웨어 현황

## 보유 장비
- RoArm-M3-Pro 3대 (follower /dev/ttyUSB0, leader /dev/ttyUSB1, spare)
- Azure Kinect DK (pyk4a 1.5.0 + libk4a 1.4.2)
- USB Hub: Kinect + ttyUSB0 + ttyUSB1

## 현재 데이터 상태 (2026-03-24 기준)
- 74 에피소드 수집 완료
- SmolVLA sponge pick 100% 성공 (open-loop 4-chunk, v3)
- 카메라 재장착 이벤트 발생 — 위치 동일 여부 미확인

## 카메라 재장착 위험도 분석

### 물리적 변위 기준
| 장착 방식 | 병진 오차 | 회전 오차 |
|---|---|---|
| 삼각대 퀵릴리스 | 5~15mm | 0.5~2° |
| C-클램프 수동 | 10~30mm | 1~5° |

### 모델 입력(224px) 기준 픽셀 변위 (500mm 작업거리)
| 물리 변위 | 224px 기준 |
|---|---|
| 10mm | 2~4 px |
| 20mm | 4~7 px |
| 30mm | 6~11 px |
| 1° 회전 | 2~4 px |
| 3° 회전 | 7~12 px |

### 임계값
- < 10px (224px 기준): OK
- 10~20px: 주의 (성능 저하 가능)
- > 20px: 재수집 권장

### Azure Kinect 재장착 시 변하지 않는 것
- RGB intrinsics (공장 캘리브레이션, EEPROM 저장)
- Depth-RGB 정렬 행렬
- IMU-카메라 상대 위치

### 변하는 것
- 카메라 → 로봇 베이스 extrinsic (R, t)
- SmolVLA가 보는 raw 픽셀 위치 → OOD 직결

## 손목 카메라 분석 결론 (2026-03-24)
- ZED Mini 손목 장착: 권장하지 않음
  - Joint 4 ±190° 회전 → USB 케이블 1.06회 꼬임 → 파단
  - 스테레오 최소 깊이 10cm > grasp 거리 3~8cm → 근거리 깊이 무효
  - 물리적 크기 124.6mm > 손목 플레이트 50mm → 작업공간 충돌
- 웹캠(C270): 조건부 가능 (Joint 4 ±90° 제한 + RGB only) — 단, 반드시 렌즈 리포커스 필요
- RealSense D405: 근거리 특화(7cm~) 이상적이나 미보유
- 권장: Azure Kinect 3대 중 2대를 멀티 외부 시점으로 구성

## C270 포커스 스펙 (2026-03-24 확정)

### 공장 포커스 설정 (스톡 상태)
| 거리 | 선명도 | Laplacian var |
|---|---|---|
| 10-30cm | 매우 흐림 | ~10-30 |
| 50cm | 흐림 | ~80-150 |
| 70-100cm | **설계 목표 거리** | ~400-800 |
| 150cm+ | 선명 (심도 ∞) | ~400-800 |

- 실측 (2026-03-24): 20cm에서 Laplacian 11.9 → [FAIL] 조작 거리 불가
- 설계 의도: 1m 거리 화상회의용 렌즈 (공장 접착제로 고정)

### 렌즈 물리 스펙
| 항목 | 값 |
|---|---|
| 렌즈 마운트 | M12 (S-mount) 스크류 |
| 대각 FOV | 60° |
| 수평 FOV | ~52° |
| 수직 FOV | ~30° |
| 무게 (클립 포함) | ~83g |
| 무게 (카메라만) | ~75g |
| 클립 마운트만 | ~8g |
| 포커스 잠금 | 공장 시아노아크릴레이트(순간접착제) |
| UVC 오토포커스 | 없음 (소프트웨어 포커스 제어 불가) |

### 리포커스 개조 가능 여부
- **가능**: 렌즈 배럴이 M12 스크류 → 접착제 파괴 후 90-180° 회전으로 포커스 ~15-20cm로 이동
- 소요 시간: 10분
- 리스크: 되돌릴 수 없음 (원거리 포커스 불가), 하우징 분해 필요
- 개조 후 예상 Laplacian: 20cm에서 ~300-600 (조작 사용 가능)
- 대안: ELP/ESCAM M12 근거리 전용 보드카메라 ($15-25) — ALOHA/UMI/DROID에서 사용하는 방식

## 검증 스크립트
- `hw_camera_remount_verify.py` (2026-03-24 생성)
- `hw_wrist_camera_feasibility.py` (2026-03-24 생성) — 손목 카메라 타당성 정량 분석
  - `--mode save_ref`: 기준 이미지 저장
  - `--mode check`: SSIM + ORB shift 측정
  - `--mode aruco`: ArUco 마커 정량 측정
  - `--mode stability`: 카메라 고정 안정성 측정

**Why:** SmolVLA는 raw 픽셀 학습 — 카메라 포즈 변화 = OOD = 성능 저하/실패
**How to apply:** 카메라 재장착 후 반드시 --mode check 실행. 10px 이상이면 조치.
