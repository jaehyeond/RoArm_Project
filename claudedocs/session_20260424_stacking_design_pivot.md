# Session 2026-04-24 (Stacking Task Pivot + HW Failure)

## Context
- Step D (SigLIP 50-ep 0.7232) + Step E (sim_v1 build 완료, 72s)
- 교수님 실제 target 공개: **N=2 sponge stacking** (중앙 탑 해체 → 근처 안전거리에 재조립)
- Follower HW 응답 없음 이슈 발견 (세션 중반)

## Task Paradigm Shift
Prior: single sponge pick (v6, Stage 1 = 67%)
**New: Long-horizon N=2 sponge stacking (Isaac Lab + Mimic)**

### N=2 Layout (URDF world, mm)
| 위치 | x | y | 바닥 z | 초기/최종 |
|---|---|---|---|---|
| A (source) | +280 | 0 | -12 | 2-stack (시작) → empty |
| B (target) | +280 | +130 | -12 | empty → 2-stack (끝) |
| Temp | +280 | -110 | -12 | empty → 1 스펀지 → empty |

Safety distance y=130, Temp y=-110 = 유저 "왼쪽/오른쪽 구분" = LEFT (B) / RIGHT (Temp).

### 3-step Pick-Place Sequence
1. A top (z=+200) → Temp (table)
2. A bottom (z=+30) → B (table)
3. Temp (table) → B top (place on bottom at z=+113)

## Sponge 재확인
- **125 × 47 × 20 mm**, ~5g, **수직 세움 (125mm axis up)** — v6 convention 유지
- v6 ep0 frame30/65/80 이미지로 확인: gripper 위에서 top-down 접근, 47mm wide face 양옆 핀치
- 2-stack 총 높이 250mm, 밑면 47×20mm, **aspect ratio 6.25:1 → 넘어지기 쉬움**

## Gripper Stroke 분석 (3중 교차 검증)
1. **URDF**: single rotating jaw, 0~1.571 rad (0~90°), revolute
2. **v6 데이터 (6942 frames)**: action range 0.79~88.59°, mean 18.5°, mode 0-5° (70% closed) / 60-80° (16% open)
3. **Waveshare 공식 spec**: 30mm stroke at 90°

→ **20mm 스펀지 파지에 필요한 25-30mm 개방 가능. Stroke 문제 없음 ✅**

## Follower HW 실측 실패 (중요)
- `{"T":105}` raw serial 응답 = `[180, -180, -90, -180, 180, 180]` = CLAUDE.md 명시 에러 기본값
- `move_init` 명령 ESP32까지 도달했으나 서보 물리 응답 없음
- ESP32 T:106 reset + torque_set(1) 시도 실패
- **유저 물리 확인 필요**: 전원 12V 5A, 서보 LED 점등, 버스 케이블, E-stop
- 회복 후 `gripper_stroke_probe.py` 재실행 가능 (이미 작성됨)

## v6 데이터 Reach 분석 (metadata.json 50 eps 집계)
- `z_at_grip_close`: mean -91.5mm ESP32 = **+30.5mm world** (table +42mm)
- `max_z`: 303mm ESP32 = **+425mm world**
- 즉 v6은 "테이블 위 스펀지 하나 집기"만 커버

### **결정적 발견** — v6 재사용 한계
Stacking 4 동작 중:
- Step 2 bottom pick (z=+30): **v6 분포 내 ✅**
- Step 3 temp pick (z=+30): **v6 분포 내 ✅**
- Step 1 top pick (z=+200): **OOD ❌**
- 모든 Place 동작 (z=+113/+238): **OOD ❌**

**결론: v6 재사용 가능성 ~50%. 나머지는 sim 생성 demo 필수**.

## Depth Overshoot 위험 분석 (핵심)
```
허용 depth 윈도우 (Step 1 top pick):
  최상단 파지 경계: z=+235 (top 끝 - finger length)
  BOT 충돌 경계: z=+148 (BOT top + finger 30mm 하한)
  → 허용 ±45mm
```

제어 정확도 RSS:
- Kinect depth ±1-3mm + VLA z 출력 ±5-10mm + servo ±3-5mm + FK ±2-5mm
- 누적 **±7-13mm** (낙관) / **±20-30mm** (closed-loop drift 감안)

**위험 시나리오 3**: TCP가 z<+148 → fingers가 BOT 스펀지 측면 충돌 → tower 붕괴

**완화 전략**:
1. 데이터: Sim Mimic으로 high-z pick demos 생성 (가장 강력)
2. 제어: deploy time hard limit `z_world > +148+3mm`
3. 인식: Kinect 매 step tower-top z 측정 → adaptive limit
4. 커리큘럼: Phase A (단독 pick) → B (1-stack 해체) → C (2-stack 전체)

## NVIDIA Isaac 도메인 발견 (이미 설치됨)
- **Isaac Lab Stack task**: `IsaacLab/source/isaaclab_tasks/.../manipulation/stack/`
  - Franka/UR10/Galbot config, visuomotor + Cosmos env 완비
- **Isaac Lab Mimic**: `IsaacLab/source/isaaclab_mimic/`
  - 소수 human demo → 1000+ variation 자동 생성
  - Stack/PickPlace env cfg 완비
- **Cosmos** (optional): world foundation model, visual domain randomize

## Phase 재설계 (N=2)
- Phase 0 ✅ v6 single pick baseline (Stage 1 = 67%)
- Phase 1 ✅ sim_v1 visual replay (SigLIP 0.7232)
- Phase 2 (2-3주): Stacking 환경 구축 (isaaclab RoArm 포팅 + 물리 검증)
- Phase 3 (1-2주): Demo 생성 (Real seed 5-10 + Mimic 500+)
- Phase 4 (3-5일): SmolVLA finetune + Sim eval
- Phase 5 (1-2주): Real deployment + iterate
- Phase 6 (future): 3D print 변형 물체 확장

## 의심 리스트 (교수님께 보고 전 검증 필요)
1. "v6 bottom pick 재사용 가능" — tower 있는 이미지는 OOD 가능성
2. "Kinect ±10mm로 충분" — 얇은 edge 감지 실험 필요
3. "Sim rigid body 물리 = real" — sim vs real tower stability 비교 실험 필요
4. "제어 RSS ±13mm" — 낙관적, 보수 ±20mm 가정해 설계해야
5. "SmolVLA 450M이 long-horizon 가능" — HARD RULE #4 문헌 검증 미수행

## 산출물
- `gripper_stroke_probe.py` (follower 회복 후 실행 대기)
- `sim_scripts/sim_to_lerobot.py` (Step E 완료)
- `sim_v1/` LeRobot v3 dataset (87MB, GO baseline)
- `claudedocs/stepDE_siglip50_sim_v1_20260424.md` (Step D/E 상세)
- 이 문서

## 유저 결정 필요 (다음 세션 시작 시)
1. **Follower HW 물리 확인** → 복구 후 `gripper_stroke_probe.py` 재실행
2. **Stacking 3-step vs 4-step** — 후자가 더 안전 but pick-place 많아짐
3. **Curriculum 학습 도입** — Phase A→B→C 단계별 or 바로 C?
4. **Safety limit hard-code** — `z_world > +148mm` 강제 제한?
5. **Layout 좌표 확정** — A(+280,0), B(+280,+130), Temp(+280,-110) OK?
6. **기존 sim_v1 역할** — Stage 1 baseline 비교용 유지 (archive 금지)
