---
name: Isaac Lab 전이 장벽 완전 분석 (2026-03-26)
description: "sim-to-real gap만 해결하면 된다"는 질문에 대한 정직한 답변. 실제 장벽의 70%가 sim-to-real gap과 무관함을 확인.
type: project
---

# Isaac Lab → RoArm-M3 전이 장벽 완전 분석

**Why:** "sim-to-real gap만 해결하면 된다"는 과도한 단순화. 실제 장벽 목록을 정직하게 확인해야 Isaac Lab 사용 여부를 올바르게 판단할 수 있음.

**How to apply:** Isaac Lab 관련 논의 시 "sim-to-real gap" 단독 언급을 피하고 전체 파이프라인 장벽을 함께 설명.

## Isaac Lab 내장 태스크 (실제 확인)

- `reach`: EE → target pose 이동. 우리 구현 있음. 5관절(gripper 없음)
- `lift`: 오브젝트 들기. Franka/OpenArm 지원. ground truth object pose 의존
- `stack`: 큐브 쌓기. UR10 지원. 2 object pose ground truth 의존
- `cabinet`: 서랍/문 열기. Franka 지원. cabinet joint state ground truth 의존
- `pick_place`: GR1T2/Unitree humanoid 전용. RoArm M3 미지원
- `inhand (dexsuite)`: Kuka+Allegro, Shadow Hand. RoArm M3 해당 없음

## 장벽 분류

### 진짜 sim-to-real gap (~30%)
- actuator dynamics 불일치 (ImplicitActuator vs Feetech 서보)
- contact/friction 모델 (Coulomb vs 실제 고무 접촉)
- DR 없음 (mass, friction, delay randomization 미구현)
- URDF 물리 파라미터 미검증 (sysid 없음)

### sim-to-real gap이 아닌 장벽 (~70%)
- action space 변환 코드 없음: relative_rad_delta → absolute_deg (1-2일 코딩)
- 제어 주파수 동기화 없음: sim 30Hz vs 실제 USB 지연 가변 (2-4시간)
- object pose ground truth 의존: lift/stack은 카메라+pose estimation pipeline 필요 (2-4주)
- perception pipeline 없음: RL은 목표 좌표 외부 주입 필요 (VLA와 근본 차이)
- 학습 미완성: 100iter = position_error 0.097m (9.7cm)

## 태스크별 전이 가능성

| 태스크 | 전이 가능성 | 총 작업량 | 성공 확률 |
|--------|------------|----------|----------|
| reach | 가능 (조건부) | 3-5일 | 40-60% |
| lift/pick-place | 어려움 | 6-10주 | 20-40% |
| cabinet | 조건부 | 4-8주 | 15-30% |
| stack | 매우 어려움 | 3-6개월 | 5-15% |

## 핵심 통찰

RL policy = "좌표를 주면 거기 도달" → perception pipeline 별도 필요
VLA = "이미지 보고 결정" → e2e, perception pipeline 불필요

RoArm M3 수준에서 자율 pick-and-place:
- RL 방식: [카메라] → [object detection] → [pose estimation] → [RL] → [변환] → [로봇]
- VLA 방식: [카메라] → [SmolVLA] → [로봇]

**결론: Stage 1에서 Isaac Lab RL 추구하지 말 것. Stage 2+에서 sim data augmentation용으로만 활용.**

## 분석 파일
`/home/cgxr/Documents/Robotics/RoArm_Project/sim_isaaclab_barrier_analysis.py`
