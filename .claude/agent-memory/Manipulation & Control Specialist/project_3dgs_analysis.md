---
name: 3DGS + Manipulation Technical Analysis (2026-03-24)
description: 8개 3DGS 논문의 RoArm M3 적용성 분석. SigLIP frozen 제약, joint-space vs 3D geometry, 구현 우선순위.
type: project
---

## 핵심 결론

3DGS의 실질적 가치 = 학습 데이터 증강 도구 (학습 시 오프라인)
NOT = 실시간 3D perception 도구

**Why:** SigLIP frozen 제약으로 GeoPredict 방식의 geometry-aware VLM 학습 불가.
3DGS novel view rendering → 학습 데이터 다양화는 가능 (SplatSim, RoboSplat 방식).

**How to apply:** 새 태스크 제안 시, CoRL deadline 이후 석사논문 범위에서 3DGS ablation 포함 권고.
CoRL 5/28 이전에 3DGS 구현 추가 = 리스크.

## 논문별 적용성 (우리 RoArm M3 setup)

| 논문 | 적용성 | 이유 |
|------|--------|------|
| RoboSplat (2504.13175) | 중간 | multi-view RGB 필요하지만 Azure Kinect 3대로 가능 |
| SplatSim (2409.10161) | 중간 | 정적 씬 sim-to-real, single-view 미검증 |
| GeoPredict (2512.16811) | 낮음 | SigLIP frozen → auxiliary geometry head 추가 불가 |
| ArtGS (2507.02600) | 없음 | articulated object 전용, 우리 물체는 rigid |
| DexFruit (2508.07118) | 없음 | 다지 핸드 전용 |

## 3DGS vs 2D augmentation 경계

2D augmentation으로 불가한 것 (3DGS 필수):
- 카메라 viewpoint 변화
- 물체 3D 위치/포즈 변화
- 로봇 embodiment 변화

2D augmentation으로 충분한 것:
- 조명 변화 (color jitter)
- 텍스처 변화
- 카메라 노이즈

## SigLIP Frozen 제약의 의미

GeoPredict 방식 = SigLIP에 geometry-aware feature 학습시키는 것이 핵심.
SmolVLA에서 SigLIP frozen → geometry prediction auxiliary task 추가해도 VLM weights 업데이트 불가.
→ GeoPredict 완전 구현 불가.

적용 가능한 경량 버전:
```
Azure Kinect depth → backproject → object_3d_pos
→ Action Expert conditioning 추가 (SigLIP 건드리지 않음)
```

## Joint Space vs 3D Geometry

Wrist_R 폭주 (-3 → -92) = action distribution 문제, NOT visual feature 문제.
3DGS가 이를 해결하지 않음. JOINT_LIMITS 하드코딩이 1차 방어선.

3DGS가 도움되는 것: chunk boundary에서 visual feature 안정화 (viewpoint 다양화로 robust).
3DGS가 도움 안 되는 것: joint-space OOD drift.

## 구현 우선순위 (우리 setup)

1. 즉시 가능: Azure Kinect depth → object 3D position → observation에 추가
2. 2-4주: 정적 workspace 3DGS 학습 → novel view → augmented training data
3. 보류: GeoPredict 완전 구현, 동적 씬 3DGS, 실시간 재구성
