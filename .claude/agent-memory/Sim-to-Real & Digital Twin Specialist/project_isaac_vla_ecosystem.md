---
name: Isaac Lab + VLA Ecosystem Critical Analysis (2026-03-24)
description: Isaac+VLA 연구 생태계 비판적 분석, GR00T N1 숫자 검증, RoArm M3 적용성 평가
type: project
---

## Isaac + VLA 논문 분포

- 총 논문 수: ~18-25개 (2024-2026, 검색 범위 내)
- NVIDIA 자체/파트너: ~50%. 독립 연구: ~50%
- Isaac Lab이 de facto standard 방향은 맞으나 아직 진행형 (확신도: MEDIUM)

**Why:** NVIDIA 논문 비율이 높아 ecosystem 성숙도는 MuJoCo보다 낮음.
**How to apply:** "Isaac Lab이 표준" 주장 시 "아직 진행형, NVIDIA 의존도 높음" 단서 필수.

## GR00T N1 숫자 비판적 검증

| 주장 | 검증 상태 | 조건 |
|------|----------|------|
| 780K 궤적 11시간 | 확인 불가 (LOW) | H100 클러스터. RTX 4090 laptop은 ~70K/hour (reach task) |
| 40% 향상 | 미검증 (MEDIUM) | Franka/humanoid 기준. Consumer arm 미검증. Baseline 불명확 |
| Consumer arm 작동 | 미검증 (LOW) | GR00T N1 = 휴머노이드 pre-training. RoArm M3 fine-tune 사례 없음 |

## SmolVLA RL Fine-tune in Isaac: BLOCKED

이유:
1. SmolVLA flow-matching = standard RL gradient 불친화적
2. SigLIP frozen → Isaac rasterizer cosine dist ~0.6-0.8 → 전이 불가
3. VRAM: SmolVLA 10GB + Isaac 512envs 4.3GB = 14.3GB (15.6GB 한계에 근접)

VRAM 실현 가능 조합:
- SmolVLA + Isaac 64envs headless: 10.8GB (OK)
- SmolVLA + Isaac 512envs headless: 14.3GB (마진 없음)
- SmolVLA + Isaac 512envs RTX: 16.3GB (OOM)

## 수렴 중인 3가지 접근법 (우리 적용 가능성)

1. RL in Sim → Transfer: BLOCKED (flow-matching 비호환)
2. World Model as Sim: BLOCKED (Cosmos 규모 컴퓨팅 필요)
3. Photorealistic Rendering + DR: 조건부 가능 (3DGS multi-view 한정)

## CoRL 5/28 Isaac 전략

- Isaac 역할: ablation/comparison only
- 최소 실험: 기존 reach RL state data → action prior 비교 1개 표
- 추가 소요: pick-and-place task + converter pipeline = 5-8주 (HIGH RISK)
- 블로커: Isaac → LeRobot v3 변환 파이프라인 미구현 (1-2주 개발 필요)

**Why:** Isaac+VLA 완전 파이프라인은 기간 내 불가. AR+Oracle을 primary로 유지.
**How to apply:** Isaac 관련 제안 시 항상 "ablation only, not primary contribution" 프레이밍.
