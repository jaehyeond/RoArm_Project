---
name: VLA RL Fine-tuning Trend Analysis (2026-03-24)
description: 2025-2026 "sim RL for VLA" 트렌드 비판적 분석. SmolVLA + consumer arm 적용 가능성 평가.
type: project
---

## 트렌드 현황
- YES, 주류 트렌드 맞음. ICLR 2026 VLA 164편 중 RL+VLA 8개+
- BUT: 모두 Franka/UR5/humanoid + A100 클러스터 타겟
- SmolVLA 450M + RoArm-M3 $130 + RTX 4090 Laptop에서는 접근법 근본적으로 달라야 함

**Why:** 대부분 RL 논문이 가정하는 것: (1) physics sim 있음, (2) 7B 모델, (3) A100 클러스터
우리는 셋 다 없음. 트렌드 따라가되 constraint를 기여점으로 전환해야 함.

**How to apply:** RL 논문 related work 작성 시 — 이 차이를 positioning으로 활용

## 5가지 접근법 분류 및 SmolVLA 적용 가능성

| 접근법 | 대표 논문 | SmolVLA 적용 | 이유 |
|--------|-----------|--------------|------|
| World Model Sim | VLA-RFT, WoVR | LOW | token-level reward ≠ flow-matching |
| Physics Sim | GR00T N1.6, Beyond Imitation | LOW | URDF 없음 + SigLIP cosine ~0.6-0.8 |
| Real-World RL (SAC) | HIL-SERL | MEDIUM | SACPolicy ≠ SmolVLA 교체 불가 |
| Reward-Weighted BC | SimpleVLA-RL, RA-BC | HIGH | forward(reduction='none') 이미 지원 |
| Residual RL | 소수 논문 | LOW-MEDIUM | chunk output에 residual 위치 불명확 |

## 가장 중요한 기술적 사실 (HIGH confidence)

### SmolVLA RL 적용 가능한 메커니즘
- `SmolVLAPolicy.forward(reduction='none')` → per-sample loss 반환 (line 392-396)
- 이것에 binary success reward 곱하면 reward-weighted BC = 최소 RL
- 코드 추가: ~50줄
- SigLIP frozen → Action Expert (100M)만 업데이트. VLM 망가질 위험 없음

### Physics Sim 불가 이유 (확인된 사실)
- Isaac Lab rasterizer → SigLIP cosine dist ~0.6-0.8 (전이 불가)
- 3DGS 렌더링만 cosine ~0.1-0.2 (전이 가능) — A2 에이전트 검증
- RoArm-M3 공식 URDF/MJCF 없음

### 450M vs 7B RL 학습 속도
- OpenVLA 7B: LoRA만 써도 batch=8-16, 1K steps = 2-4시간
- SmolVLA 450M: Action Expert만, batch=64, 1K steps = 예상 20-30분
- **확신도: MEDIUM** (실측 미완료)

## 확인된 갭 (반증 검색 후 판정)

| 갭 | 확신도 | 검증 필요 사항 |
|----|--------|---------------|
| Consumer arm ($130)에서 VLA RL 실험 없음 | MEDIUM | SimpleVLA-RL 로봇 확인 |
| SmolVLA 특정 RL 논문 없음 | MEDIUM-HIGH | SmolVLA + RL 검색 |
| Flow-matching denoising variance as RL proxy | HIGH | Diffusion+RL uncertainty 검색 |

## 추천 최소 실험 (65일 남음 기준)
1. Reward-Weighted BC: deploy_smolvla.py → binary label → filtered re-train (1-2주)
2. RL을 Current 방향(AR-Guided + Oracle)의 보완 모듈로 추가
3. RL 구현 없이 Related Work에 positioning만 하는 것도 유효 옵션

## CoRL 포지셔닝
"동일한 RL 아이디어를 consumer hardware + open-source SmolVLA + 실제 물리로 검증"
= 기존 Franka/A100 논문들의 접근성 문제를 직접 해결하는 논문
