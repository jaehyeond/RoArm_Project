---
name: VLA Sim-to-Real 문헌 전수조사 결과 (2026-03-26)
description: VLA에서 sim 이미지 사용 가능성 — 문헌 조사 결과 및 SmolVLA 적용 판단
type: project
---

## 핵심 결론

이전 결론 ("Isaac Sim 이미지 불가, 궤적만 가능") 유효 — 문헌으로 검증됨.

**Why:** SigLIP frozen VLA + sim 이미지 혼합 성공 사례 = 0편 (2025년 8월 기준). GR00T N1만 예외이나 humanoid + NVIDIA 전용 DR 파이프라인 + 클러스터 조건.

**How to apply:** sim 이미지 기반 증강 제안이 나오면 이 조사로 반증. 단 3DGS / 배경 in-painting은 별개로 탐색 가치 있음.

## sim 이미지 사용 현황

| 시스템 | sim 이미지 | 조건 | 확신도 |
|--------|-----------|------|--------|
| GR00T N1.6 | YES (~85%) | Isaac Sim + DR + humanoid | MEDIUM |
| RoboCasa | YES (~95%) | ACT/Diffusion — VLM 없음 | HIGH |
| pi0, OpenVLA, Octo, SmolVLA | NO | real only | HIGH |

## cosine distance 기준

- Isaac Sim rasterizer: 0.6-0.8 → FAIL
- 3DGS rendering: 0.1-0.2 → 탐색 가능
- Real (동일 장면): 0.05-0.15 → PASS

## 탐색 가치 있는 대안

1. 배경 in-painting aug (RoboAgent 방식): real 이미지 배경만 교체, 71%→87% 선례
2. 3DGS workspace 재구성: cosine 0.1-0.2, 공수 2-3일+GPU

## 연구 갭 가능성

"SigLIP-frozen VLA + sim 혼합 체계적 연구" = 0편 발견 (확신도 MEDIUM — 2025년 9월 이후 추가 검색 필요)
