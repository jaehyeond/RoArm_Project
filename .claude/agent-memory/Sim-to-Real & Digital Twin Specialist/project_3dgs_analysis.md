---
name: 3DGS Sim-to-Real Critical Analysis (2026-03-24)
description: 4개 논문 실험 조건 분석, single-view RGBD 한계, SigLIP 통과 임계값, 65일 타임라인 평가
type: project
---

## 4개 논문 실험 조건 요약

### SplatSim (arXiv:2409.10161)
- 로봇: Franka Emika Panda (고강성, $30k+)
- 카메라: multi-view RGB (single-view 아님)
- 태스크: tabletop pick-and-place
- 82% 전이율: plain rasterizer(~45%) 대비. Franka+multi-view 조건 한정.
- **우리 RoArm M3 + single-view에 직접 적용 불가** (확신도: HIGH)

### RoboSplat (arXiv:2504.13175, RSS 2025)
- one-shot 데모 + 3DGS augmentation → 수십 개 equivalent 생성
- 요구사항: multi-view reconstruction
- Single-view 가능성: turntable 스캔(50장) 한정으로 가능, 실시간 단일 프레임은 불가
- **물체 turntable 스캔 방식만 현실적** (확신도: MEDIUM)

### GaussTwin (arXiv:2603.05108, 2026-03)
- 목적: 3DGS로 physics simulator 파라미터 역추정 (렌더링이 아님)
- 동적 씬: 오프라인 처리. 실시간 업데이트 = 현재 불가
- 65일 내 구현: 불가 (확신도: HIGH)

### High-Fidelity Simulated Data (arXiv:2510.10637)
- 적용 모델: Diffusion Policy, ACT (fine-tunable backbone)
- SmolVLA SigLIP frozen → 이 논문 결과 직접 이전 불가
- **SmolVLA에 적용 불가** (확신도: HIGH)

## Single-View RGBD → 3DGS 품질

| 방식 | PSNR 추정 | SigLIP cosine dist | SmolVLA 통과 |
|------|-----------|-------------------|-------------|
| 단일 프레임 | ~18-22 | ~0.4-0.5 | LOW |
| turntable 50장 | ~25-28 | ~0.2-0.3 | MEDIUM |
| 3-view RGBD | ~30+ | ~0.1-0.2 | HIGH |

## SigLIP 임계값 (추정)
- cosine dist < 0.2: 전이 가능
- cosine dist 0.2-0.3: 불확실, 실험 필요
- cosine dist > 0.3: 전이 불가

**검증 필수**: sim_siglip_validation.py로 실측 전 결정 금지

## 65일 타임라인 (2026-03-24 → 2026-05-28)
- 3DGS 파이프라인: 2-3주
- 실험: 2-3주
- 논문: 2-3주
- 총: 6-9주 (버퍼 없음, 고위험)

## DR로 해결 안 되는 갭
- Actuator lag 20-50ms: 불가
- Stiction dead-band 1-3°: 불가
- SigLIP visual encoding: frozen이라 DR 무의미
- 물체 변형 (스펀지): FEM 없는 rigid-body sim 한계

## 권고사항
- 3DGS 메인 contribution: 비추천 (선행연구 포화, 65일 위험)
- 3DGS 보조 도구 최적 방법: 씬 일관성 oracle (데이터 수집 중 씬 변형 검출)
- AR-Guided + Demo Quality Oracle 논문에 통합 = +1주 추가 공수

**Why:** 3DGS+robot은 SplatSim, RoboSplat, RoboGSim이 이미 2024-2025에 완료. single-view 제약은 reviewer 공격 포인트.
**How to apply:** 3DGS 논의 시 항상 "single-view 한계"와 "SigLIP frozen 제약"을 먼저 체크.
