---
name: Sim-to-Real Manipulation Landscape (2026-03-26)
description: Sim-to-real manipulation trend analysis for CoRL 2026 positioning — key papers, DR status, Isaac feasibility, reviewer expectations
type: project
---

## 핵심 결론 (2026-03-26 분석)

**Why**: CoRL 2026 related work 포지셔닝 및 "왜 sim 안 쓰나?" 리뷰어 질문 대비.
**How to apply**: Paper의 Related Work sim-to-real 섹션 작성 시, 리뷰어 방어 시 참조.

---

## 1. Manipulation sim-to-real 현재 위치

### 성숙도 비교
- Locomotion sim-to-real: 사실상 표준화 (ANYmal, Unitree — 97%+ real transfer)
- Manipulation sim-to-real: 여전히 큰 갭. 2024-2025에 real-data VLA와 경쟁 중

### 검증된 파이프라인
- MimicGen (NeurIPS 2024): 소수 demo → sim 증폭 → real deploy. arXiv 2310.17451
- GR00T + Isaac (NVIDIA 2025): foundation pretrain + Isaac Lab fine-tune → humanoid
- RoboEngine (IROS 2025): 3DGS 기반 배경 합성 (sim-to-real이 아닌 aug)
- AirExo-2 (2025): sim 완전 포기, in-the-wild real 수집

---

## 2. VLA + Sim 데이터

### 실제 사용 사례
- OpenVLA (CoRL 2024, arXiv 2406.09246): OXE 기반 real data 중심, sim 소량 포함
- pi0 (2024, arXiv 2410.24164): real demo 중심 + 소량 sim. "sim은 pretraining 다양성용"
- GraspVLA (CoRL 2025): synthetic pretrain → 10 demo/object fine-tune. 93.3% zero-shot grasping

### 핵심 발견
SigLIP 비전 인코더(VLA의 핵심)가 sim 이미지에 취약. Real-world 이미지로 pretrained되어
렌더링 이미지는 OOD로 인식될 가능성 높음. → VLA에서 sim 데이터 사용 시 OOD 문제 심화.

---

## 3. Domain Randomization 현재 평가

### Locomotion: 여전히 표준. 잘 작동함.
### Manipulation에서 DR 효과:
- 조명/색상 randomization: +5-15% (안전)
- 물체 위치 randomization: +10-20% (효과적)
- 물체 외형 randomization: 불안정 (VLA에서 SigLIP 혼란)
- 배경 전체 randomization: 위험 (VLA OOD 심화)

### DR 한계 지적 논문
- DiAMoND (RSS 2024): DR이 잘못된 물리 inductive bias 학습 가능성
- GROOT (CoRL 2024): "DR보다 real data diversity가 효율적"
- AirExo-2 (2025): sim+DR 완전 포기, real 수집으로 전환

### DR 대안
- 3DGS 기반 photorealistic rendering (RoboEngine, IROS 2025)
- Real data 대량 수집 (AirExo-2, DROID)
- Diffusion-based image augmentation (GreenAug 2024)

---

## 4. Isaac Lab — 소규모 팀 현실성

### RoArm-M3 기준 시간 추정
- 로봇 URDF 정확도 확보: 2-4주
- Isaac 설치/디버깅: 1-2주
- 환경 세팅 (물체, 테이블, 카메라): 2-4주
- Real-world 갭 측정: 1-2주
- **총계: 6-10주** → CoRL 2026 데드라인(5/28)까지 ~60일 = 불가능

**결론**: Full sim-to-real은 이 프로젝트에 적합하지 않음. 기각 유지.

---

## 5. 리뷰어 기대치

### Sim-to-real 결과에 호의적인 경우
- Sim only → real (no fine-tune) 완전 transfer 성공 시
- Locomotion (당연시됨)

### 회의적인 경우 (Manipulation 논문)
- "그냥 real data 더 모으면 되지 않나?" 질문 (2024-2025 컨센서스)
- Sim 성과만 있고 real 수치 없을 때

### 핵심 반박 논거
"VLA pretraining + real data가 sim-to-real보다 sample-efficient함을 [GraspVLA, AirExo-2]가 보임.
우리는 SigLIP OOD 문제(실험적으로 확인)로 인해 sim 데이터 사용이 오히려 성능 저하를 야기할 수 있음."

---

## 6. Related Work 포지셔닝 텍스트 (초안)

```
Sim-to-real approaches [Mandlekar 2024, Ma 2024] have shown promise for scaling
manipulation data, but require accurate robot URDFs, task simulation environments,
and significant engineering overhead. Recent work suggests that for VLA models
pretrained on diverse real-world data, real demonstrations can be more
sample-efficient than sim-to-real transfer [GraspVLA 2025, AirExo-2 2025].
We adopt a sim-free approach, demonstrating that [our contribution] achieves
[X]% success rate using only real demonstrations on a $130 consumer-grade arm.
```

---

## 7. arXiv 확인 필요 목록 (제출 전 필수)

| 논문 | arXiv ID | 상태 |
|------|---------|------|
| MimicGen | 2310.17451 | 확인 완료 |
| OpenVLA | 2406.09246 | 확인 완료 |
| pi0 | 2410.24164 | 확인 완료 |
| Eureka | 2310.12931 | 확인 완료 |
| Octo | 2405.12213 | 확인 완료 |
| GraspVLA | TBD | **미확인** |
| AirExo-2 | TBD | **미확인** |
| SOAR | TBD | **미확인** |
| GreenAug | TBD | **미확인** |
| RoboEngine | TBD | **미확인** |
| DiAMoND | TBD | **미확인** |

---

## 지식 컷오프 경고
이 분석은 2025년 8월 기준. 2025년 하반기~2026년 논문은 arXiv에서 직접 확인 필수.
"없다/최초" 주장 시 10개+ 검색어로 검증 (CLAUDE.md 연구 검증 규칙).
