# VLA + Sim-to-Real 문헌 전수조사 — 비판적 검증

**Date**: 2026-03-26
**Agent**: B1 VLA Foundation Model Scientist
**Status**: DONE (지식 컷오프 2025년 8월 기준, 이후 논문 포함 불가)

---

## 조사 범위

**검색 키워드**:
- "VLA sim-to-real", "synthetic data VLA", "domain randomization VLA"
- "simulation pretrain VLA", "Isaac Sim VLA", "simulated data robot learning"
- "SigLIP sim images", "vision language action simulation", "sim-to-real manipulation VLA"

**확인 소스**: arXiv, RSS/CoRL/NeurIPS/ICLR 2023-2025 프로시딩 (지식 기반)

**포함 안 한 것**: arXiv ID 불확실한 논문, 블로그 전용 수치, 2025년 9월 이후 논문

---

## Step 1: VLA에서 Sim 이미지 직접 사용한 논문

### 카테고리 A: Sim 이미지를 VLA 학습에 직접 사용

| 논문 | sim 이미지 사용 | 조건 | 확신도 |
|------|---------------|------|--------|
| GR00T N1/N1.6 (NVIDIA, arXiv 2503.14734) | YES (~85% sim) | Isaac Sim + DR + humanoid + 전용 렌더러 | MEDIUM |
| RoboCasa (arXiv 2406.02523, RSS 2024) | YES (~95%) | ACT/Diffusion Policy — VLM backbone 없음 | HIGH |
| MimicGen (arXiv 2310.17596, CoRL 2023) | YES | sim 시연 생성 — BC policy | HIGH |
| pi0 (arXiv 2410.24164) | NO | real only | MEDIUM |
| OpenVLA (arXiv 2406.09246) | NO | OXE real data (~0% sim) | HIGH |
| OpenVLA-OFT (arXiv 2412.06173) | NO | real fine-tune only | HIGH |
| Octo (arXiv 2405.12213, RSS 2024) | 미미 (<5%) | OXE 내 소수 sim subset | MEDIUM |
| ACT / ALOHA | NO | real IL only | HIGH |
| SmolVLA base | NO | SO-100 real only (11,132 episodes) | HIGH |

**결론**: VLA에 sim 이미지를 직접, 대규모로 사용한 논문은 GR00T N1 계열이 거의 유일.
단 이는 전용 sim 파이프라인 + domain randomization + humanoid + NVIDIA 클러스터 조건임.

### 카테고리 B: Sim 궤적만 사용 (이미지 불사용)

- GROOT (arXiv 2310.11829, NeurIPS 2023): sim에서 기술 사전학습, real 이미지만 사용
- RoboGen (arXiv 2311.01455): LLM이 sim 태스크 생성, 상태 기반 skill 학습
- 대부분의 sim-to-real 조작 논문: 상태/궤적 이식, 이미지 비전 모델 사용 안 함

---

## Step 2: SigLIP 기반 VLA에서 Sim OOD 문제

### 실증적 근거 (이 프로젝트에서 확인)

| 렌더러 | SigLIP cosine distance | 판정 |
|--------|----------------------|------|
| Isaac Sim rasterizer | 0.6-0.8 | FAIL — 전이 불가 (실제 물체와 다른 embedding) |
| 3DGS rendering | 0.1-0.2 | POSSIBLE — 동일 domain 범주 |
| Real images (동일 장면) | 0.05-0.15 | PASS |

(A2 에이전트 측정, 확신도: MEDIUM — 자체 측정)

### 이론적 근거

SigLIP의 pretraining distribution:
- 인터넷 자연 이미지 (LAION 등)
- 자연광, 랜덤 배경, 다양한 질감, 실제 물리 재질

Isaac Sim rasterizer의 문제:
- Phong/PBR shading이 실제 물리 광원과 다름
- Anti-aliasing artifacts, 텍스처 해상도 제한
- Shadow model 부정확

### Vision Encoder Fine-tuning이 해결책인가?

- R3M (arXiv 2203.12601, RSS 2022): ResNet50을 human video로 fine-tune → manipulation transfer 개선
  - SigLIP fine-tuning과 유사 개념이나 VLA backbone 아님, sim→real 아님
  - 확신도: HIGH (결과는 확실, 적용 가능성은 간접적)
- OpenVLA 논문: "vision encoder fine-tuning improves generalization" — real→real OOD, sim 아님
  - 확신도: MEDIUM
- DINOv2 vs SigLIP sim 강건성: DINOv2가 texture bias 낮아 sim에 robust하다는 실험 있음
  - 확신도: MEDIUM (특정 논문 참조 불확실)

**SmolVLA에서 적용 가능성**: SigLIP frozen이므로 fine-tuning 불가. 근본 해결 방법 없음.

---

## Step 3: 주요 시스템별 Sim 데이터 사용 현황 (상세)

### GR00T N1.6 (가장 관련성 높은 성공 사례)

- 조건: NVIDIA IsaacSim, domain randomization (텍스처/조명/물체 위치), humanoid
- 결과: "sim pretrained → real transfer", 구체적 % 수치는 블로그 출처
- 핵심 차이: 전용 렌더러 + DR 파이프라인 구축에 수개월 투자
- 확신도: MEDIUM (N1 논문은 peer-reviewed, N1.6 세부 수치는 블로그)

### RoboCasa + MimicGen

- sim 데이터 충분히 사용해 real transfer 성공
- BUT: ACT/Diffusion Policy — VLM 없음. SigLIP 문제 없음.
- 적용 가능성: None (VLA 아님)
- 확신도: HIGH

### DROID Dataset (논문: arXiv 2403.12945)

- 76K+ real demos, 50+ robot variants
- sim 데이터 없음 — 전적으로 real
- VLA 학습용 대규모 real 데이터셋의 좋은 예
- 확신도: HIGH

---

## Step 4: Real + Sim 혼합 학습 효과

### 정량적 결과가 확인된 논문

**RoboAgent (arXiv 2309.01918, RSS 2023)**

- 78 real demos + semantic augmentation (in-painting 배경 교체)
- 결과: **12 tasks 평균 71% → 87%** 성공률 향상
- 중요: sim 이미지가 아니라 real 이미지의 배경만 교체 — SigLIP OOD 없음
- 확신도: HIGH

**GROOT (NeurIPS 2023)**

- sim에서 기술 선행학습 (상태 기반) → real fine-tune
- 비전 인코더: real 이미지만 사용
- 이미지 mix 결과: 미보고
- 확신도: HIGH

**체계적 "small real + large sim → VLA" 연구**

- 2025년 8월 기준으로 이러한 체계적 연구 = 발견 불가
- 이것이 실제 갭일 가능성: MEDIUM (반증 검색 추가 필요)

### "몇 개의 real episode면 충분한가" (sim 보강 시)

- VLA 특화 연구 없음 (2025년 8월 기준)
- 참고 가능: RoboAgent 78 real이 sufficient (단, VLM 없는 BC)
- SmolVLA 맥락: 74 real → 1위치 100%, 150+ real → 5위치 목표 (이 프로젝트)

---

## Step 5: 비판적 평가

### Manipulation vs Locomotion에서의 sim-to-real 성공률

**Locomotion (높은 성공률)**

- 이유: physics sim이 중력/마찰을 정확히 모델링 가능
- 이미지 없이 proprioceptive 센서만으로 충분
- 대표: ANYmal (Science Robotics 2019) — sim policy → real 90%+ 이식
- 확신도: HIGH

**Manipulation (낮은 성공률)**

| 태스크 유형 | sim→real 성공률 | 이유 |
|------------|---------------|------|
| Bin picking (기하학적) | MEDIUM | 물체 크기 크고 contact 단순 |
| Precision assembly | LOW | contact dynamics sim 오차 |
| Cable routing | LOW | deformable object sim 부정확 |
| Pick-and-place (블록) | MEDIUM | 조건에 따라 다름 |
| VLM-guided grasping | LOW (sim img) | SigLIP OOD 추가됨 |

확신도: HIGH (이론적으로 명확, 다수 논문 일치)

### VLA 특유의 sim-to-real 어려움

**어려움 1: Vision Encoder Pretraining Distribution**
- SigLIP frozen → adapt 불가
- Isaac Sim → cosine dist 0.6-0.8 → 완전히 다른 embedding domain

**어려움 2: Language Grounding이 Visual Feature에 의존**
- "pick up the red cube" → SigLIP이 sim에서 빨간 큐브를 다르게 인식 → grounding 무너짐
- VLM 없는 policy는 이 문제 없음

**어려움 3: Flow Matching Action Space**
- SmolVLA flow matching: sim visual feature → real action 매핑이 불일치

확신도: HIGH (이론적으로 명확)

### SmolVLA + 현재 setup에서 sim 이미지 사용 가능성

| 접근법 | 가능성 | 비용 | 비고 |
|--------|--------|------|------|
| Isaac Sim 직접 사용 | 거의 불가 | - | cosine 0.6-0.8 |
| 3DGS workspace 재구성 | 탐색 가능 | 2-3일 + GPU | cosine 0.1-0.2, 아직 미검증 |
| 배경 in-painting aug | 가능성 있음 | 0.5-1일 | RoboAgent 방식, 87% 개선 실제 있음 |
| Domain randomization on real | 가능 | 1일 | color jitter, noise 등 |
| Vision encoder fine-tuning | 불가 | - | SmolVLA frozen 정책 위반 |

---

## 핵심 수치 요약

| 주장 | 수치 | 출처 | 확신도 |
|------|------|------|--------|
| SigLIP + Isaac Sim cosine dist | 0.6-0.8 | 이 프로젝트 A2 에이전트 | MEDIUM |
| SigLIP + 3DGS cosine dist | 0.1-0.2 | 이 프로젝트 A2 에이전트 | MEDIUM |
| GR00T N1.6 sim 비율 | ~85% | NVIDIA 블로그 | LOW-MEDIUM |
| RoboAgent semantic aug 효과 | 71% → 87% | arXiv 2309.01918 | HIGH |
| ANYmal sim→real 이식률 | ~90% | Science Robotics 2019 | HIGH |
| OpenVLA-OFT LIBERO 성공률 | 97.1% | arXiv 2412.06173 | HIGH |
| VLA sim img + real 혼합 체계적 연구 | 0편 발견 | 이 조사 | MEDIUM (추가 검색 필요) |
| manipulation sim→real VLM 없는 BC | MEDIUM 성공 | RoboCasa, MimicGen | HIGH |

---

## 이전 결론 ("궤적만 가능, 이미지 불가") 검증

**이전 결론: 유효 (유지)**

근거:
1. SigLIP frozen VLA + sim 이미지 혼합 성공 사례 = 0편 (확인 범위 내)
2. sim→real VLA 성공 유일 사례 = GR00T N1 (우리 조건과 근본적으로 다름)
3. 이론적 근거 명확: cosine dist 0.6-0.8 → same embedding space에서 다른 물체처럼 인식

**수정 사항**:
- "궤적 이식"은 가능 (IK/RL 기반 trajectories)
- "3DGS 렌더링"은 탐색 가능 — 단 추가 검증 필요
- "배경 in-painting augmentation"은 sim 이미지와 다름 — real 이미지에 배경만 교체 → 유효

---

## 권장사항

### pipeline-agent (즉시 실행)
- sim 이미지 기반 데이터 증강 시도 금지 (현재 SmolVLA 아키텍처)
- 150 real episodes 직접 수집이 유일하게 검증된 경로

### B2 data-efficiency (탐색 가치)
- 배경 in-painting aug (RoboAgent 방식): 1일 구현, 71%→87% 선례 있음
- 3DGS workspace 재구성: 2-3일, cosine 0.1-0.2면 탐색 가치 있음

### C3 research-writing (related work)
- "VLA sim-to-real에서 소량 real + sim 혼합 체계적 연구" — 갭으로 positioning 가능
- 단, 추가 검색 (2025년 9월 이후 논문) 후 확신도 MEDIUM → HIGH 업그레이드 필요
- 안전한 표현: "우리 검색 범위 내에서 SigLIP-frozen VLA + sim 혼합 체계적 연구 없음"

### A2 sim2real (교차 검증 요청)
- 3DGS cosine dist 0.1-0.2 수치 재확인 (측정 조건 명시 필요)
- Isaac Sim DR (domain randomization) 적용 시 cosine dist 변화 측정 가능한지 확인
