# VLA Research Ideas

석사 논문을 위한 VLA(Vision-Language-Action) 연구 아이디어 정리.

## Candidate 1: CC-VLA (Confidence-Conditioned VLA) ⭐

### Why Novel?
- **Confidence Calibration in VLA (2025.07, arxiv 2507.17383)**: confidence를 분석/보정만 함
- 논문 원문: "우리는 모델의 동작을 변경하지 않고 보고된 신뢰도만 수정합니다"
- **Gap**: Confidence를 action generation의 **조건(condition)**으로 사용하는 연구 없음!

### Core Formulation

```python
# 기존 VLA
a = π(o, l)                      # observation, language → action
L = ||a - a*||²                  # MSE loss

# CC-VLA (제안)
c = σ(MLP_conf(Enc(o, l)))       # Confidence 예측 (0~1)
a = Dec(z, embed(c))             # Confidence-conditioned action generation

# Heteroscedastic Loss
L = ||a - a*||² / c² + α*log(c) + λ*BCE(c, success)
#    ↑ 불확실하면      ↑ 너무 낮은   ↑ 실제 성공과
#    loss 가중치 낮춤   confidence    confidence 정렬
#                      방지
```

### Key Contributions
1. **Confidence as Explicit Condition**: action generation에 confidence를 명시적 입력으로
2. **Heteroscedastic Loss**: 불확실할 때 자동으로 보수적 행동 유도
3. **Human Handoff Decision**: confidence threshold로 사람 개입 시점 결정
4. **Calibrated Uncertainty**: post-hoc이 아닌 학습 중 직접 calibration

### Related Work
| Paper | What they do | Gap |
|-------|-------------|-----|
| Confidence Calibration in VLA | Post-hoc confidence 분석 | 행동 변경 안함 |
| Heteroscedastic Regression | 일반 회귀에서 uncertainty | VLA 미적용 |
| Ensemble Methods | 다중 모델 variance | 계산 비용 높음 |

---

## Candidate 2: R-VLA (Reversibility-aware VLA) ⭐⭐

### Why Novel?
- **Safe RL에서 reversibility 연구**: Google의 RAE/RAC ([Self-Supervised Reversibility-Aware RL](https://research.google/blog/self-supervised-reversibility-aware-reinforcement-learning/))
- **ReVLA는 다른 개념**: "reversible training" (메모리 효율), action reversibility 아님
- **Gap**: Action의 reversibility를 VLA의 **조건/필터**로 사용하는 연구 없음!

### Core Formulation

```python
# Reversibility Predictor (Google RAE/RAC에서 영감)
# Event A가 B 전에 일어나는지 예측 → reversibility proxy
r = R_θ(o, a)  # reversibility score (0=irreversible, 1=fully reversible)

# Option 1: Action Filtering (RAC 스타일)
while R_θ(o, a_sampled) < threshold:
    a_sampled = π(o, l)  # resample until reversible
a = a_sampled

# Option 2: Loss Weighting
L = (1 + β*(1-r)) * ||a - a*||²
#   ↑ 불가역 행동에 더 높은 loss → 보수적 학습

# Option 3: Condition으로 사용
a = π(o, l, embed(r))  # reversibility-aware action
```

### Key Contributions
1. **Safe RL → VLA 적용**: Google의 reversibility 개념을 VLA에 최초 적용
2. **Action Filtering**: 불가역 행동 사전 차단
3. **Risk-aware Manipulation**: 물체 파손, 충돌 등 irreversible 상황 회피
4. **CC-VLA와 상호보완**: CC는 모델 불확실성, R은 환경 불가역성

### vs CC-VLA
| 측면 | CC-VLA | R-VLA |
|------|--------|-------|
| 측정 대상 | 모델의 예측 불확실성 | 환경/행동의 불가역성 |
| 관점 | Epistemic (내가 모르는 것) | Aleatoric (세상의 특성) |
| 응용 | Human handoff | Catastrophe prevention |

---

## Candidate 3: CR-VLA (Confidence + Reversibility VLA) ⭐⭐⭐

### Why Novel?
- CC-VLA + R-VLA 통합
- 두 가지 orthogonal한 safety signal
- **Gap**: 모델 불확실성 + 환경 불가역성을 동시에 고려하는 VLA 없음

### Core Formulation

```python
# 두 가지 safety score
c = σ(MLP_conf(Enc(o, l)))    # Confidence (model uncertainty)
r = R_θ(o, a)                  # Reversibility (environment property)

# Safety score 통합
safety = c * r  # 둘 다 높아야 안전

# Safety-conditioned action
a = π(o, l, embed(safety))

# Unified Loss
L = ||a - a*||² / (c * r)² + α*log(c) + β*log(r) + λ*BCE(c, success)
```

### Safety Matrix
|              | r 높음 (가역) | r 낮음 (불가역) |
|--------------|-------------|---------------|
| **c 높음 (확신)** | ✅ 실행 | ⚠️ 주의 실행 |
| **c 낮음 (불확실)** | 🔄 탐색 가능 | ❌ 거부/handoff |

### Key Contributions
1. **Dual Safety Signal**: epistemic + aleatoric uncertainty 통합
2. **Principled Decision**: 언제 실행/거부/handoff 할지 명확한 기준
3. **Novel Loss Function**: 두 safety 요소를 통합한 heteroscedastic loss

---

## Candidate 4: Social-Intent VLA (SI-VLA)

### Idea
- 주변 사람의 의도(intent) 예측 → 협력적 행동
- "저 사람이 뭘 하려는지" 이해하고 돕기

### Status: ⚠️ 부분 포화
- Human intent prediction for HRI: 이미 연구 활발 ([Frontiers article](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1708987/full))
- Figure Helix: 두 로봇이 협력하는 데모 존재
- VLA에 직접 통합은 gap 있을 수 있으나, 연구 방향 겹침

---

## Candidate 5: Modular Skill VLA (MS-VLA)

### Idea
- Task를 skill로 분해 → skill composition
- "Pick and place" = Pick + Place

### Status: ❌ 포화
- NVIDIA GR00T N1: System 1 + System 2 dual architecture
- Agentic VLA: LLM planner + VLA skills
- Hierarchical decomposition 이미 활발

---

## Exploration Log

### 2026-02-06: 초기 탐색
- Anticipatory VLA → UP-VLA, HiF-VLA 존재 (Red Ocean)
- Memory-Augmented → MemoryVLA, MAP-VLA, EchoVLA (Saturated)
- Cross-Embodiment → ET-VLA, X-VLA (Saturated)

### 발견된 Gap (2차 검증 완료)
1. **Confidence Conditioning** ← CC-VLA ⭐ (가장 선명한 gap)
2. **Reversibility-aware** ← R-VLA (VLA 미적용 확인)
3. **Confidence + Reversibility 통합** ← CR-VLA (Safety VLA 경쟁 치열)

### 포화된 방향
- Style/Personalization → GRAPE (2024.11)가 이미 preference alignment
- Skill Composition → GR00T N1, Agentic VLA
- Human Intent → HRI 분야에서 활발
- CoT Reasoning → CoT-VLA, FlowVLA, CoA-VLA 다수
- **추가 Modality → 완전 포화!** (아래 참조)

### 추가 Modality 현황 (2025-2026, 전부 있음!)
| Modality | 논문 | 비고 |
|----------|------|------|
| Tactile | Tactile-VLA | 촉각 센서 |
| Audio | Audio-VLA | 접촉 소리 |
| Force/Torque | ForceVLA, TA-VLA | 6축 힘센서, 토크 |
| Speech | SVA (Speech-VLA) | 음성 명령 |
| Depth | 다수 | Point cloud 통합 |
| Thermal | OmniSegmentor | 열화상 |

### Safety VLA 경쟁 현황
| 논문 | 접근 방식 | CC-VLA와의 차이 |
|------|----------|----------------|
| SafeVLA (NeurIPS 2025 Spotlight) | CMDP, unsafe 분류 | Binary constraint |
| VLSA/AEGIS | Control Barrier Function | Physics constraint |
| CompliantVLA | Variable Impedance | 접촉 순응 |
| Confidence Cal (2507.17383) | Post-hoc 보정만 | 행동 변경 안함 |
| **CC-VLA (제안)** | **Heteroscedastic conditioning** | **Continuous self-awareness** |

---

## 최종 추천 순위 (4차 검증 - Manipulation 특화 재탐색)

### 4차 검증 결론: VLA 분야는 거의 모든 방향이 탐색됨
2025-2026에 폭발적으로 성장하여 "___-VLA" 형태의 대부분의 아이디어가 이미 존재.
완전히 새로운 패러다임보다는 **기존 기법의 깊이 있는 분석 + 새로운 조합 + 실제 로봇 실험**이 현실적.

### 4차 탐색에서 확인한 추가 포화 방향
| 방향 | 기존 연구 | 상태 |
|------|----------|------|
| Object-Centric | Oat-VLA, OmniManip (CVPR 2025), ObjectVLA | ❌ 포화 |
| 3D Spatial | SpatialVLA, GraphCoT-VLA, Spatial Forcing | ❌ 포화 |
| World Model | WorldVLA, Cosmos Policy, WALL-A | ❌ 포화 |
| Failure Recovery | FailSafe-VLM | ❌ 존재 |
| Dynamic Object | DynamicVLA (2601.22153) | ❌ 존재 |
| Grasping 특화 | GraspVLA, VLA-Grasp, DexGraspVLA | ❌ 포화 |
| Tactile+VLA | VTLA (peg insertion), Tactile-VLA | ❌ 포화 |
| Ensemble/Voting | VOTE (2507.05116), PI-VLA AURD | ❌ 존재 |
| Tree of Thoughts | Embodied ToT (2512.08188), MCTS | ❌ 존재 |
| LoRA Fine-tuning | LoRA-VLA (2512.11921), OpenVLA-OFT | ❌ 포화 |
| RL Fine-tuning | VLA-RL, VLA-RFT | ❌ 포화 |
| Action Chunking | PD-VLA, FAST, RTC, VLA-Cache | ❌ 포화 |
| Counterfactual | RoCoDA (data augmentation만) | ⚠️ VLA 직접 적용은 없음 |

### CC-VLA 약점 인정 (3차에서 유지)
- Heteroscedastic regression은 이미 잘 알려진 기법 (Kendall & Gal, 2017)
- "기법 적용" 수준이지 "새로운 패러다임"이라 하기엔 약함
- Reviewer: "SafeVLA랑 뭐가 다름?" 에 대한 명쾌한 답 필요
- PI-VLA의 AURD도 action disagreement로 uncertainty 감지 (유사)
- **단독으로는 top venue 어려움, 보강 필요**

### 3차 탐색에서 추가로 포화 확인된 방향
| 방향 | 기존 연구 | 상태 |
|------|----------|------|
| Test-Time Adaptation | TT-VLA, HyperVLA, PLD | ❌ 포화 |
| Test-Time Compute Scaling | SITCOM, Dynamic TTC | ❌ 포화 |
| Language-to-Impedance | HumanoidVLM, OmniVIC, ImpedanceGPT | ❌ 포화 |
| In-Context Learning | MoS-VLA, RoboPrompt, DEMONSTRATE | ⚠️ 활발 |
| Data Quality | Consistency Matters, PLD, EMMA | ⚠️ 시작됨 but gap 있음 |

### ICLR 2026 전문가가 지적한 Gap (Moritz Reuss 블로그)
1. **Data Quality**: "surprisingly few submissions focused on data curation"
2. **In-Context Learning**: "expected more work here but found almost none"
→ 하지만 2차 검색에서 ICL 논문 다수 발견 (MoS-VLA 등)

---

## 현실적 추천 순위 (4차 최종)

| 순위 | 후보 | 강점 | 약점 | 목표 venue |
|------|------|------|------|-----------|
| **1** | **CC-VLA → SA-VLA (Self-Aware VLA)** | 명확한 gap, 구현 가능, 프레임워크로 확장 | SafeVLA/PI-VLA와 차별화 필수 | Journal / Workshop |
| **2** | **DQ-VLA (Data Quality)** | 전문가 인정 gap, 실험 바로 가능 | "Consistency Matters"와 차별화 필요 | Conference / Journal |
| 3 | **CA-VLA (Counterfactual-Augmented)** | VLA 직접 적용 없음 (gap!) | 시뮬레이터 필요, 구현 복잡 | Conference |
| 4 | R-VLA | VLA 미적용 확인 | 실험 데이터 구축 어려움 | Journal |

### 1위: SA-VLA (Self-Aware VLA) - CC-VLA의 확장

CC-VLA를 단순 confidence 논문이 아닌 **종합 프레임워크**로 확장:

```python
# SA-VLA: Self-Aware VLA Framework

# Module 1: Confidence Estimation (CC-VLA 핵심)
c = σ(MLP_conf(Enc(o, l)))       # Confidence (0~1)

# Module 2: Phase Detection
phase = PhaseClassifier(o, l)     # approach/align/grasp/transport/place

# Module 3: OOD Detection
ood_score = MahalanobisOOD(z)     # 학습 분포에서 벗어난 정도

# Module 4: Unified Safety Score
safety = f(c, phase, ood_score)   # 통합 safety score

# Module 5: Adaptive Action Generation
if safety > τ_execute:
    a = π(o, l, embed(safety))    # 정상 실행
elif safety > τ_cautious:
    a = π_slow(o, l, embed(safety))  # 보수적 실행 (느린 속도)
else:
    a = STOP + request_human_help()   # 거부 + 사람 요청

# Heteroscedastic Training Loss
L = ||a - a*||² / c² + α*log(c) + λ*BCE(c, success) + γ*CE(phase, phase*)
```

**SafeVLA/PI-VLA와의 차별화:**
| 측면 | SafeVLA | PI-VLA | SA-VLA (제안) |
|------|---------|--------|--------------|
| 안전 신호 | Binary (safe/unsafe) 분류 | Action disagreement | Continuous confidence |
| 방법 | CMDP constraint | Multiple sampling + voting | Single-pass estimation |
| 비용 | 별도 safety classifier 필요 | Multiple forward passes | 단일 forward pass |
| Phase 인식 | 없음 | Symmetry-aware | Phase-explicit |
| OOD 감지 | 없음 | 없음 | Mahalanobis distance |
| 핵심 차이 | 외부 제약 (external) | 다중 샘플 합의 | 내부 자각 (internal) |

### 2위: DQ-VLA (Data Quality-aware VLA)

```python
# 기존: 모든 데모를 동등하게 학습
L = Σ ||a_i - a*_i||²

# DQ-VLA: 데모 품질에 따라 가중 학습
q_i = QualityScorer(demo_i)      # 데모 품질 자동 평가
L = Σ q_i * ||a_i - a*_i||²     # 고품질 데모에서 더 많이 학습

# + Curriculum: 쉬운(고품질) → 어려운(저품질) 순서로
```

차별화 vs "Consistency Matters":
- 그들: metric 정의에 초점
- DQ-VLA: metric → 실제 학습에 반영 (quality-weighted training)

차별화 vs PLD:
- PLD: RL로 새 데이터 자동 생성
- DQ-VLA: 기존 데이터의 품질 기반 커리큘럼

차별화 vs EMMA:
- EMMA: challenging sample reweighting (어려운 샘플 가중)
- DQ-VLA: quality scoring + curriculum (품질 측정 + 순서 학습)

### 3위: CA-VLA (Counterfactual-Augmented VLA)

```python
# 기존: demonstration 그대로 학습
(o_t, a_t, o_{t+1}) → L = ||π(o_t) - a_t||²

# CA-VLA: counterfactual 경험도 학습
# Step 1: 원래 trajectory에서 action을 perturbation
a_cf = a_t + δ                    # counterfactual action

# Step 2: 시뮬레이터/world model로 결과 예측
o_cf = WorldModel(o_t, a_cf)      # counterfactual outcome

# Step 3: 원래 vs counterfactual 비교 학습
L = ||π(o_t) - a_t||² + β * max(0, margin - d(o_{t+1}, o_cf))
#   원래 행동 모방       + counterfactual이 나쁘다는 것도 학습
```

차별화 vs RoCoDA:
- RoCoDA: scene-level augmentation (배경, 물체 위치 변경)
- CA-VLA: action-level counterfactual (행동 변경의 결과 학습)

---

## Exploration Log

### 2026-02-06: 4차 탐색 (Manipulation 특화)
검색 범위를 "로봇 팔 manipulation + VLA"로 좁혀서 재탐색.
- Object-centric, Spatial, World Model, Grasping, Dynamic → 전부 존재
- Action chunking, LoRA fine-tuning, RL fine-tuning → 전부 포화
- Counterfactual reasoning: VLA 직접 적용은 없음 → CA-VLA gap 확인
- PI-VLA (AURD): action disagreement 기반 uncertainty 감지 → CC-VLA와 유사점 발견
- VOTE: ensemble voting → Self-Consistency 아이디어 이미 존재
- Embodied ToT: Tree of Thoughts 로봇 적용 이미 존재
- 최종 결론: 완전히 새로운 패러다임 어려움, **깊이 있는 프레임워크 + 실험**이 현실적

---

## 5차 탐색: 3DGS + CG/XR + Robot 교차 분야 (2026-02-06)

### 배경
교수님 분야(Gaussian Splatting, Computer Graphics, XR)와 로봇 manipulation을 결합하는 방향으로 전환.
VLA 자체의 novelty는 포화 상태이므로, **3DGS/CG/XR 기술을 로봇에 적용**하는 교차 연구를 탐색.

### 3DGS + Robot 기존 연구 지도

| 방향 | 기존 연구 | 포화도 |
|------|----------|--------|
| 3DGS → Grasping/Affordance | GaussianGrasper, GraspSplats (CoRL'24), Splat-MOVER | ⚠️ 활발 |
| 3DGS → Data Augmentation | RoboSplat (RSS'25), R2R2R (CoRL'25) | ⚠️ 활발 |
| 3DGS → World Model | GWM (ICCV'25), ManiGaussian (ECCV'24) | ⚠️ 존재 |
| 3DGS → Sim2Real | SplatSim (ICRA'25), RoboGSim | ⚠️ 존재 |
| 3DGS → VR Teleoperation | Human-in-the-Loop GS (RAL'25) | ⚠️ 존재 |
| 3DGS → Policy Evaluation | Real-to-Sim Policy Eval with GS (2511.04665) | ⚠️ 존재 |
| 3DGS → RL Representation | GSRL | ⚠️ 시작됨 |
| 3DGS → Self-Correction | GS-Splatted Foresight (AAAI'25) | ⚠️ 존재 |
| 3DGS → Object Tracking | POGS, Object-Aware GS | ⚠️ 존재 |
| Single-view 3DGS | SVG3D, SPAGS, SIGMA | ⚠️ 활발 |
| 3DGS → Long-term Service | GS-LTS | 존재 |
| 3DGS → SLAM | SemGauss-SLAM (IROS'25), RGBDS-SLAM (RAL'25) | ⚠️ 활발 |

### 5차 최종 후보

#### 1위: Depth-GS-Aug (Depth-Guided 3DGS for Low-Cost Robot Data Augmentation)

**핵심**: Azure Kinect 1대 (RGB-D) → depth-guided 3DGS 재구성 → novel view + object/lighting variation → policy training data 증강

```
[Azure Kinect RGB-D] → [Depth-Guided 3DGS] → [Scene Editing] → [Novel Views] → [Policy Training]
   single camera         few-shot recon        object/light       augmented       improved
   (우리 장비)            depth prior 활용       variation          demonstrations   performance
```

**차별화:**
| 측면 | R2R2R (Berkeley) | RoboSplat (RSS'25) | Depth-GS-Aug (제안) |
|------|-----|---------|---------|
| Input | Multi-view smartphone scan | Multi-view images | **Single RGB-D camera** |
| 물리 시뮬레이터 | IsaacLab 필요 | 불필요 | **불필요** |
| Object scan | 별도 필요 | 불필요 | **불필요** |
| 타겟 | 대형 랩 | 대형 랩 | **소규모 연구실** |
| Depth 활용 | 간접적 | 없음 | **Depth prior for few-shot 3DGS** |

**구현 파이프라인:**
1. Azure Kinect로 manipulation scene 촬영 (RGB + Depth)
2. Depth-guided few-shot 3DGS reconstruction (SVG3D/FSGS 기반)
3. Scene editing: object pose randomization, lighting variation, background change
4. Novel view rendering → synthetic demonstration 생성
5. 원본 데이터 + augmented 데이터로 policy 학습 (Diffusion Policy or VLA)
6. 실제 로봇에서 성능 비교 평가

**교수님 분야 적합도**: ★★★★☆ (3DGS + rendering + depth reconstruction)
**구현 난이도**: ★★★☆☆ (기존 3DGS 라이브러리 활용 가능)
**장비 적합**: ★★★★★ (Azure Kinect, RTX 4070 Ti, RoArm M3 모두 활용)

#### 2위: GS-Progress (3DGS-based Manipulation Task Progress Estimation)

**핵심**: manipulation 작업의 before/during/after를 3DGS로 복원 → 3D geometry 변화 → task progress metric

```
[Scene t=0] → [3DGS_0] ─┐
[Scene t=T] → [3DGS_T] ─┤→ [Gaussian Distance] → [Progress Score]
                          └→ [3D Change Map]    → [Visualization]
```

**차별화:**
- 기존: 2D image change detection for task eval (많음)
- 기존: Real-to-Sim Policy Eval (2511.04665): soft-body 특화
- 제안: **3DGS geometry 변화량으로 general manipulation progress를 정량화**

**교수님 분야 적합도**: ★★★★☆ (3DGS scene understanding)
**구현 난이도**: ★★☆☆☆ (비교적 간단)
**논문 임팩트**: ★★★☆☆ (metric 논문, contribution 작을 수 있음)

#### 3위: GS-XR-Demo (3DGS + XR for Interactive Robot Demo Generation)

**핵심**: 3DGS로 환경 복원 → XR(AR/VR)에서 사용자가 scene 편집 → 편집된 3DGS → 새 demonstration 생성

```
[Real Scene] → [3DGS Recon] → [XR Environment] → [Interactive Edit] → [New Demo]
 Azure Kinect    복원            HMD/AR display    물체 이동/추가       policy 학습용
```

**차별화:**
- RoboSplat: 자동 augmentation
- Human-in-the-Loop GS (RAL'25): real-time teleop (같은 공간)
- **제안: interactive XR editing으로 diverse demonstration 생성 (원격 가능)**

**교수님 분야 적합도**: ★★★★★ (XR + 3DGS 완벽 매칭)
**구현 난이도**: ★★★★☆ (XR 헤드셋 필요, 구현 복잡)
**논문 임팩트**: ★★★★☆ (교차 분야 novelty 높음)
**리스크**: XR 장비 보유 여부, 구현 시간

### 5차 탐색 결론

VLA 자체보다 **3DGS/CG + Robot 교차 분야**가 더 현실적:
1. 교수님이 3DGS/XR 전문가이므로 지도 가능
2. 3DGS + Robot은 VLA 대비 아직 교차 조합의 여지 있음
3. 보유 장비(Azure Kinect, RTX 4070 Ti, RoArm M3)를 모두 활용 가능
4. 졸업 논문 수준으로 적절한 scope

**교수님과 상의할 때 제안 순서:**
1. 교수님이 XR 장비 있으면 → GS-XR-Demo (3위지만 교수님 분야 최적)
2. XR 장비 없으면 → Depth-GS-Aug (1위, 가장 안전)
3. 빠른 논문 필요 → GS-Progress (2위, 구현 가장 빠름)

---

## 5차 탐색 References (3DGS + Robot)

### Data Augmentation / Sim2Real
- RoboSplat (RSS 2025): [arxiv 2504.13175](https://arxiv.org/abs/2504.13175)
- Real2Render2Real (CoRL 2025): [arxiv 2505.09601](https://arxiv.org/abs/2505.09601)
- SplatSim (ICRA 2025): [arxiv 2409.10161](https://arxiv.org/abs/2409.10161)
- RoboGSim: [arxiv 2411.11839](https://arxiv.org/abs/2411.11839)

### World Model / Scene Understanding
- GWM (ICCV 2025): [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.pdf)
- ManiGaussian (ECCV 2024): [GitHub](https://github.com/GuanxingLu/ManiGaussian)
- Self-Correcting via GS Foresight (AAAI): [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34866)
- SceneSplat (ICCV 2025): [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SceneSplat_Gaussian_Splatting-based_Scene_Understanding_with_Vision-Language_Pretraining_ICCV_2025_paper.pdf)

### Grasping / Affordance
- GaussianGrasper: [arxiv 2403.09637](https://arxiv.org/abs/2403.09637)
- Splat-MOVER: [arxiv 2405.04378](https://arxiv.org/abs/2405.04378)
- POGS: [Berkeley PDF](https://autolab.berkeley.edu/assets/publications/media/2025-ICRA-POGS-CRv5.pdf)

### Teleoperation / XR
- Human-in-the-Loop GS Teleoperation (RAL 2025)
- Communication Efficient Robotic MR with GS: [arxiv 2508.08624](https://arxiv.org/abs/2508.08624)
- OpenVR Teleoperation: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352711025000214)
- ScaleGS (XR rendering): [ACM TACO](https://dl.acm.org/doi/10.1145/3774425)

### Policy Learning / RL
- GSRL: [arxiv 2404.07950](https://arxiv.org/abs/2404.07950)
- 3D Diffusion Policy (DP3, RSS 2024): [arxiv 2403.03954](https://arxiv.org/abs/2403.03954)
- DP4 (ICCV 2025): [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Spatial-Temporal_Aware_Visuomotor_Diffusion_Policy_Learning_ICCV_2025_paper.pdf)
- Real-to-Sim Policy Eval with GS: [arxiv 2511.04665](https://arxiv.org/abs/2511.04665)

### Single/Few-shot 3DGS
- SVG3D: [Nature Scientific Reports](https://www.nature.com/articles/s41598-025-03200-7)
- SPAGS: [arxiv 2511.17092](https://arxiv.org/abs/2511.17092)
- FSGS: [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72933-1_9)
- Next Best Sense (Stanford): [Stanford ARM](https://arm.stanford.edu/next-best-sense)

### SLAM
- SemGauss-SLAM (IROS 2025)
- RGBDS-SLAM (RAL 2025)
- Multi-robot 3D recon with GS (RAL 2025)

### Surveys
- 3DGS in Robotics Survey: [arxiv 2410.12262](https://arxiv.org/abs/2410.12262)
- Awesome 3DGS in Robotics: [GitHub](https://github.com/zstsandy/Awesome-3D-Gaussian-Splatting-in-Robotics)
- Radiance Fields in XR Survey: [arxiv 2508.04326](https://arxiv.org/abs/2508.04326)

---

## References
- Confidence Calibration in VLA: arxiv 2507.17383
- Evaluating Uncertainty in VLA: arxiv 2507.17049
- UP-VLA: arxiv 2501.18867
- HiF-VLA: arxiv 2512.09928
- GRAPE (Preference Alignment): arxiv 2411.19309
- SafeVLA (NeurIPS 2025): arxiv 2503.03480
- VLSA/AEGIS (Safety Constraint): arxiv 2512.11891
- CompliantVLA: arxiv 2601.15541
- VLAC (Critic): arxiv 2509.15937
- Tactile-VLA: arxiv 2507.09160
- Audio-VLA: arxiv 2511.09958
- ForceVLA: OpenReview
- TA-VLA (Torque-aware): arxiv 2509.07962
- SVA (Speech-VLA): ScienceDirect
- CRT (Corruption Restoration): arxiv 2602.01158
- Google RAE/RAC: https://research.google/blog/self-supervised-reversibility-aware-reinforcement-learning/
- GR00T N1: NVIDIA dual-system VLA
